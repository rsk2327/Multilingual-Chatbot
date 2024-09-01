import os
import re
import getpass
from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
import functools
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import DistanceStrategy

from ml_chat.agents.utils import *

USER_SYSTEM_PROMPT = """You are a translator. Translate the text provided by the user into {user_language}. Output only the 
    translated text. If the text is already in {user_language}, return the user's text as it is.
"""

USER_SYSTEM_PROMPT2 = """You are a {user_language} translator, translating a conversation between work colleagues. Translate the message provided by the user into {user_language}. 

    Obey the following rules : 
    1. Only translate the text thats written after 'Message:' and nothing else
    2. If the text is already in {user_language} then return the message as it is.
    3. Return only the translated text
    4. Ensure that your translation uses formal language
    5. If there are HTML 

    Message:
    {message}
"""

AYA_AGENT_PROMPT = """Your name is Aya and you are an assistant that answers questions. Only respond if the question is addressed to you (Aya)!! 
If its not specifically addressed to you, respond with 'NULL'.
    

    Obey the following rules while responding : 
    1. Only use the text provided in the context to answer the question.
    2. If you don't know the answer or the context doesnt have the required info, respond with "Sorry, I don't know that". 
    3. Use three sentences maximum and keep the answer concise.
    
    Question: {question} 
    Context: {context} 
    Answer:
"""

# SUPERVISOR_AGENT_PROMPT = """
# You are a supervisor tasked with managing the chat messages with Aya, an AI assistant. Given a chat
# message, identify whether the message was directed to Aya. If so, respond with 'Aya'. Otherwise, 
# respond with 'User'

# Chat Message : {message}
# """

AYA_SUPERVISOR_PROMPT = """
You are a supervisor tasked with managing the chat messages with Aya, an AI assistant. Given a chat
message, identify whether the message was directed to Aya and if so, what was its intention. 

Respond with one of the below 5 options : 
User : Output this if the message does not address Aya. If the message looks like its being addressed to another user and not to Aya, the AI assistant, then output this.
Query : Output this is the message is asking Aya to answer a question
Save : Output this is the message asks Aya to save a peice of information to the knowledge base
Summarize : Output this if the message asks Aya to summarize the conversation
Simplify : Output this if the message asks Aya to simplify the previous message


Examples
Chat Message: Aya, can you summarize the previous 10 messages
Output:Summarize

Chat Message: Do you know what is the deadline for the submission
Output:User

Chat Message: Aya, do you know what is the deadline for the submission
Output:Query

Chat Message: Can you simplify the last message Aya?
Output:Simplify

Chat Message: Great meeting guys! There were a lot of insightful discussions today
Output:User

Chat Message: I dont know about that but maybe Aya might know it
Output:User

Chat Message: Can you save the last message Aya?
Output:Save

Chat Message: Do we know when the next meeting is supposed to be?
Output:User

Chat Message: Aya, when is the next meeting scheduled for?
Output:Query


Using the above samples as example, interpret the chat message and respond with just the correct option
Chat Message: {message}
Output:
"""

AYA_TRANSLATE_PROMPT = """You are an English translator. Translate the message provided by the user into English.

Obey the following rules : 
    1. Only translate the text thats written after 'Message:' and nothing else
    2. If the text is already in English then return the message as it is.
    3. Return only the translated text
    4. Ensure that your translation uses formal language
    5. Ensure that names of people are also translated, especially the name 'Aya'

    Message:
    {message}
"""

AYA_SUMMARIZE_LENGTH_PROMPT = """ You are tasked with extracting the number of messages that need to be summarized. Given a message, 
extract the number of messages that need to be summarized as mentioned in the message

Obey the following rules : 
1. Output only an integer. If a number is not specified, return 0

Examples
Message: Aya, can you summarize the last 5 messages
Output:5

Message: Aya, please summarize the last 20 texts
Output:20

Message: Aya, summarize the past 12 messages
Output:12

Message:Aya, summarize this chat thread
Output:0

Using the above samples as example, extract the number of messages that need to be summarized from the below message
Message:{message}
Output:
"""

AYA_SUMMARIZER_PROMPT = """
You are AI assistant that summarizes conversations in a chat application. Given message history, provide the following outputs :
Summary : Short summary of the message history
Tasks : Mention any tasks that have been assigned to a particular person

Obey the following rules: 
1. Summary should be 3-4 sentences maximum
2. Make sure to include specifics like numbers, dates, names etc. 
3. Summarize the full message and not just the first and last sentences of the message
4. If there are no tasks mentioned in the message history, return None
5. Always include the 'Summary:' and 'Tasks:' identifiers in the output

Examples
Message History: Hey. How are you doing?
Im doing great. What about you?
Im doing good too. I wanted to ask you about the Tempus project. Whats the status of the project?
The project is currently in a standstill as no one is working on the model development. 
Ohh, I wasnt aware of that. Could you ask Ishaan to pick up the model development task. 
Sure. I can ask Ishaan on that. And do we stick to the timeline of Jan 25.
Yes. we should stick to that timeline
Summary: The discussion focused on the Tempus project and its current status. It was learned that the project is on a standstill because of no progress on the model development. 
It was decided to ask Ishaan to work on model development and to stick to the original timeline of Jan 25
Task: Ishaan was tasked with model development work


Using the above samples as example, summarize and identify tasks from the following message history.
Message History: {message}
Summary:
"""

class ChatMessage(object):

    def __init__(self, message : BaseMessage, sender : str = None):

        self.message = message
        self.sender = sender

        self.content = message.content

    def __repr__(self) -> str:
        
        return f"{self.sender} | {self.content}"

def reducer(a : list, b : list | str ) -> list:

    if type(b) == list: 
        return a + b
    else:
        return a

    
class AgentState(TypedDict):
    messages: Annotated[Sequence[ChatMessage], reducer]

###### AGENT CLASS DEFINITIONS #######


class UserAgent(object):

    def __init__(self, llm, userid, user_language):
        self.llm = llm
        self.userid = userid
        self.user_language = user_language
        self.chat_history = []

        prompt = ChatPromptTemplate.from_template(USER_SYSTEM_PROMPT2)

        self.chain = prompt | llm


    def set_graph(self, graph):
        self.graph = graph

    def send_text(self,text:str, debug = False):

        message = ChatMessage(message = HumanMessage(content=text), sender = self.userid)
        inputs = {"messages": [message]}
        output = self.graph.invoke(inputs, debug = debug)
        print(f"Within send_text. output type : {type(output)}")
        return output

    def display_chat_history(self, content_only = False):

        for i in self.chat_history:
            if content_only == True:
                print(f"{i.sender} : {i.content}")
            else:
                print(i)

    
    def invoke(self, message:BaseMessage) -> AIMessage:
              
        output = self.chain.invoke({'message':message.content, 'user_language':self.user_language})
        print(f"Within {self.userid} {self.user_language}| {output.content}")

        return output

class AyaTranslate(object):

    def __init__(self, llm) -> None:
        self.llm = llm 
        prompt = ChatPromptTemplate.from_template(AYA_TRANSLATE_PROMPT)
        self.chain = prompt | llm 
        
    def invoke (self, message: str) -> AIMessage:
        output = self.chain.invoke({'message':message})
        return output

class AyaQuery(object):

    def __init__(self, llm, store, retriever) -> None:
        self.llm = llm
        self.retriever = retriever
        self.store = store
        qa_prompt = ChatPromptTemplate.from_template(AYA_AGENT_PROMPT)
        self.chain = qa_prompt | llm

    def invoke(self, question : str) -> AIMessage:

        context = format_docs(self.retriever.invoke(question))
        rag_output = self.chain.invoke({'question':question, 'context':context})

        return rag_output

class AyaSupervisor(object):

    def __init__(self, llm):
        
        prompt = ChatPromptTemplate.from_template(AYA_SUPERVISOR_PROMPT)
        self.chain = prompt | llm

    def invoke(self, message : str) -> str:
        output = self.chain.invoke(message)
        return output.content
    
class AyaSummarizer(object):

    def __init__(self, llm):
        
        message_length_prompt = ChatPromptTemplate.from_template(AYA_SUMMARIZE_LENGTH_PROMPT)
        self.length_chain = message_length_prompt | llm 
        
        prompt = ChatPromptTemplate.from_template(AYA_SUMMARIZER_PROMPT)
        self.chain = prompt | llm



    def invoke(self, message : str, agent : UserAgent) -> str:

        length = self.length_chain.invoke(message)

        try:
            length = int(length.content.strip())
        except:
            length = 0

        chat_history = agent.chat_history

        if length == 0:
            messages_to_summarize = [chat_history[i].content for i in range(len(chat_history))]
        else:
            messages_to_summarize = [chat_history[i].content for i in range(min(len(chat_history), length))]
        
        print(length)
        print(messages_to_summarize)

        messages_to_summarize = "\n ".join(messages_to_summarize)
        
        output = self.chain.invoke(messages_to_summarize)
        output_content = output.content 

        print(output_content)

        return output_content


    
class AyaAgent(object):

    def __init__(self, llm, store, retriever):

        self.query_agent = AyaQuery(llm, store, retriever)
        self.translator_agent = AyaTranslate(llm)
        self.supervisor_agent = AyaSupervisor(llm)
        self.summarizer_agent = AyaSummarizer(llm)
        self.store = store
        self.llm = llm
        self.chat_history = []
        
        

    def invoke(self, question : str) -> AIMessage:
        pass


#### AGENT NODE FUNCTIONS #######

def get_aya_node(state, aya, user_knowledge_file, name):

    latest_message = state["messages"][-1]

    # Convert messages into English
    translated_message = aya.translator_agent.invoke(latest_message.content)

    # print(translated_message)

    # Identify the type of task associated with message
    task = aya.supervisor_agent.invoke(translated_message.content)
    
    if task == 'Query':

        result = aya.query_agent.invoke(translated_message.content)

        return {
        'messages' : [ChatMessage(result, sender = 'Aya')]
        }
    elif task == 'Save':

        # TO-DO : Check if there is exists a previous message or not
        previous_message = aya.chat_history[-1]

        append_to_file(user_knowledge_file, previous_message.content)
        ## Adding the previous message text to vector database
        aya.store.add_texts([previous_message.content])   

        return_message = AIMessage("The previous message has been added to the knowledge base")

        return {
                'messages' : [ChatMessage(return_message, sender = 'Aya')]
            }
    elif task == 'Summarize':

        summary_message = aya.summarizer_agent.invoke(translated_message.content, aya)

        return_message = AIMessage(summary_message)
        return {
                'messages' : [ChatMessage(return_message, sender = 'Aya')]
            }

    elif task == 'Simplify':
        return None
    else:
        aya.chat_history.append(translated_message)
        return {'messages':[]}


def get_user_node(state, agent, name):
    
    
    # latest_message : ChatMessage
    latest_message = state["messages"][-1]
    

    result = agent.invoke(latest_message.message)

    # Append the translated result with the sender info
    agent.chat_history.append(ChatMessage(result, sender = latest_message.sender))

    return {
        "messages": [],
    }


def get_supervisor_node(state):
    
    latest_message = state["messages"][-1]
    
    return {'messages': []} 

