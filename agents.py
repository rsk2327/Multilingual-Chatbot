import os
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



USER_SYSTEM_PROMPT = """You are a translator. Translate the text provided by the user into {user_language}. Output only the 
    translated text. If the text is already in {user_language}, return the user's text as it is.
"""

AYA_AGENT_PROMPT = """Your name is Aya and you are an assistant that answers questions. Only respond if the question is addressed to you (Aya)!! 
If its not specifically addressed to you, respond with 'NULL'.
    

    Obey the following rules while responding : 
    1. Only use the text provided in the context to answer the question.
    2. If you don't know the answer, just say that you don't know. 
    3. Use three sentences maximum and keep the answer concise.
    
    Question: {question} 
    Context: {context} 
    Answer:
"""

SUPERVISOR_AGENT_PROMPT = """
You are a supervisor tasked with managing the chat messages with Aya, an AI assistant. Given a chat
message, identify whether the message was directed to Aya. If so, respond with 'Aya'. Otherwise, 
respond with 'User'

Chat Message : {message}
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

    def __init__(self, llm, user_language):
        self.llm = llm
        self.user_language = user_language
        self.chat_history = []

        system_prompt = USER_SYSTEM_PROMPT.format(user_language = user_language)
    
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", system_prompt,
                ),
                MessagesPlaceholder(variable_name = 'user_text'),
            ]
        )

        self.chain = prompt | llm


    def set_graph(self, graph):
        self.graph = graph

    def send_text(self,text, debug = False):

        message = ChatMessage(message = HumanMessage(content=text), sender = self.user_language)
        inputs = {"messages": [message]}
        output = self.graph.invoke(inputs, debug = debug)
        return output

    def display_chat_history(self, content_only = False):

        for i in self.chat_history:
            if content_only == True:
                print(f"{i.sender} : {i.content}")
            else:
                print(i)

    
    def invoke(self, message:ChatMessage):
        
        output = self.chain.invoke({'user_text':[message]})
        return output
            


    
class AyaAgent(object):

    def __init__(self, llm, retriever):

        self.llm = llm
        self.retriever = retriever
        qa_prompt = ChatPromptTemplate.from_template(AYA_AGENT_PROMPT)
        self.chain = qa_prompt | llm 

    def invoke(self, question : str) -> AIMessage:

        context = format_docs(self.retriever.invoke(question))
        rag_output = self.chain.invoke({'question':question, 'context':context})
        answer = rag_output
        return answer
          
class SupervisorAgent(object):

    def __init__(self, llm):
        
        prompt = ChatPromptTemplate.from_template(SUPERVISOR_AGENT_PROMPT)
        self.chain = prompt | llm

    def invoke(self, message : str) -> str:
        output = self.chain.invoke(message)
        return output.content
    
       

#### AGENT NODE FUNCTIONS #######

def get_aya_node(state, agent, supervisor_agent, name):

    latest_message = state["messages"][-1]  
    agent_check = supervisor_agent.invoke(latest_message.content)

    if agent_check == 'Aya':

        result = agent.invoke(latest_message.content)

        return {
        'messages' : [ChatMessage(result, sender = 'Aya')]
    }
                              
    else:
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



def get_supervisor_node(state, agent : SupervisorAgent):
    
    latest_message = state["messages"][-1]
    
    return {'messages': []} 

