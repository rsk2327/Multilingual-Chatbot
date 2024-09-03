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
from ml_chat.agents.prompts import *


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
        
        return output

    def display_chat_history(self, content_only = False):

        for i in self.chat_history:
            if content_only == True:
                print(f"{i.sender} : {i.content}")
            else:
                print(i)

    
    def invoke(self, message:BaseMessage) -> AIMessage:
              
        output = self.chain.invoke({'message':message.content, 'user_language':self.user_language})
        

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

