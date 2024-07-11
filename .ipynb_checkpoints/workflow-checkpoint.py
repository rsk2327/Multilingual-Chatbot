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


from utils import *
from agents import * 



def load_retriever():

    loader = PyPDFLoader("machine_learning_basics.pdf")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    store = FAISS.from_documents(docs, OpenAIEmbeddings(), distance_strategy=DistanceStrategy.COSINE)
    
    retriever = store.as_retriever(search_kwargs={"k": 3})

    return retriever



def generate_agents():

    llm = ChatOpenAI(model="gpt-4", streaming=True)

    retriever = load_retriever()

    french_agent = UserAgent(llm, "French")
    spanish_agent = UserAgent(llm, "Spanish")
    english_agent = UserAgent(llm, "English")
    
    aya_agent = AyaAgent(llm, retriever)
    
    supervisor_agent = SupervisorAgent(llm)
    
    french_node = functools.partial(get_user_node, agent=french_agent, name="French")
    spanish_node = functools.partial(get_user_node, agent=spanish_agent, name="Spanish")
    english_node = functools.partial(get_user_node, agent=english_agent, name="English")
    
    aya_node = functools.partial(get_aya_node, agent = aya_agent, supervisor_agent = supervisor_agent, name ="Aya")
    
    supervisor_node = functools.partial(get_supervisor_node, agent = supervisor_agent)


    workflow = StateGraph(AgentState)
    
    agentList = {'French':{'node':french_node, 'agent':french_agent},
                       'Spanish':{'node':spanish_node, 'agent':spanish_agent},
                       'English':{'node':english_node, 'agent':english_agent},
                       }
    
    
    #Defining nodes
    for agent in agentList:
        workflow.add_node(agent, agentList[agent]['node'])
    
    workflow.add_node("Supervisor", supervisor_node)
    
    #Defining edges
    workflow.add_edge(START, "Supervisor")
    
    for agent in agentList:
        workflow.add_edge("Supervisor", agent)
        workflow.add_edge(agent, END)
    
    
    workflow.add_node("Aya",aya_node) 
    
    def router2(state) :
            
        last_message = state["messages"][-1]
        if last_message.sender == "Aya":
            return "END"
        else:
    
            return 'Aya'
    
    workflow.add_conditional_edges("Supervisor", router2  ,{'Aya':'Aya','END':END})
    
    def router(state) :
        
        last_message = state["messages"][-1]
        if last_message.sender == "Aya":
            return "Supervisor"
        else:
            return 'END'
            
    workflow.add_conditional_edges("Aya", router, {'Supervisor':'Supervisor','END':END})

    app = workflow.compile()

    french_agent.set_graph(app)
    spanish_agent.set_graph(app)
    english_agent.set_graph(app)

    return english_agent, spanish_agent, french_agent



def generate_response(sender, text):

    english_agent, spanish_agent, french_agent = generate_agents()

    agent_map = {'English':english_agent, 'Spanish':spanish_agent,'French':french_agent}

    agent_map[sender].send_text(text)

    output = [english_agent.chat_history[-1].content,
              spanish_agent.chat_history[-1].content,
              french_agent.chat_history[-1].content]

    return output