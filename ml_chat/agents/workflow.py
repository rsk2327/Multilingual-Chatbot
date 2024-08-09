import os
import getpass
from pathlib import Path
import warnings

from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
import functools
from typing import TypedDict, Annotated, Sequence, Tuple, Optional, List, Any
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import DistanceStrategy


from agents.utils import *
from agents.agents import * 



def load_retriever(path : Path, valid_extensions : List[str]):

    # Identifying the valid files
    valid_files = [file for file in path.iterdir() if file.is_file() and file.suffix.lower() in valid_extensions]

    # Reading the file data 
    docs = []
    for file in valid_files : 
        doc_data = PyPDFLoader(path / file).load()
        docs += doc_data
    
    # Splitting the text data into chunks 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # Initializing the database and retriever
    store = FAISS.from_documents(splits, OpenAIEmbeddings(), distance_strategy=DistanceStrategy.COSINE)
    retriever = store.as_retriever(search_kwargs={"k": 3})

    return retriever


class MultilingualChatWorkflow(object):

    def __init__(self, user_list: List[Tuple[str,str]], knowledge_base_directory: Optional[str] = None, llm: Optional[str] = None, valid_extensions: Optional[List[str]] = ['.pdf'], retriever : Optional[Any] = None ):


        ## Initializing the LLM
        if llm is None:
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
        else:
            self.llm = llm


        self.user_list = user_list
        self.valid_extensions = valid_extensions

        self.knowledge_base_directory = self.validate_knowledge_base(knowledge_base_directory, valid_extensions)

        if retriever is None:
            self.retriever = load_retriever(self.knowledge_base_directory, valid_extensions)
        else:
            self.retriever = retriever

        



    def validate_knowledge_base(self, directory_path : str, valid_extensions: List[str]) -> Path | None:
        """Checks the user provided knowledge base directory to ensure that its valid. Also checks for the number of valid files found within the directory

        Args:
            directory_path (str): Directory containing the knowledge base files

        Returns:
            Path: The user provided directory_path as a Path object
        """

        if directory_path is None:
            return None

        path = Path(directory_path)
        
        if not path.exists() or not path.is_dir():
            raise NotADirectoryError(f"The path '{directory_path}' is not a valid directory.")
        
        # Count the valid files
        valid_files = [file for file in path.iterdir() if file.is_file() and file.suffix.lower() in valid_extensions]
        file_count = len(valid_files)

        # Check if there are no valid files and raise a warning if so
        if file_count == 0:
            warnings.warn(f"No .pdf, .docx, or .doc files found in the directory '{directory_path}'.")

        # Print the number of valid files
        print(f"Number of valid files (.pdf, .docx, .doc) in the directory: {file_count}")

        return path 


    def initialize_agent_workflow(self):
        """ Initializes the agent workflow. Specifically, it creates the agent nodes, sets up the workflow (nodes and edges) and compiles it

        Returns:
            _type_: _description_
        """

        
        self.user_agents = {}

        for (userid, lang) in self.user_list:
            self.user_agents[userid] = {'agent': UserAgent(self.llm, lang)}

        self.aya_agent = AyaAgent(self.llm, self.retriever)
        
        self.supervisor_agent = SupervisorAgent(self.llm)


        for userid in self.user_agents:
            self.user_agents[userid]['node'] = functools.partial(get_user_node, agent=self.user_agents[userid]['agent'], name=userid)
        

        aya_node = functools.partial(get_aya_node, agent = self.aya_agent, supervisor_agent = self.supervisor_agent, name ="Aya")
        supervisor_node = functools.partial(get_supervisor_node, agent = self.supervisor_agent)

        workflow = StateGraph(AgentState)

        
        ## Defining nodes
        for userid in self.user_agents:
            workflow.add_node(userid, self.user_agents[userid]['node'])
        
        workflow.add_node("Supervisor", supervisor_node)
        workflow.add_node("Aya",aya_node) 
        
        ## Defining edges
        workflow.add_edge(START, "Supervisor")
        
        for userid in self.user_agents:
            workflow.add_edge("Supervisor", userid)
            workflow.add_edge(userid, END)
        
        
        def router(state) :
            last_message = state["messages"][-1]
            if last_message.sender == "Aya":
                return "Supervisor"
            else:
                return 'END'
                
        workflow.add_conditional_edges("Aya", router, {'Supervisor':'Supervisor','END':END})


        def router2(state) :
            last_message = state["messages"][-1]
            if last_message.sender == "Aya":
                return "END"
            else:
                return 'Aya'
        
        workflow.add_conditional_edges("Supervisor", router2  ,{'Aya':'Aya','END':END})

        ## Compiling Graph
        self.app = workflow.compile()

        for userid in self.user_agents:
            self.user_agents[userid]['agent'].set_graph(self.app) 


def generate_agents(user_list):
    """Creates all required agents for the multilingual chatroom. This includes : 
    1. Individual agents corresponding to each user
    2. A supervisor agent to coordinate the workflow
    3. An Aya agent to assist with RAG applications

    Args:
        user_list Tuple(string, string):  

    Returns:
        _type_: _description_
    """

    llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)

    retriever = load_retriever()
    
    user_agents = {}

    for (userid, lang) in user_list:
        user_agents[userid] = {'agent': UserAgent(llm, lang)}

    # french_agent = UserAgent(llm, "French")
    # spanish_agent = UserAgent(llm, "Spanish")
    # english_agent = UserAgent(llm, "English")
    
    aya_agent = AyaAgent(llm, retriever)
    
    supervisor_agent = SupervisorAgent(llm)

    # Creates the node functions for each of the user agents
    for userid in user_agents:
        user_agents[userid]['node'] = functools.partial(get_user_node, agent=user_agents[userid]['agent'], name=userid)
    
    # french_node = functools.partial(get_user_node, agent=french_agent, name="French")
    # spanish_node = functools.partial(get_user_node, agent=spanish_agent, name="Spanish")
    # english_node = functools.partial(get_user_node, agent=english_agent, name="English")
    
    aya_node = functools.partial(get_aya_node, agent = aya_agent, supervisor_agent = supervisor_agent, name ="Aya")
    supervisor_node = functools.partial(get_supervisor_node, agent = supervisor_agent)

    workflow = StateGraph(AgentState)
    
    # agentList = {'French':{'node':french_node, 'agent':french_agent},
    #                    'Spanish':{'node':spanish_node, 'agent':spanish_agent},
    #                    'English':{'node':english_node, 'agent':english_agent},
    #                    }
    
    
    #Defining nodes
    for userid in user_agents:
        workflow.add_node(userid, user_agents[userid]['node'])
    
    workflow.add_node("Supervisor", supervisor_node)
    
    #Defining edges
    workflow.add_edge(START, "Supervisor")
    
    for userid in user_agents:
        workflow.add_edge("Supervisor", userid)
        workflow.add_edge(userid, END)
    
    
    workflow.add_node("Aya",aya_node) 
    
    
    
    def router(state) :
        last_message = state["messages"][-1]
        if last_message.sender == "Aya":
            return "Supervisor"
        else:
            return 'END'
            
    workflow.add_conditional_edges("Aya", router, {'Supervisor':'Supervisor','END':END})


    def router2(state) :
        last_message = state["messages"][-1]
        if last_message.sender == "Aya":
            return "END"
        else:
            return 'Aya'
    
    workflow.add_conditional_edges("Supervisor", router2  ,{'Aya':'Aya','END':END})

    app = workflow.compile()

    for userid in user_agents:
        user_agents[userid]['agent'].set_graph(app)

    # french_agent.set_graph(app)
    # spanish_agent.set_graph(app)
    # english_agent.set_graph(app)

    return user_agents, 



def generate_response(sender, text):

    user_agents = generate_agents()

    # agent_map = {'English':english_agent, 'Spanish':spanish_agent,'French':french_agent}


    user_agents[sender]['agent'].send_text(text)
    # agent_map[sender].send_text(text)

    output = [english_agent.chat_history[-1].content,
              spanish_agent.chat_history[-1].content,
              french_agent.chat_history[-1].content]

    return output