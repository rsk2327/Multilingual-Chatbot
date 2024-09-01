import os
import getpass
from pathlib import Path
import warnings

from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
import functools
from functools import partial
from typing import TypedDict, Annotated, Sequence, Tuple, Optional, List, Any, Dict
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_cohere import ChatCohere
from langchain_anthropic import ChatAnthropic


from ml_chat.agents.utils import *
from ml_chat.agents.agents import AyaAgent, UserAgent, get_aya_node, get_supervisor_node, get_user_node, AgentState

def router(state, sender, dest1, dest2):
    last_message = state["messages"][-1]
    if last_message.sender == sender:
        return dest1
    else:
        return dest2


def load_retriever(path : Path, valid_extensions : List[str]):

    # Identifying the valid files
    valid_files = [file for file in path.iterdir() if file.is_file() and file.suffix.lower() in valid_extensions]

    # Reading the file data 
    docs = []
    for file in valid_files : 
        if '.pdf' in file.suffix:
            doc_data = PyPDFLoader(path / file).load()
        elif '.txt' in file.suffix:
            doc_data = TextLoader(path / file).load()
        else:
            continue
        docs += doc_data
    
    # Splitting the text data into chunks 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # Initializing the database and retriever
    store = FAISS.from_documents(splits, OpenAIEmbeddings(), distance_strategy=DistanceStrategy.COSINE)
    retriever = store.as_retriever(search_kwargs={"k": 3})

    return store, retriever


class MultilingualChatWorkflow(object):

    def __init__(self, user_list: List[Tuple[str,str]], knowledge_base_directory: Optional[str] = None, llm: Optional[str] = 'OpenAI', valid_extensions: Optional[List[str]] = ['.pdf','.txt']):


        ## Initializing the LLM
        if llm.lower() == 'openai':
            self.llm = ChatOpenAI(model="gpt-4o", streaming=True, temperature = 0.0)
        elif llm.lower() == 'aya':
            self.llm = ChatCohere(model="c4ai-aya-23-35b", streaming=True, temperature = 0.0)
        elif llm.lower() == 'claude':
            self.llm = ChatAnthropic(model='claude-3-opus-20240229', streaming = True, temperature = 0.0)
        else:
            raise ValueError("Invalid option provided by LLM. Only accepts the following options : OpenAI, Aya")


        self.user_list = user_list
        self.valid_extensions = valid_extensions

        self.knowledge_base_directory = self.validate_knowledge_base(knowledge_base_directory, valid_extensions)

        
        self.store, self.retriever = load_retriever(self.knowledge_base_directory, valid_extensions)
        

        ## Initializing the agent workflow
        self.initialize_agent_workflow()
        



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

        # Checking for user provided knowledge base file
        user_file = path / 'user_data.txt'
        if not user_file.exists():
            open(user_file, 'w').close()

        return path 


    def initialize_agent_workflow(self):
        """ Initializes the agent workflow. Specifically, it creates the agent nodes, sets up the workflow (nodes and edges) and compiles it

        Returns:
            _type_: _description_
        """

        
        self.user_agents = {}

        for (userid, lang) in self.user_list:
            self.user_agents[userid] = {'agent': UserAgent(self.llm, userid, lang)}
        
        self.aya_agent = AyaAgent(self.llm, self.store, self.retriever)

        user_knowledge_file = str(self.knowledge_base_directory.joinpath('user_data.txt'))

        for userid in self.user_agents:
            self.user_agents[userid]['node'] = functools.partial(get_user_node, agent=self.user_agents[userid]['agent'], name=userid)
        
        aya_node = functools.partial(get_aya_node, aya = self.aya_agent, user_knowledge_file = user_knowledge_file, name ="Aya")        
        supervisor_node = functools.partial(get_supervisor_node)



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
        

        workflow.add_conditional_edges("Aya",
                                        partial(router, sender = 'Aya', dest1 = 'Supervisor', dest2 = 'END'),
                                        {'Supervisor':'Supervisor','END':END})
    
        workflow.add_conditional_edges("Supervisor", 
                                       partial(router, sender = 'Aya', dest1 = 'END', dest2 = 'Aya'),
                                       {'Aya':'Aya','END':END})

        ## Compiling Graph
        self.app = workflow.compile()

        for userid in self.user_agents:
            self.user_agents[userid]['agent'].set_graph(self.app) 


    def generate_response(self, sender:str , message:str) -> Dict[str,str]:
        """ Starts the agent workflow when a message is sent by a user. Returns the message to be displayed to each of the user's

        Args:
            sender (str): ID of the user sending the message
            message (str): Content of the message

        Returns:
            Dict[str,str]: Dict containing the messages to be shown to each user
        """

        # Start the workflow by sending the message from the starting user 
        self.user_agents[sender]['agent'].send_text(message)

        output = {}
        for userid in self.user_agents:
            output[userid] = self.user_agents[userid]['agent'].chat_history[-1].content

        return output
    
    def send_message(self, sender:str, message:str)-> None:

        self.user_agents[sender]['agent'].send_text(message)


    def get_latest_message(self) -> List[Dict]:

        output = []

        userid = self.user_list[0][0]
        last_sender = self.user_agents[userid]['agent'].chat_history[-1].sender


        json_output = {}
        
        for userid in self.user_agents:
            json_output[userid] = {'content':self.user_agents[userid]['agent'].chat_history[-1].content,'sender': self.user_agents[userid]['agent'].chat_history[-1].sender}

        output.append(json_output)


        if last_sender == 'Aya':
            json_output = {}
        
            for userid in self.user_agents:
                json_output[userid] = {'content':self.user_agents[userid]['agent'].chat_history[-2].content,'sender': self.user_agents[userid]['agent'].chat_history[-2].sender}

            output.append(json_output)
            
        

        return output



