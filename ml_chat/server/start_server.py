import argparse
import socket
import uvicorn
from fastapi import FastAPI
import pandas as pd
import threading
from pydantic import BaseModel

from ml_chat.agents.workflow import MultilingualChatWorkflow

class InputMessage(BaseModel):
    sender: str
    message: str

def find_available_port(host='localhost', start_port=8000):
    """Helper function to retrieve available ports to run API service

    Args:
        host (str, optional): _description_. Defaults to 'localhost'.
        start_port (int, optional): _description_. Defaults to 8000.

    Returns:
        _type_: _description_
    """
    for port in range(start_port, 65536):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available ports")

def user_input_arg(value: str):
    """
    Custom type converter function to parse a list of tuples from either a comma-separated string or a CSV file.
    """

    if value.endswith('.csv'):

        try: 
            user_list = [tuple(x) for x in pd.read_csv(value).to_numpy()]
            return user_list
        except FileNotFoundError:
            raise argparse.ArgumentTypeError("Value must be a comma-separated list of tuples or a path to a CSV file")
    else:
        try:
            # Try to parse the value as a comma-separated string of tuples
            user_list = [tuple(x.split(',')) for x in value.split()]
            return user_list
        except:
            raise ValueError("User list argument is incorrect")
      
      
            

def create_app(user_list, knowledge_base_directory):

    app = FastAPI()

    workflow = MultilingualChatWorkflow(user_list=user_list, knowledge_base_directory= knowledge_base_directory)
    workflow.initialize_agent_workflow()

    @app.post("/process")
    async def generate_response(input_message : InputMessage):

        sender = input_message.sender
        message = input_message.message
        workflow.send_message(sender, message)

    
    @app.get("/")
    async def get_messages():

        output = workflow.get_latest_message()
        return output 

    return app

def start_api_service(host, port):
    app = create_app()
    uvicorn.run(app, host=host, port=port)



def main():
    parser = argparse.ArgumentParser(description="Start an API service")
    parser.add_argument("--host", default="localhost", help="Host address")
    parser.add_argument("--port", type=int, help="Port number (optional)")
    parser.add_argument('-u', '--users', type=user_input_arg, nargs='+', required=True, help='List of tuples or path to CSV file')
    parser.add_argument('--data_directory', type=str, required=False, help="Directory containing the files for the knowledge base (RAG)")
    
    # Add more arguments as needed
    
    args = parser.parse_args()

    host = args.host

    if args.port:
        port = args.port
    else:
        port = find_available_port(args.host)

    print(f"Starting API service on {args.host}:{port}")

    user_list = args.users[0]
    print(user_list)
    data_directory = args.data_directory

    # print(f"Number of users in chatroom : {len(user_list)}")


    app = create_app(user_list=user_list, knowledge_base_directory=data_directory)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()