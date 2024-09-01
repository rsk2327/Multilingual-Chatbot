import argparse
import socket
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import pandas as pd
import threading
from pydantic import BaseModel
import json 
from typing import List 
from ml_chat.agents.workflow import MultilingualChatWorkflow

class InputMessage(BaseModel):
    sender: str
    message: str

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)



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
      
      
            

def create_app(user_list, knowledge_base_directory, llm):

    app = FastAPI()

    workflow = MultilingualChatWorkflow(user_list=user_list, knowledge_base_directory= knowledge_base_directory, llm = llm)
    
    manager = ConnectionManager()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_json()
                sender = data.get('sender', 'Unknown')
                message = data.get('message', '')
                
                workflow.send_message(sender,message)

                output = workflow.get_latest_message()

                # Broadcast the message to all connected clients
                if len(output)==2:                    
                    await manager.broadcast(output[1])
                    await manager.broadcast(output[0])
                else:
                    await manager.broadcast(output[0])
                
                
               
        except WebSocketDisconnect:
            manager.disconnect(websocket)
            await manager.broadcast({"sender": "System", "message": f"Client #{id(websocket)} left the chat"})
        except json.JSONDecodeError:
            await websocket.send_json({"error": "Invalid JSON"})
        except KeyError as e:
            await websocket.send_json({"error": f"Missing required key: {str(e)}"})


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





def main():
    parser = argparse.ArgumentParser(description="Start an API service")
    parser.add_argument("--host", default="localhost", help="Host address")
    parser.add_argument("--port", type=int, help="Port number (optional)")
    parser.add_argument('-u', '--users', type=user_input_arg, nargs='+', required=True, help='List of tuples or path to CSV file')
    parser.add_argument('--data_directory', type=str, required=False, help="Directory containing the files for the knowledge base (RAG)")
    parser.add_argument('--llm', default = "openai", type=str, required=False, help="LLM model to power the agent workflow")
    
    # Add more arguments as needed
    
    args = parser.parse_args()

    host = args.host
    llm = args.llm

    if args.port:
        port = args.port
    else:
        port = find_available_port(args.host)


    print(f"Starting API service on {args.host}:{port}")

    user_list = args.users[0]
    print(user_list)
    data_directory = args.data_directory

    app = create_app(user_list=user_list, knowledge_base_directory=data_directory, llm = llm)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()