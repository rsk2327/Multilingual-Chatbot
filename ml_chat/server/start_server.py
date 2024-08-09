import argparse
import socket
import uvicorn
from fastapi import FastAPI
import pandas as pd
import threading

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

def user_input_arg(value):
    """
    Custom type converter function to parse a list of tuples from either a comma-separated string or a CSV file.
    """
    try:
        # Try to parse the value as a comma-separated string of tuples
        user_list = [tuple(x.split(',')) for x in value.split()]
        return user_list
    except ValueError:
        # If the value is not a comma-separated string, try to open it as a CSV file
        try:
            user_list = []
            user_list = [tuple(x) for x in pd.read_csv(value).to_numpy()]
            return user_list
        except FileNotFoundError:
            raise argparse.ArgumentTypeError("Value must be a comma-separated list of tuples or a path to a CSV file")

def create_app():
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "API service is running"}

    # Add more routes and API functionality here

    return app

def start_api_service(host, port):
    app = create_app()
    uvicorn.run(app, host=host, port=port)



def main():
    parser = argparse.ArgumentParser(description="Start an API service")
    parser.add_argument("--host", default="localhost", help="Host address")
    parser.add_argument("--port", type=int, help="Port number (optional)")
    parser.add_argument('-u', '--users', type=user_input_arg, nargs='+', required=True, help='List of tuples or path to CSV file')
    
    # Add more arguments as needed
    
    args = parser.parse_args()

    if args.port:
        port = args.port
    else:
        port = find_available_port(args.host)

    print(f"Starting API service on {args.host}:{port}")

    user_list = args.users 

    print(f"Number of users in chatroom : {len(user_list)}")

    # Start the API service in a separate thread
    # api_thread = threading.Thread(target=start_api_service, args=(args.host, port))
    # api_thread.start()

    # # Keep the main thread alive
    # try:
    #     while True:
    #         pass
    # except KeyboardInterrupt:
    #     print("Shutting down API service")

if __name__ == "__main__":
    main()