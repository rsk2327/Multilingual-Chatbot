import argparse
import socket
import uvicorn
from fastapi import FastAPI
import threading

def find_available_port(host='localhost', start_port=8000):
    for port in range(start_port, 65536):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available ports")



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
    
    # Add more arguments as needed
    
    args = parser.parse_args()

    if args.port:
        port = args.port
    else:
        port = find_available_port(args.host)

    print(f"Starting API service on {args.host}:{port}")
    
    # Start the API service in a separate thread
    api_thread = threading.Thread(target=start_api_service, args=(args.host, port))
    api_thread.start()

    # Keep the main thread alive
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Shutting down API service")

if __name__ == "__main__":
    main()