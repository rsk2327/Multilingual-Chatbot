import argparse
import socket
import uvicorn
from fastapi import FastAPI
import threading
from pydantic import BaseModel

def main():

	app = FastAPI()
	host="0.0.0.0"
	port=8000

	class InputData(BaseModel):
		message: str

	@app.post("/process")
	async def process_input(input_data: InputData):

		print(f"Hey there : {input_data.message}")

		return f"Hey there : {input_data.message}"

		# try:
		#     # Process the input message
		#     system_message = f"Hello, {input_data.user}! Your message has been received."
			
		#     # Detect the language of the input message
		#     detected_lang, _ = langid.classify(input_data.message)
			
		#     return {
		#         "system_message": system_message,
		#         "language": detected_lang
		#     }
		# except Exception as e:
		#     raise HTTPException(status_code=500, detail=str(e))



	uvicorn.run(app, host=host, port=port)




if __name__ == "__main__":
	main()