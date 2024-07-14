from agents.workflow import *

english_agent, spanish_agent, french_agent = generate_agents() 

english_agent.send_text("Hello World!")

english_agent.display_chat_history()

spanish_agent.display_chat_history()
