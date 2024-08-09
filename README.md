
## Key Files
**agents/agents.py**
Describes the design of the agent systems using Langgraph

**agents/app.py**
Streamlit app to demo the agent system 

**agents/workflow.py**
Code that initializes the agent graph that facilitates the interaction between agents.py and the streamlit app

**MVP.ipynb**
Notebook to run the MVP

**app.py**
streamlit file to run the interactive chat room



## Run Instructions 

**MVP.ipynb**
Direct initialization of 3 agents (English, French and Spanish) is done using the _generate_agents_ functions. If you want to create custom agents for different languages, refer to the code within _generate_agents_.

Once initialized, each agent can be used to send texts using the _send_text_ method. 

The RAG agent, Aya, can be addressed by adding the term Aya in your text. The bot currently answers questions related to ML which are covered in the _machine_learning_basics.pdf_.

To view the texts from the perspective of an agent, use the _display_chat_history_ method


**app.py**

Run the interactive chat room using the command

`streamlit run app.py`
