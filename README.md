# Multilingual Multi-Agent Chat application for Workplace

A LLM based chat application to address language barriers and workplace inefficiences in large enterprises. Developed as part of the 1st edition of the [Aya Expedition](https://www.youtube.com/watch?v=oKstMwSUElA) organized by [Cohere for AI](https://cohere.com/research)



[![Screenshot-2024-09-01-at-12-25-24-AM.png](https://i.postimg.cc/gJ5j0sYL/Screenshot-2024-09-01-at-12-25-24-AM.png)](https://postimg.cc/dk2wH8ZJ)



## Run Instructions

1. Update your `PYTHONPATH` variable to include the repo's directory. This can be done through the terminal using:
```
export PYTHONPATH="/Users/roshansk/Documents/GitHub/Multilingual-Chatbot/:$PYTHONPATH”
```
2. Initiate the FASTAPI server that runs the agent workflow. 
```
python ml_chat/server/start_server.py -u user_list.csv --data_directory /Users/roshansk/Documents/GitHub/Multilingual-Chatbot/ml_chat/playground/data
```
3. Run the Chat UI by opening the HTML pages under `App UI` on your web browser

---

**Defining User list** : To initialize the backend agent workflow, it requires a list of users along with their preferred languages. This is to be passed as a csv to the `-u` argument of start_server.py 

**Changing LLM backend** : You can select different LLM backends by selecting one of 'openai','claude' or 'aya' and passing it to `--llm` argument of start_server.py. The code assumes that you have the API keys for these services already set in the environment. If not, you can modify the code to manually feed in the API keys

**Defining knowledge base** : Its recommended to assign a directory to the `--data_directory` argument that serves as the knowledge base for the RAG system. The code currently accepts documents that are in .pdf or .txt formats. 
 
 
## Demo Video
[Video](https://youtu.be/_i8WKbTXojM)


## Features 

With the goal to address language barriers and other common workplace inefficiences, the app offers the following features : 
* #### Multilingual translation

Allow users to read and write in their preferred languages, with all messages being translated by language specific agents
* #### Aya, workplace AI assistant

Aya serves as a in-built AI assistant available in all chats that helps with common workplace tasks. Some of these are listed below

* #### RAG-based Q&A

Aya can respond to user queries using the relevant context derived from docs in the internal knowledge base
* #### Documentation-on-the-Go!

In cases where you have shared a detailed answer to a question or explained the specifics of a process, Aya allows you to save these chat messages to your knowledge base. This allows Aya to be able to answer similar questions in the future

* #### Smart Summarize
Allows users to catch up quickly on a chat thread by summarizing messages and also identifying action items assigned to users


## LLM Backend
The app provides 3 different options for the LLM backend to power the agentic workflow. These include : 
1. Claude
2. ChatGPT
3. [Aya](https://huggingface.co/CohereForAI/aya-23-8B)

In terms of performance, especially the translation of messages, we have observed the general trend of **Claude > ChatGPT > Aya**
