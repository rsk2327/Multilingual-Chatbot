import streamlit as st
from functools import partial

from agents.workflow import *

st.set_page_config(layout="wide")
st.title('Multilingual Chat Room')

col1,col2,col3 = st.columns(3)
col1 = col1.container(height = 400, border  = True)
col2 = col2.container(height = 400, border  = True)
col3 = col3.container(height = 400, border  = True)

input1,input2,input3 = st.columns(3)
input1 = input1.container()
input2 = input2.container()
input3 = input3.container()

col1.write("English")
col2.write("Spanish")
col3.write("French")


message_history_keys = ['messages_english','messages_spanish','messages_french']
container_mapping = {'messages_english':col1,'messages_spanish':col2,'messages_french':col3}

# Initialize chat history
for message_type in message_history_keys:
    if message_type not in st.session_state:
        st.session_state[message_type] = []


# Display chat messages from history on app rerun
for message_type in message_history_keys:
    container = container_mapping[message_type]
    
    for message in st.session_state[message_type]:
        
        with container.chat_message(message["role"]):
            st.write(message["content"])

def on_submit(input_key, username):

    text = st.session_state[input_key]
    sender_mapping = {'col1_input':'English',
                      'col2_input':'Spanish',
                      'col3_input':'French'
                     }
    
    sender = sender_mapping[input_key]

    agent_output = generate_response(sender, text)
    
    
    for output_text, message_type in zip(agent_output,message_history_keys):
        st.session_state[message_type].append({"role": username, "content":output_text})
    

on_submit1 = partial(on_submit, input_key = 'col1_input', username = 'english')
on_submit2 = partial(on_submit, input_key = 'col2_input', username = 'spanish')
on_submit3 = partial(on_submit, input_key = 'col3_input', username = 'french')

col1_input = input1.chat_input(key = "col1_input", on_submit = on_submit1)
col2_input = input2.chat_input(key = "col2_input", on_submit = on_submit2)
col3_input = input3.chat_input(key = "col3_input", on_submit = on_submit3)

