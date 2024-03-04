"""
Main code to run the chatbot

This chatbot runs in three stages.

1st stage: Ask questions prepared in advance to understand the users' preference
2nd stage: Summarize the answers then deliver recommendations
3rd stage: Chat with ChatGPT 3.5 Turbo
"""


import openai
import streamlit as st
from st_questions import questions
from st_sim import sim

import time

st.title("Imperial IPA Chat")

openai.api_key = st.secrets['OAK']

user_ans = []

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:

    st.session_state.messages = [       
        {
        "role": "assistant", 
        #"content": "Hi! I am your beer assistant.  \nI am here to help you find a Imperial IPA perfect for you.  \nI will ask you 4 questions to find out your preference.  \n  \nHere is the first question:  \n  \nHop Intensity: Do you prefer an Imperial IPA with a dominant, overpowering hop presence or a more balanced hop-malt profile?"
        "content": "Hey there, beer buddy! 🍻  \n  \nI'm your hoppy helper, ready to embark on a quest to find your dream Imperial IPA. Brace yourself for 4 quirky questions that'll unlock the secret to your hop-tastic heart!  \n  \nReady, set, hop it up! 🚀  \n  \nHop Wars: Do you want your Imperial IPA to have hops that burst through the door like the cool kids at a party or prefer a chill hop-malt hangout? 🌟"
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    if message["role"] == "user":
        user_ans.append(message["content"])

prompt = st.chat_input()
if prompt:
    
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Load Prepared Questions
    if len(user_ans) < 3:
        with st.chat_message("assistant"):
            
            message_placeholder = st.empty()
            full_response = ""
            question = questions(len(user_ans))
            
            for word in question.split():
                full_response += word + " "
                message_placeholder.markdown(full_response + "|")
                time.sleep(0.1)
            
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": question})

    # Generate Recommendations
    elif len(user_ans) == 3:

        rec_msg = ""
        with st.chat_message("assistant"):
            response_msg = "We are finding a perfect Imperial IPA for you, based on my answer!  \nPlease wait a moment 🍻"
            st.markdown(response_msg)
            beer = sim(st.session_state.messages)
            st.markdown(beer)
        st.session_state.messages.append({"role": "assistant", "content": beer})

    # Chat the rest with ChatGPT
    else:
        with st.chat_message("assistant"):

            message_placeholder = st.empty()
            
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=st.session_state.messages,
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "|")

            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})



































