import openai
import streamlit as st
from st_questions import questions
from st_sim import sim

st.title("Imperial IPA Chat")

openai.api_key = st.secrets['OAK']

user_ans = []

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:

    st.session_state.messages = [       
        {
        "role": "assistant", 
        "content": "Hi! I am your beer assistant. \nI am here to help you find a Imperial IPA perfect for you. \nI will ask you 5 questions to find out your preference. \nHere is the first question: Hop Intensity: Do you prefer an Imperial IPA with a dominant, overpowering hop presence or a more balanced hop-malt profile?"
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

    if len(user_ans) < 3:
        with st.chat_message("assistant"):
            question = questions(len(user_ans))
            st.markdown(question)
        st.session_state.messages.append({"role": "assistant", "content": question})

    elif len(user_ans) == 3:

        rec_msg = ""
        with st.chat_message("assistant"):
            response_msg = "We are finding a perfect Imperial IPA for you, based on my answer!"
            st.markdown(response_msg)
            beer = sim(st.session_state.messages)
            st.markdown(beer)
        st.session_state.messages.append({"role": "assistant", "content": beer})


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



































