import openai
import streamlit as st

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

openai.api_key = st.secrets['OAK']
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

com_embedding = np.load('com_embeddings.npy')

def summarize(conversations):
    
    full_response = ""

    for response in openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversations,
        stream=True,
    ):
        full_response += response.choices[0].delta.get("content", "")

    return full_response    

def sim(conversations):

    summary = summarize(conversations)
    emb_summary = model.encode(summary)

    similarity = np.matmul(com_embedding, emb_summary)

    df_sim = pd.DataFrame(similarity)
    top_3 = df_sim[0].sort_values(ascending=False).head(3).index

    df_beer = pd.read_csv("0303_imperial_ipa_rate.csv")
    li_rec = df_beer.loc[top_3, ["beer_name", "beer_url"]].values

    desired_format = ""
    for i, entry in enumerate(li_rec):
        beer_name, beer_url = entry
        desired_format += f"{i}. {beer_name} : {beer_url}\n"

    return desired_format
