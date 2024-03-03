This is a simple chatbot built to demonstrate part of my coding abilities. 

Data (Imperial IPA) was scraped from a beer forum

Sentence2Vec (Huggingface Transformer: 'sentence-transformers/all-MiniLM-L12-v2') is applied to comments

Part of Speech Tagging: 
Adjectives in each comment are extracted
Results are fed to ChatGPT 3.5 to generate 5 questions to understand user's beer preference

Based on the answers of the questions
Comment Similarity (Matrix Multiplication) is calculated to return top 3 most similar Imperial in database. 

