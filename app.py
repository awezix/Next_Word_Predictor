import streamlit as st
import pickle
import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

model=load_model('next_word_lstm.h5')

with open('tokenizer.pkl','rb') as handle:
    tokenizer=pickle.load(handle)

def predict_next_word(model,tokenizer,text,max_sequence):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence:
        token_list=token_list[-(max_sequence-1):]
    token_list=pad_sequences([token_list],maxlen=max_sequence-1,padding='pre')
    predicted_word=model.predict(token_list,verbose=0)
    predicted_word_index=np.argmax(predicted_word,axis=1)
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None

# streamlit app
st.title("Next word prediction with LSTM RNN")
input_text=st.text_input('Enter the sequence of words')
max_sequence_len=model.input_shape[1]+1
if st.button('predict next word'):
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f'Next Word : {next_word}')