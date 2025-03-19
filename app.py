import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

## load the LSTM model

model = load_model("next_word_lstm.h5")

## load the tokenizer 

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequences_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequences_len:
        token_list = token_list[-(max_sequences_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequences_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]  # Fix: Extract scalar value

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:  # Fix: Use scalar value
            return word
    return None

## streamlit app

st.title("Next word prediction with LSTM and Early Stopping")
input_text = st.text_input(" Enter the sequnece of words", " to be or not to")
if st.button(" predict the Next word"):
    max_sequences_len = model.input_shape[1] + 1 ## Retrieve the max sequence length from the input model
    next_word = predict_next_word(model, tokenizer, input_text, max_sequences_len)
    st.write(f"Next word :{next_word}")


