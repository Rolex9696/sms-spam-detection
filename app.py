import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
nltk.download('punkt')

def transform_Message(Message):
    messageessage = Message.lower()
    Message = nltk.word_tokenize(Message)

    y = []
    for i in Message:
        if i.isalnum():
            y.append(i)

    Message = y[:]
    y.clear()

    for i in Message:
        if i not in ENGLISH_STOP_WORDS and i not in string.punctuation:
            y.append(i)

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Email/SMS Spam Detector")

input_sms = st.text_area("Enter the message")
if st.button('predict'):

    # 1. preprocess
    transform_sms = transform_Message(input_sms)
    # 2. vectirize
    vector_input = tfidf.transform([transform_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")