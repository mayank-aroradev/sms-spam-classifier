import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import sklearn

nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('stopwords')

ps = PorterStemmer()
tfidf_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
  text=text.lower()
  y=[]
  text=nltk.word_tokenize(text)
  for i in text :
      if i.isalnum():
        
        y.append(i)

  text=y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
    
 
  text =y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)



st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter the message")
if st.button("Predict"):
    # preprocess
    transformed_sms = transform_text(input_sms)
    # vectorize
    vectorized_sms = tfidf_vectorizer.transform([transformed_sms])
    # PREDICT
    prediction = model.predict(vectorized_sms)[0]
    if prediction == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")