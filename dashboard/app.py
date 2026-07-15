import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')


# Load saved files
model = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\svm_model.pkl")
vectorizer = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\tfidf_vectorizer.pkl")
label_encoder = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\label_encode.pkl")

stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

def cleaned_text(text):

    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    text = re.sub(r'\.(?=[A-Z])', '. ', text)                        
    text = re.sub(r'-\s', ' ', text)                                 
    text = text.lower()                                                                 
    tokens = word_tokenize(text)                                                        
    tokens = [w for w in tokens if w not in punctuations and w not in stop_words]       
    tagged = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged]           
    return ' '.join(tokens)  


st.set_page_config( page_title="Disease Prediction using NLP", layout="wide")
st.title("Disease Prediction using NLP")

user_input = st.text_area("Enter Symptoms")

if st.button("Predict Disease"):

    if user_input.strip() == "":
        st.warning("Please enter symptoms.")
    else:
        clean_input = cleaned_text(user_input)
        
        vector = vectorizer.transform([clean_input])

        prediction = model.predict(vector)

        disease = label_encoder.inverse_transform(prediction)

        st.success(f"Predicted Disease: {disease[0]}")
