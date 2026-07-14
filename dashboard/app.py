import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load saved files
model = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\svm_model.pkl")
vectorizer = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\tfidf_vectorizer.pkl")
label_encoder = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\label_encode.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def cleaned_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]
    return " ".join(words)


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
