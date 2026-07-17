import streamlit as st
import joblib
import re
import nltk
# from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import string
from pathlib import Path

@st.cache_resource
def download_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)

download_nltk()


st.set_page_config(
                    page_title="Clinical Trial Disease Category Classification",
                    page_icon="🩺",
                    layout="wide",
                    initial_sidebar_state="expanded"
)

# -------------------------------
# Model Folder Path
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"


# -------------------------------
# Load Saved Files
# -------------------------------


@st.cache_resource
def load_prediction_models():
    model = joblib.load(MODEL_DIR / "svm_model.pkl")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    label_encoder = joblib.load(MODEL_DIR / "label_encode.pkl")

    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_prediction_models()


@st.cache_resource
def load_model(file_name):
    return joblib.load(MODEL_DIR / file_name)


stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

def cleaned_text(text):
    def get_wordnet_pos(word_tag):
        if word_tag.startswith('J'):
            return wordnet.ADJ
        elif word_tag.startswith('V'):
            return wordnet.VERB
        elif word_tag.startswith('N'):
            return wordnet.NOUN
        elif word_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    text = re.sub(r'\.(?=[A-Z])', '. ', text)                        
    text = re.sub(r'-\s', ' ', text)                                 
    text = text.lower()         

    tokens = re.findall(r"[a-zA-Z]+", text)                                                        
    tokens = [
                token for token in tokens
                if token not in punctuations and token not in stop_words
    ]   

    tagged = pos_tag(tokens)
    tokens = [
                    lemmatizer.lemmatize(word, get_wordnet_pos(tag))
                    for word, tag in tagged
    ]   
           
    return ' '.join(tokens)  




st.sidebar.title("Disease Prediction using NLP")
page = st.sidebar.radio( "Content", ["Home","EDA","Disease Prediction"] )


# Home Page:
if page == "Home":
        st.title("🩺 Clinical Trial Disease Category Classification")
        st.markdown("""
                        ### Project Overview

                        This application predicts disease categories from clinical trial summaries using Natural Language Processing.

                        ### Model

                        - TF-IDF Vectorizer
                        - Linear Support Vector Machine

                        ### Features

                        - Disease Prediction
                        - Interactive EDA
                        - Confusion Matrix
                        - Age Distribution
                        - Phase Distribution
                        """)


# EDA:
elif page == "EDA":
        st.header("Exploratory Data Analysis")

        selection = st.pills("Data Visualization", 
        options=["Disease category", "Sex eligibility disease category", "Age Distribution", "Phase Distribution",
                 "Status Distribution", "Study type dist", "Correlation Matrix by Disease category", 
                 "Word count disease category","Confusion Matrix" ]
                    )
        
        if selection == "Disease category":
                #disease_category_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\Disease_category_count.pkl")
                disease_category_fig = load_model("Disease_category_count.pkl")
                st.plotly_chart(disease_category_fig, use_container_width=True)

        elif selection == "Sex eligibility disease category":
                #sex_disease_dist_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\Sex_Eligibility_Disease_Cat.pkl")
                sex_disease_dist_fig = load_model("Sex_Eligibility_Disease_Cat.pkl")
                st.plotly_chart(sex_disease_dist_fig, use_container_width=True)

        elif selection == "Age Distribution":
              col1, col2 = st.columns([1,1])
              with col1:
                    #min_age_dist_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\min_age_dist_box_plot.pkl")
                    min_age_dist_fig = load_model("min_age_dist_box_plot.pkl")
                    st.plotly_chart(min_age_dist_fig, use_container_width=True)

              with col2:
                    #max_age_dist_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\max_age_dist_box_plot.pkl")
                    max_age_dist_fig = load_model("max_age_dist_box_plot.pkl")
                    st.plotly_chart(max_age_dist_fig, use_container_width=True)

              col3 = st.columns(1)[0]
              with col3:
                    #min_max_age_dist_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\min_max_age_dist.pkl")
                    min_max_age_dist_fig = load_model("min_max_age_dist.pkl")
                    st.plotly_chart(min_max_age_dist_fig, use_container_width=True)
        

        elif selection == "Correlation Matrix by Disease category":
                #correlation_mat_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\Correlation_matrix_disease_cat.pkl")
                correlation_mat_fig = load_model("Correlation_matrix_disease_cat.pkl")
                st.plotly_chart(correlation_mat_fig, use_container_width=True)
                

        elif selection == "Word count disease category":
                #tot_word_count_disease_cat_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\total_Word_Count_Disease_Category.pkl")
                tot_word_count_disease_cat_fig = load_model("total_Word_Count_Disease_Category.pkl")
                st.plotly_chart(tot_word_count_disease_cat_fig, use_container_width=True)

                #avg_word_count_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\avg_Word_Count_Disease_Category.pkl")
                avg_word_count_fig = load_model("avg_Word_Count_Disease_Category.pkl")
                st.plotly_chart(avg_word_count_fig, use_container_width=True)

        elif selection == "Phase Distribution":
                #trial_phase_dist_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\Trial_Phase_Distribution.pkl")
                trial_phase_dist_fig = load_model("Trial_Phase_Distribution.pkl")
                st.plotly_chart(trial_phase_dist_fig, use_container_width=True)

        elif selection == "Status Distribution":
                #status_dist_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\Trial_status_dist.pkl")
                status_dist_fig = load_model("trial_status_dist.pkl")
                st.plotly_chart(status_dist_fig, use_container_width=True)

        elif selection == "Study type dist":
                #study_type_dist_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\study_type_dist.pkl")
                study_type_dist_fig = load_model("study_type_dist.pkl")
                st.plotly_chart(study_type_dist_fig, use_container_width=True)

        elif selection == "Confusion Matrix":
                #cm_test_svm_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\confu_matrix_test_svm.pkl")
                cm_test_svm_fig = load_model("confu_matrix_test_svm.pkl")
                st.plotly_chart(cm_test_svm_fig, use_container_width=True)

                #cm_train_svm_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\confu_matrix_train_svm.pkl")
                cm_train_svm_fig = load_model("confu_matrix_train_svm.pkl")
                st.plotly_chart(cm_train_svm_fig, use_container_width=True)



# Model prediction:
elif page == "Disease Prediction":
        st.header("Disease Prediction using NLP")

        user_input = st.text_area("Enter Symptoms")

        if st.button("Predict Disease"):

            if user_input.strip() == "":
                st.warning("Please enter symptoms.")

            else:
                clean_input = cleaned_text(user_input)

                vector = vectorizer.transform([clean_input])

                prediction = model.predict(vector)[0]

                disease = label_encoder.inverse_transform([prediction])[0]

                st.success(f"Predicted Disease: {disease}")

