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
    tokens = word_tokenize(text)                                                        
    tokens = [w for w in tokens if w not in punctuations and w not in stop_words]       
    tagged = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged]           
    return ' '.join(tokens)  


st.sidebar.title("Disease Prediction using NLP")
page = st.sidebar.radio( "Content", ["Home","EDA","Disease Prediction"] )


# Home Page:
if page == "Home":
        st.header("Disease Prediction using NLP")
        st.set_page_config(page_title="Home Page", layout="wide" )


# EDA:
elif page == "EDA":
        st.header("Exploratory Data Analysis")
        st.set_page_config(page_title="EDA", layout="wide" )

        selection = st.pills("Data Visualization", 
        options=["Disease category", "Sex eligibility disease category", "Age Distribution", "Phase Distribution",
                 "Status Distribution", "Study type dist", "Correlation Matrix by Disease category", 
                 "Word count disease category","Confusion Matrix" ]
                    )
        
        if selection == "Disease category":
                disease_category_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\Disease_category_count.pkl")
                st.plotly_chart(disease_category_fig)

        elif selection == "Sex eligibility disease category":
                sex_disease_dist_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\Sex_Eligibility_Disease_Cat.pkl")
                st.plotly_chart(sex_disease_dist_fig)

        elif selection == "Age Distribution":
              col1, col2 = st.columns([1,1])
              with col1:
                    min_age_dist_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\min_age_dist_box_plot.pkl")
                    st.plotly_chart(min_age_dist_fig)

              with col2:
                    max_age_dist_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\max_age_dist_box_plot.pkl")
                    st.plotly_chart(max_age_dist_fig)

              col3 = st.columns(1)[0]
              with col3:
                    min_max_age_dist_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\min_max_age_dist.pkl")
                    st.plotly_chart(min_max_age_dist_fig)
        
        elif selection == "Correlation Matrix by Disease category":
                correlation_mat_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\Correlation_matrix_disease_cat.pkl")
                st.plotly_chart(correlation_mat_fig)
                
        elif selection == "Word count disease category":
                        tot_word_count_disease_cat_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\total_Word_Count_Disease_Category.pkl")
                        st.plotly_chart(tot_word_count_disease_cat_fig)

                        avg_word_count_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\avg_Word_Count_Disease_Category.pkl")
                        st.plotly_chart(avg_word_count_fig)

        elif selection == "Phase Distribution":
                trial_phase_dist_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\Trial_Phase_Distribution.pkl")
                st.plotly_chart(trial_phase_dist_fig)

        elif selection == "Status Distribution":
                status_dist_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\Trial_status_dist.pkl")
                st.plotly_chart(status_dist_fig)

        elif selection == "Study type dist":
                study_type_dist_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\study_type_dist.pkl")
                st.plotly_chart(study_type_dist_fig)

        elif selection == "Confusion Matrix":
                    cm_test_svm_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\confu_matrix_test_svm.pkl")
                    st.plotly_chart(cm_test_svm_fig)

                    cm_train_svm_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_5-Clinical_Trial_Disease_Category_Classification\models\confu_matrix_train_svm.pkl")
                    st.plotly_chart(cm_train_svm_fig)



# Model prediction:
elif page == "Disease Prediction":
        st.set_page_config( page_title="Disease Prediction", layout="wide")
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
