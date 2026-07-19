# Clinical_Trial_Disease_Category_Classification

# Project Overview:
- The project focus is to preprocess and transform the clinical trial text data into a structured format and apply Natural Language Processing (NLP) and Machine Learning techniques to classify clinical trial summaries into appropriate disease categories. Additionaly to build an Intelligent disease classification system using the “Brief Summary” column of clinical trial data to automatically predict disease categories based on medical text content.
- The workflow includes data collection, data cleaning, data preprocessing, exploratory data analysis (EDA), Feature Engineering, Text Preprocessing using Tokenization & Lemmmatization, Model Testing and Evaluation is done with Classification Model. The entire solution is integrated into a Streamlit dashboard, enabling users to explore insights and predict type of disease category using Breif Summary.

# Technology Used:
- python
- Pandas & NumPy
- Scikit-Learn
- Matplotlib & Seaborn
- Plotly
- Streamlit

# Data Collection:
- Data collection is to gather and organize the raw data in a structured format, making it ready for preprocessing, analysis, feature engineering, and machine learning model development.
### Dataset:
- The dataset size is large. So, downloadable link is given.
[Download the dataset from Kaggle](https://www.kaggle.com/datasets/yourusername/clinical-trial-disease-category-classification)

- The raw dataset (clinical_trials_raw_patient2trial_conditions.csv) contains 60,337 records and 16 columns, sourced from clinical trial registrations. The two columns central to this project are:
- brief_summary — free-text description of each trial (model input, X)
- source_condition_query — the disease category label (target, y), with 8 unique values

| S.No | Disease Category |
|---|---|
| 1 | Breast Cancer - Abnormal cells grow uncontrollably in the breast |
| 2 | Type 2 Diabetes - High blood sugar |
| 3 | COVID-19 - contagious viral infection caused by the SARS-CoV-2 virus. |
| 4 | Anxiety - Mental health disorder characterized by excessive fear or worry|
| 5 | Chronic Obstructive Pulmonary Disease - Long-term lung disease that restricts airflow and makes breathing difficult|
| 6 | Rheumatoid Arthritis - Autoimmune disease where the immune system attacks the joints|
| 7 | Glaucoma - Eye disease that damages the optic nerve and can lead to vision loss. |
| 8 | Sickle Cell Anemia - Inherited blood disorder that produces abnormal sickle-shaped red blood cells |

# Data Cleaning
- The process of cleaning and preparing raw data so that it can be effectively used for analysis and machine learning.
- Handle missing values (null data) to ensures the dataset is complete and prevents errors during analysis and model training.

| S.No | Feature | Missing Values Count |
|---|---|---|
| 1 | official_title | 786 |
| 2 | conditions | 1 |
| 3 | interventions | 6061 |
| 4 | phase | 37001 |
| 5 | sex | 33 |
| 6 | minimum_age | 3216 |
| 7 | maximum_age | 31937 |
| 8 | healthy_volunteers | 1480 |
| 9 | eligibility_criteria | 11 |

- Null value for 'Official title' can be cleared by using 'title' feature like moreover same data is present in both feature so, just copy the details from 'title' feature and fill it in missing records.
  - For 'conditions' fill 'Unknown' for missing value.
  - For 'interventions' fill 'No_Interventions_Mentioned' as null value.
  - For 'phase' fill 'Unknown_Phase' as null value.
  - For 'sex', null value can be filled by using mode function between All, Female & Male.
  - For 'minimum_age' fill '0' as null value and 'maximum_age' fill '120' as null value.
  - For 'healthy_volunteers' null value can be filled by grouping 'study_type' and 'phase' then use mode function to fill the value.
  - For 'eligibility_criteria' fill 'No_Criteria_Mentioned' as null value.
- Remove duplicate records to prevents the same data from being counted multiple times, which could change the analysis and predictions.
- Perform data type conversion to enable accurate calculations, filtering, and machine learning operations.

# NLP Text Preprocessing:
- NLP Text preprocessing is done for feature 'brief_summary' because it is an input for the model evaluation.
- The concept used for text preprocessing is Tokenization, POS, Lemmatization.
    - First step is to Formating correction by inserting missing spaces after immediately followed by a capital letter
    - Eg: 'recovery.This' → 'recovery. This'
    - Second step is normalises the words by using Lowercasing operation.
    - Eg: 'Breast Cancer Patients Often'  -> 'breast cancer patients often'
    - Third step is Tokenization process where text splitted into individual word tokens using NLTK's word_tokenize.
    - Eg:  'breast cancer patients often have perioperative'   -->  'breast', 'cancer', 'patients', 'often', 'have', 'perioperative'
    - Fourth Step is to remove stopword and Punctuation using NLTK's English stopword list and Python's string.punctuation set.
    - Eg:  'anxiety', 'and', 'depression', ',', 'which', 'can', 'lead', 'to', 'poor'  -->  'anxiety', 'depression', 'lead', 'poor'
    - Fifth step is to tagging with help of POS (Part-of-speech) where each token is tagged as  a noun, verb, adjective, or adverb using NLTK's averaged perceptron tagger
    - Eg: ('study', 'NN'), ('aims', 'VBZ'), ('determine', 'VBP')
    - Last step is to Lemmatize each word according to POS, each token is reduced to its dictionary base form using its POS tag, so verb forms are correctly normalized by converting from plural to singular form.
    - Eg: 'aims', 'improved', 'diagnosed'  -->  'aim', 'improve', 'diagnose'.

# EDA (Exploratory Data Analysis):
- Exploratory Data Analysis (EDA) is used to summarize, visualize, and understand the dataset.
### No of Trial per Disease Category:
<img width="1000" height="500" alt="no_of_trial_per_disease_category" src="https://github.com/user-attachments/assets/ae6b1cf5-f8a6-4ec1-8632-e8d74aa7ffff" />
- This chart shows about no of trials of each disease category and from this we can understand first three disease breast cancer, type 2 diabetes, covid-19 are above 10k count. Where as anxiety has around 9260 count. other four disease chronic obstructive pulmonary disease, rheumatoid arthritis, glaucoma, sickle cell anemia are below 7000 counts only.

### Study type by Disease category:
<img width="1000" height="600" alt="study_type_disease_category" src="https://github.com/user-attachments/assets/0cbcc92c-ef72-4983-a936-bd932e27d66f" />

### Age distribution:
<img width="1000" height="500" alt="Avg_max_min_age_by_disease_cat" src="https://github.com/user-attachments/assets/e342050e-6b31-4739-978e-13c69b67c1eb" />

### Sex - Disease category:
<img width="1000" height="600" alt="sex_eligibility_disease_category" src="https://github.com/user-attachments/assets/8220ac66-3d27-4f48-9176-347c06bbb4d6" />

### Status Distribution:
<img width="1000" height="500" alt="overall_trial_status_dist" src="https://github.com/user-attachments/assets/4a8b7e45-d323-4c9f-8d0d-0749a92a1a53" />

### Healthy Volunteer Distribution:
<img width="800" height="500" alt="healthy_Vollunteer_dist" src="https://github.com/user-attachments/assets/f9feba50-fcd0-4254-bf28-2e6407ef0c54" />


# Feature Engineering:
- Feature Engineering is the process of creating and transforming variables to improve the performance of machine learning models.
- cleaned_summary — free-text description of each trial (model input, X)
- source_condition_query — the disease category label (target, y), with 8 unique values
- Applying Label encoding for 'source_condition_query' feature. 'Anxiety' = '0', 'Breast Cancer' = '1', 'Chronic Obstructive Pulmonary Disease' = '2', 'COVID-19' = '3', 'Glaucoma' = '4', 'Rheumatoid Arthritis' = '5', 'Sickle Cell Anemia' = '6' & 'Type 2 Diabetes' = '7'.
- Creating a new feature as 'cleaned_summary' where preprocessed text are present.

### Correlation Matrix:
<img width="1000" height="800" alt="vocalbulary_correlation_matrix" src="https://github.com/user-attachments/assets/b7a2e730-68cd-4ad1-a018-241edbe80dbf" />

### Word count by disease category:
<img width="1000" height="500" alt="tot_word_count_disease_category" src="https://github.com/user-attachments/assets/010853cf-7e6a-4a33-b786-937c012337f5" />

<img width="800" height="500" alt="avg_word_count_disease_category" src="https://github.com/user-attachments/assets/e1c9a4bf-87ed-48d8-813c-0c90deb597f7" />

## TF-IDF:(Term Frequency - Inverse Document Frequency)
- TF-IDF is used to convert each word into numbers where machine can learn easily.
- tfidf_vectorizer = TfidfVectorizer( max_features=5000,  ngram_range=(1, 2),  min_df=5,  max_df=0.9 )

#### Fit and transform:
X_tfidf = tfidf_vectorizer.fit_transform(clinical_text_encoded['cleaned_summary'])

- Sample words = aim: 0.0556, aim determine: 0.1252, also: 0.0721, anxiety: 0.1437, anxiety depression: 0.2256, breast: 0.1171, breast cancer: 0.1225, bring: 0.1535, cancer: 0.1135, cancer patient: 0.1888, concern: 0.1183

# Model Training:
- Models are trained,
    - Naive Bayes Model, Logistic regression,  Linear SVM (Linear Support Vector Machine) & Random Forest
- The results are,

## Naive Bayes Evaluation Metrics

| Metric | Training Set | Testing Set |
|--------|-------------:|------------:|
| Accuracy | 91.29% | 90.74% |
| Precision | 92.66% | 92.43% |
| Recall | 88.90% | 87.64% |
| F1-Score | 90.51% | 89.65% |

## Logistic Regression Evaluation Metrics

| Metric | Training Set | Testing Set |
|--------|-------------:|------------:|
| Accuracy | 96.43% | 94.34% |
| Precision | 97.02% | 95.01% |
| Recall | 94.80% | 92.09% |
| F1-Score | 95.85% | 93.44% |

## Linear SVM Evaluation Metrics

| Metric | Training Set | Testing Set |
|--------|-------------:|------------:|
| Accuracy | 98.10% | 94.76% |
| Precision | 98.37% | 95.23% |
| Recall | 97.66% | 92.99% |
| F1-Score | 98.01% | 94.03% |

## Random Forest Evaluation Metrics

| Metric | Training Set | Testing Set |
|--------|-------------:|------------:|
| Accuracy | 85.12% | 84.92% |
| Precision | 94.34% | 94.24% |
| Recall | 76.89% | 76.04% |
| F1-Score | 82.88% | 82.19% |

- From this Linear SVM is best when compare with other models results.

## Confusion matrix:
### Based on testing data:
<img width="1000" height="800" alt="confusion_matrix_testing_data" src="https://github.com/user-attachments/assets/6214f67c-164c-4812-bf69-c1521c53fcbf" />

### Based on training data:
<img width="1000" height="800" alt="confusion_matrix_training_data" src="https://github.com/user-attachments/assets/7ed65e55-6d1e-4216-8360-e04edd4bb053" />


## Model Deployment:
https://clinicaltrialdiseasecategoryclassification-cclawzbathmejrp8sxe.streamlit.app/













