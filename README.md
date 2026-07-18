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

| S.No | Disease Category | Trial Count |
|---|---|---|
| 1 | Breast Cancer | 16,265 |
| 2 | Type 2 Diabetes | 11,398 |
| 3 | COVID-19 | 10,103 |
| 4 | Anxiety | 9,260 |
| 5 | Chronic Obstructive Pulmonary Disease | 6,142 |
| 6 | Rheumatoid Arthritis | 3,618 |
| 7 | Glaucoma | 2,150 |
| 8 | Sickle Cell Anemia | 1,136 |

# Data Cleaning






## Model Deployment:
https://clinicaltrialdiseasecategoryclassification-cclawzbathmejrp8sxe.streamlit.app/













