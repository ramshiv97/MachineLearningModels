# Spam Email Classifier using Naive Bayes

## Overview
This project builds a spam email classifier using the Naive Bayes algorithm. It utilizes natural language processing (NLP) techniques to preprocess text data and the Term Frequency-Inverse Document Frequency (TF-IDF) approach to extract features. The model is trained to differentiate between spam and non-spam (ham) messages.

# Project overview

## Setup Instructions

### 1. Clone the Repository
git clone <repository-url> cd spam-classifier


### 2. Install Dependencies
pip install -r requirements.txt


### 3. Run the Model Training
python src/preprocess.py


### 4. Test the Model
To test the saved model on new data, modify the `preprocess.py` script to load the model and predict on custom messages.

## Features
- **Data Preprocessing**: Cleaned and tokenized the text data.
- **Feature Extraction**: Used TF-IDF vectorization to convert text into numerical features.
- **Model Training**: Trained a Naive Bayes classifier to classify spam vs. ham messages.
- **Model Evaluation**: Achieved an accuracy score of XX% on the test dataset.

## Dataset
The project uses the [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset) containing 5,572 messages labeled as spam or ham.

## Dependencies
- Python 3.x
- Pandas
- Scikit-learn
- NLTK
- Joblib

## Author
Shivram Edathatta
