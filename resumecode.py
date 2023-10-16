# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:27:42 2023

@author: Bhavana Sri Sowjanya
"""
# Import necessary libraries
import numpy as np
import pandas as pd
import warnings

# Ignore warnings to prevent clutter in the output
warnings.filterwarnings('ignore')

# Read data from a CSV file into a Pandas DataFrame
data = pd.read_csv("D:\Resume_Analysis\Resume Screening.csv")

# Display the first few rows of the DataFrame
data.head()

# Get the shape of the DataFrame (number of rows and columns)
data.shape

# Display information about the DataFrame, including data types and non-null counts
data.info()

# Check for missing values in the DataFrame and display the count of missing values for each column
data.isnull().sum()

# Get unique values in the 'Category' column of the DataFrame
data['Category'].unique()

# Get the number of unique values in the 'Category' column
data['Category'].nunique()

# Count the occurrences of each unique value in the 'Category' column and store it in a new DataFrame
categories = data['Category'].value_counts().reset_index()
categories

# Create a copy of the original DataFrame for preprocessing
data1 = data.copy()

# Add a new column 'cleaned_resume' to store cleaned resume text
data1['cleaned_resume'] = ""
data1

# Import the 're' module for regular expressions
import re

# Define a function to clean resume text using regular expressions
def clean_function(resumeText):
    # Remove URLs
    resumeText = re.sub('http\S+\s*', ' ', resumeText)
    # Remove RT and cc
    resumeText = re.sub('RT|cc', ' ', resumeText)
    # Remove hashtags
    resumeText = re.sub('#\S+', '', resumeText)
    # Remove mentions
    resumeText = re.sub('@\S+', '  ', resumeText)
    # Remove punctuations
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)
    # Remove non-ASCII characters
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    # Remove extra whitespace
    resumeText = re.sub('\s+', ' ', resumeText)
    return resumeText

# Apply the 'clean_function' to the 'Resume' column and store the cleaned text in 'cleaned_resume'
data1['cleaned_resume'] = data1['Resume'].apply(lambda x: clean_function(x))

# Display the first few rows of the DataFrame with cleaned resume text
data1.head()

# Import LabelEncoder for encoding categorical labels
from sklearn.preprocessing import LabelEncoder

# Create a copy of the DataFrame for encoding labels
data2 = data1.copy()

# Encode the 'Category' column using LabelEncoder and replace it in the DataFrame
data2['Category'] = LabelEncoder().fit_transform(data2['Category'])

# Display the first few rows of the DataFrame with encoded labels
data2.head()

# Import necessary libraries for building models and processing text data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Extract the cleaned resume text and target labels
Text = data2['cleaned_resume'].values
Target = data2['Category'].values

# Create a TF-IDF vectorizer for text data with some additional settings
word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')

# Fit the vectorizer on the resume text data and transform it into numerical features
word_vectorizer.fit(Text)
WordFeatures = word_vectorizer.transform(Text)

# Display the shape of the transformed features
WordFeatures.shape

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(WordFeatures, Target, random_state=42)

# Display the shape of the training and testing sets
print(X_train.shape)
print(X_test.shape)

# Import machine learning models for classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Create a dictionary of machine learning models
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier()
}

# Import OneVsRestClassifier for multi-class classification
from sklearn.multiclass import OneVsRestClassifier

# Create an empty list to store models using the OneVsRestClassifier wrapper
model_list = []

# Loop through the models and wrap them in OneVsRestClassifier, then add to the list
for model in models.values():
    model_list.append(OneVsRestClassifier(model))

# Display the list of wrapped models
model_list

# Train each model in the list on the training data
for i in model_list:
    i.fit(X_train, y_train)
    print(f'{i} trained')

# Print a message indicating that all models have been trained
print("All models trained")

# Evaluate the accuracy of each model on the training and testing sets
for count, value in enumerate(model_list):
    print(f"Accuracy of {value} on the training set:", model_list[count].score(X_train, y_train))
    print(f"Accuracy of {value} on the test set:", model_list[count].score(X_test, y_test))
    print("*" * 100)
