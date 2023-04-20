# Kafka-ML_Classification-App

## Project Description

This project is aimed at collecting and classifying data from a mobile phone's accelerometer and gyroscope sensors. The project is divided into three main parts:
- Data collection
- Data classification
- Frontend implementation

The data collection involves creating a mobile app that collects data from the phone's sensors and stores it in a database. The data classification involves processing the data using machine learning classification models. The frontend implementation involves displaying live data from the phone and predicting the position or state of the phone based on the labeled data.

## Approach
### Data Collection
We have used the Flutter framework to create a simple app that collects the required data and uploads it to a laptop via Flask API. We have also used the sensors_plus package to access the sensors of the mobile phone. The collected data is then labeled and stored in a MongoDB.

### Data Classification
The labeled data is then ingested into the Kafka environment, and machine learning classification models are applied to classify the labels. We have used three machine learning classification models KNN, Naive Bayes, and SVM. The models are compared, and the best model is chosen for predicting the state or position of the phone.

### Frontend Implementation
The final part of the project involves implementing a frontend website using Flask or any other framework. The website takes live data from the phone connected via API and predicts the position or state of the phone based on the labeled data. The website also displays the readings of the accelerometer and gyroscope sensors.

## Technologies Used
- Flutter: A native app development platform used to develop the mobile app.
- Sensors_Plus: A package used to access the sensors of the mobile phone.
- Flask: A micro web framework used to create the API endpoint to post or get the data.
- MongoDB: Databases used to store the labeled data.
- Kafka: A distributed event streaming platform used to ingest the labeled data and process it.
- Machine Learning Classification Models (KNN, Naive Bayes, SVM): Models used to classify the labeled data.
- HTML, CSS, and JavaScript: Web development languages used to create the frontend website.

## Setup Instructions
- Clone the repository to your local machine.
- Install the required dependencies for Python and Flutter.

## Authors:
- Muhammad Ahmed Suhail
- Hamza Khan
- Abdullah Basharat

## Acknowledgments:
- This was completed as a project for Big Data Analytics at FAST - NUCES Islamabad.















