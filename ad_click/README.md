# AD CLICK PREDICTOR USING FEEDFORWARD NEURAL NETWORK

This project predicts whether a user will click on an online advertisement based on various input features.  
It uses a Feedforward Neural Network (FNN) implemented using TensorFlow/Keras.  
The dataset is preprocessed, encoded, scaled, trained, and evaluated automatically.

## Project Description

The project reads a CSV dataset named ad\_click\_dataset.csv, cleans and encodes the data, performs scaling,  
and then trains a neural network model. After training, the model is saved and can be used later for predictions.

## Features

1. Load and explore dataset
2. Handle missing values and encode categorical columns
3. Scale numerical data using StandardScaler
4. Train Feedforward Neural Network for binary classification
5. Save and load trained model
6. Predict ad click for new user data

## Libraries Required

pandas  
numpy  
matplotlib  
seaborn  
scikit-learn  
tensorflow  
joblib

## File Details

ad\_click\_fnn.py        -> main python file containing all functions and main()  
ad\_click\_dataset.csv   -> dataset file used for training  
boxplot\_age.png        -> saved boxplot image of the dataset  
adclick\_fnn.h5         -> saved trained model file

## How To Run

1. Install all dependencies

   pip install -r requirements.txt

2. Place your dataset file (ad\_click\_dataset.csv) in the same folder as ad\_click\_fnn.py
3. Run the script

   python ad\_click\_fnn.py

4. The model will start training, display accuracy, and create a saved model file.

   ## Output

   During training, accuracy of each epoch is printed on screen.  
   After completion, the final accuracy and a prediction for new user input are displayed.

   Example Output:

   Model Accuracy: 92.85%  
   Predicted Ad Click for new user: 1

   ## Notes

* The model uses a simple neural network with two hidden layers.
* You can adjust epochs, learning rate, and layer sizes in fnn\_classifier() function.
* The project automatically replaces missing values and encodes categorical features.

  ## Author

  Created by :	Avadhut Yashwant Mote

