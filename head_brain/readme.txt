Head Brain Size Prediction using Feedforward Neural Network

This project is based on predicting brain size using a Feedforward Neural Network model. The model is trained using the MarvellousHeadBrain dataset which contains information about gender, age range and head size. The goal is to predict the brain size from these input features.

Project Files
MarvellousHeadBrain.csv - dataset file
headbrain_fnn.py - main python source code
headbrain_fnn.keras - saved trained model (generated automatically)
scaler.pkl - saved scaler used for normalization (generated automatically)
README.md - documentation file

Project Workflow

Load the dataset using pandas

Display column headers, datatypes, and statistical summary

Shuffle the dataset to remove any order bias

Check for null values in the dataset

Split the data into features and target columns

Split the dataset into training and testing sets

Apply standard scaling using StandardScaler

Build a Feedforward Neural Network model using Keras

Train the model with training data and validate it using testing data

Evaluate the model using R2 score and mean squared error

Save the trained model and scaler for future predictions

Load the saved model and make predictions on new data

Libraries Used
pandas
numpy
matplotlib
scikit learn
tensorflow
joblib

How to Run the Project

Make sure that MarvellousHeadBrain.csv and headbrain_fnn.py are in the same folder

Open command prompt or terminal in that folder

Run the following command
python headbrain_fnn.py

The program will train the neural network, evaluate its performance and show the predicted brain size

Output Information
After running the script, you will see

R2 score and mean squared error printed on the console

The predicted brain size for a sample input

The trained model saved as headbrain_fnn.keras

The scaler saved as scaler.pkl