BANKNIFTY ANALYSER USING FEEDFORWARD NEURAL NETWORK



Overview

This project uses a Feedforward Neural Network (FNN) built with TensorFlow and Keras to analyze and predict the closing price of the Bank NIFTY Index based on its historical data. 

The model processes daily stock data such as Open, High, Low, Shares Traded, and Turnover to predict the closing value.



The workflow includes data preprocessing, model training, evaluation, and prediction of new values.



Features

\- Loads and preprocesses BankNIFTY dataset

\- Handles missing values and removes unnecessary columns

\- Splits data into training and testing sets

\- Scales data for efficient training

\- Builds and trains a Feedforward Neural Network

\- Evaluates model performance using R² score and MSE

\- Saves and reloads the trained model

\- Predicts the future closing price based on new input data



Technologies Used

\- Python 3.x

\- TensorFlow / Keras

\- NumPy

\- Pandas

\- Scikit-learn



Project Structure

banknifty\_fnn.py

NIFTY BANK-16-10-2024-to-16-10-2025.csv

README.txt



How It Works

1\. Load dataset using pandas

2\. Remove date column and handle missing values

3\. Split dataset into features (X) and target (Y)

4\. Normalize feature data using StandardScaler

5\. Build a Feedforward Neural Network model:

&nbsp;  Input Layer

&nbsp;  Hidden Layer 1: 64 neurons (ReLU)

&nbsp;  Hidden Layer 2: 32 neurons (ReLU)

&nbsp;  Output Layer: 1 neuron (Linear)

6\. Compile the model using Adam optimizer and MSE loss

7\. Train model for 100 epochs with batch size 16

8\. Evaluate model with test data and print R² and MSE

9\. Save trained model in .keras format

10\. Load model and predict new sample data



Input Example

The model expects the following columns (after removing the Date column):



Open | High | Low | Shares Traded | Turnover (₹ Cr) | Close



Example new input for prediction:

Open: 57872.85

High: 58224.00

Low: 57872.85

Shares Traded: 43334000

Turnover (₹ Cr): 2387.96



Output Example

Model R2 Score: 0.9123

Mean Squared Error: 352.71

Predicted Close Price: 58011.34



Usage Instructions

1\. Place the dataset file (NIFTY BANK-16-10-2024-to-16-10-2025.csv) in the same directory.

2\. Ensure Python 3 and required libraries are installed:

&nbsp;  pip install tensorflow pandas numpy scikit-learn

3\. Run the script:

&nbsp;  python banknifty\_fnn.py

4\. The script will train, evaluate, and display the predicted closing price.



Author

Avadhut Yashwant Mote



