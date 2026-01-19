# Diabetes Prediction using Feedforward Neural Network (FNN)

## Project Overview
This project predicts the likelihood of diabetes in a person using a Feedforward Neural Network (FNN) built with TensorFlow and Keras.  
It uses the Pima Indians Diabetes dataset (`diabetes.csv`) and includes a graphical user interface (GUI) built with Tkinter for real-time predictions.

## Features
- Data preprocessing with zero-value replacement and scaling  
- Feedforward Neural Network (FNN) model using TensorFlow  
- Early stopping and model checkpoint for best model saving  
- Model and scaler saving/loading for reuse  
- GUI interface for user-based input and prediction  

## Technologies Used
- Python 3.13  
- TensorFlow 2.20.0  
- NumPy  
- Pandas  
- scikit-learn  
- Joblib  
- Tkinter (for GUI)

## How It Works
1. Loads and cleans the dataset (`diabetes.csv`)
2. Replaces zero values with column mean
3. Scales the data using StandardScaler
4. Builds and trains a Feedforward Neural Network
5. Saves the trained model (`diabetes_fnn.keras`) and scaler (`scaler.pkl`)
6. Loads the model for prediction
7. Opens a GUI window to enter new patient data for prediction

## Model Architecture
Input Layer: 8 neurons (for 8 features)  
Hidden Layer 1: 64 neurons, ReLU activation  
Hidden Layer 2: 32 neurons, ReLU activation  
Output Layer: 1 neuron, Sigmoid activation  

Loss Function: Binary Crossentropy  
Optimizer: Adam  
Metrics: Accuracy  

## How to Run the Project

### Step 1: Install Dependencies
Open Command Prompt or Terminal and run:
```
pip install tensorflow pandas numpy scikit-learn joblib
```

### Step 2: Place Dataset
Make sure `diabetes.csv` is in the same folder as the Python script.

### Step 3: Run the Script
Run the following command:
```
python diabetes_fnn.py
```

### Step 4: Use GUI
After model training completes, a GUI window will appear.  
Enter patient details and click **Predict** to see diabetes prediction result.

## Sample Prediction Output (Console)
```
Test loss: 0.4321, Test accuracy: 78.45%
Sample prediction probability: 0.62
```

If probability ≥ 0.5 → Diabetic possibility  
Else → Non-Diabetic possibility  

## Output Files
- `diabetes_fnn.keras` — Trained FNN model  
- `scaler.pkl` — Saved scaler for preprocessing  
- `diabetes.csv` — Input dataset  

## Author
Developed by **Avadhut Yashwant Mote**
