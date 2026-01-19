################################################################################
#   BANKNIFTY ANALYSER USING FEEDFORWARD NEURAL NETWORK (FNN)
################################################################################

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

line = "___" * 30


################################################################################
#   1. load_data()
################################################################################
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df


################################################################################
#   2. name_of_headers()
################################################################################
def name_of_headers(df):
    return df.columns[:]


################################################################################
#   3. feature_header_of_dataset()
################################################################################
def feature_header_of_dataset(df):
    feature_headers = df.drop(df.columns[4], axis=1).columns.tolist()
    return feature_headers


################################################################################
#   4. target_header_of_dataset()
################################################################################
def target_header_of_dataset(df):
    target_header = df.columns[4:5]
    return target_header


################################################################################
#   5. remove_unnecessory_columns()
################################################################################
def remove_unnecessory_columns(df):
    if 'Date ' in df.columns:
        df = df.drop(columns='Date ')
    return df


################################################################################
#   6. split_data()
################################################################################
def split_data(df):
    X = df.drop(columns='Close ')
    Y = df['Close ']
    print("Data splitted into feature and target successfully!!")
    print(line)
    return X, Y


################################################################################
#   7. checking_null()
################################################################################
def checking_null(df):
    print(df.isnull().sum())
    print("Null value check completed.")
    print(line)


################################################################################
#   8. train_data()
################################################################################
def train_data(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    return X_train, X_test, Y_train, Y_test


################################################################################
#   9. scaled_data()
################################################################################
def scaled_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


################################################################################
#   10. build_fnn_model()
################################################################################
def build_fnn_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')
    return model


################################################################################
#   11. train_fnn_model()
################################################################################
def train_fnn_model(model, X_train_scaled, Y_train):
    history = model.fit(X_train_scaled, Y_train, epochs=100, batch_size=16, verbose=0)
    print("Model training completed successfully!")
    return model, history


################################################################################
#   12. evaluate_model()
################################################################################
def evaluate_model(model, X_test_scaled, Y_test):
    predictions = model.predict(X_test_scaled)
    predictions = predictions.flatten()      # convert (n,1) → (n,)
    Y_test = np.array(Y_test).flatten()      # ensure Y_test is also 1D

    r2 = r2_score(Y_test, predictions)
    mse = np.mean(np.square(Y_test - predictions))

    print(f"Model R2 Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    return r2, mse


################################################################################
#   13. save_fnn_model()
################################################################################
def save_fnn_model(model, filename="banknifty_fnn.keras"):
    model.save(filename)
    print(f"Model saved successfully as {filename}")
    return filename


################################################################################
#   14. load_fnn_model()
################################################################################
def load_fnn_model(filename):
    model = load_model(filename, compile=False)
    print("Model loaded successfully!")
    return model


################################################################################
#   15. predict_new_data()
################################################################################
def predict_new_data(scaler, loaded_model):
    new_data = pd.DataFrame({
        'Open ': [57872.85],
        'High ': [58224.00],
        'Low ': [57872.85],
        'Shares Traded ': [43334000],
        'Turnover (₹ Cr)': [2387.96],
    })

    new_scaled_data = scaler.transform(new_data)
    prediction = loaded_model.predict(new_scaled_data)
    return prediction


################################################################################
#   MAIN FUNCTION
################################################################################
def main():
    dataset = 'NIFTY BANK-16-10-2024-to-16-10-2025.csv'
    print(line)
    print("BANKNIFTY ANALYSER USING FEEDFORWARD NEURAL NETWORK")
    print(line)

    # Load and preprocess data
    df = load_data(dataset)
    df = remove_unnecessory_columns(df)
    checking_null(df)
    X, Y = split_data(df)
    X_train, X_test, Y_train, Y_test = train_data(X, Y)
    scaler, X_train_scaled, X_test_scaled = scaled_data(X_train, X_test)

    # Build and train model
    model = build_fnn_model(X_train_scaled.shape[1])
    model, _ = train_fnn_model(model, X_train_scaled, Y_train)

    # Evaluate model
    evaluate_model(model, X_test_scaled, Y_test)

    # Save and load model
    filename = save_fnn_model(model)
    loaded_model = load_fnn_model(filename)

    # Predict new data
    prediction = predict_new_data(scaler, loaded_model)
    print(f"Predicted Close Price: {prediction[0][0]:.2f}")


################################################################################
if __name__ == "__main__":
    main()

################################################################################
#   AUTHORISED BY : AVADHUT YASHWANT MOTE
################################################################################
