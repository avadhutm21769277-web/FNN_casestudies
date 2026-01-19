

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib


def load_data(csv_filename):
    df = pd.read_csv(csv_filename)
    return df


def name_of_headers(df):
    headers = df.columns[:]
    return headers


def feature_header_of_dataset(df):
    feature_headers = df.columns[:3]
    return feature_headers


def target_header_of_dataset(df):
    target_header = df.columns[3:]
    return target_header


def datatypes_of_headers(df):
    return df.dtypes


def shuffle_the_data(df):
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def statistical_info(df):
    return df.describe()


def checking_null_values(df):
    return df.isnull().sum()


def split_data_vertically(df, target_header):
    X = df.drop(columns=target_header, axis=1)
    Y = df[target_header]
    return X, Y


def split_data_into_four_parts(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )
    return X_train, X_test, Y_train, Y_test


def using_standerd_scaler(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


def build_fnn_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_fnn_model(model, X_train, Y_train, X_test, Y_test, model_path='headbrain_fnn.keras'):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
    ]
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=200,
        batch_size=8,
        callbacks=callbacks,
        verbose=1
    )
    return model, history


def evaluate_model(model, X_test, Y_test):
    preds = model.predict(X_test)
    r2 = r2_score(Y_test, preds)
    mse = mean_squared_error(Y_test, preds)
    return r2, mse


def save_model_and_scaler(model, scaler, model_path='headbrain_fnn.keras', scaler_path='scaler.pkl'):
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    return model_path, scaler_path


def load_model_and_scaler(model_path='headbrain_fnn.keras', scaler_path='scaler.pkl'):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_new_data(model, scaler):
    new_data = pd.DataFrame({
        'Gender': [1],
        'Age Range': [1],
        'Head Size(cm^3)': [4008]
    })
    scaled = scaler.transform(new_data)
    prediction = model.predict(scaled)
    return prediction


def main():
    print("HEAD_BRAIN MODEL USING FNN")

    csv_filename = 'MarvellousHeadBrain.csv'
    df = load_data(csv_filename)

    headers = name_of_headers(df)
    print("Headers:", headers)

    target_header = target_header_of_dataset(df)
    df = shuffle_the_data(df)

    print("Null values:", checking_null_values(df))
    print("Statistical info:", statistical_info(df))

    X, Y = split_data_vertically(df, target_header)
    X_train, X_test, Y_train, Y_test = split_data_into_four_parts(X, Y)

    scaler, X_train_scaled, X_test_scaled = using_standerd_scaler(X_train, X_test)

    model = build_fnn_model(X_train_scaled.shape[1])
    model, history = train_fnn_model(model, X_train_scaled, Y_train, X_test_scaled, Y_test)

    r2, mse = evaluate_model(model, X_test_scaled, Y_test)
    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

    model_path, scaler_path = save_model_and_scaler(model, scaler)
    print("Model saved successfully")

    loaded_model, loaded_scaler = load_model_and_scaler(model_path, scaler_path)
    print("Model loaded successfully")

    prediction = predict_new_data(loaded_model, loaded_scaler)
    print("Predicted Brain Size:", prediction[0][0])


if __name__ == "__main__":
    main()
