import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import tkinter as tk
from tkinter import messagebox

def load_csv(path):
    df = pd.read_csv(path)
    return df

def replace_zero_values_by_mean(df):
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    for c in cols:
        if c in df.columns:
            # treat zeros as missing for certain columns (except Pregnancies is sometimes valid, but kept per original logic)
            df[c] = df[c].replace(0, np.nan)
            mean_val = df[c].mean()
            df[c].fillna(mean_val, inplace=True)
    return df

def splitting_vertically(df):
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return X, y

def split_dataset(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled

def build_fnn_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_fnn_model(model, X_train, y_train, X_val=None, y_val=None,
                    epochs=100, batch_size=16, model_path="diabetes_fnn.keras"):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1)
    ]
    if X_val is None or y_val is None:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_split=0.15, callbacks=callbacks, verbose=1)
    else:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
    return model, history

def evaluate_model(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    return loss, acc

def save_model_and_scaler(model, scaler, model_path="diabetes_fnn.keras", scaler_path="scaler.pkl"):
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    return model_path, scaler_path

def load_model_and_scaler(model_path="diabetes_fnn.keras", scaler_path="scaler.pkl"):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_new_data(model, scaler, sample_dict):
    df = pd.DataFrame([sample_dict])
    Xs = scaler.transform(df)
    pred_prob = model.predict(Xs)[0][0]
    return pred_prob

def run_gui(model, scaler):
    def predict_diabetes():
        try:
            data = {
                "Pregnancies": float(entry_pregnancies.get()),
                "Glucose": float(entry_glucose.get()),
                "BloodPressure": float(entry_bp.get()),
                "SkinThickness": float(entry_skin.get()),
                "Insulin": float(entry_insulin.get()),
                "BMI": float(entry_bmi.get()),
                "DiabetesPedigreeFunction": float(entry_dpf.get()),
                "Age": float(entry_age.get())
            }
            prob = predict_new_data(model, scaler, data)
            if prob >= 0.5:
                messagebox.showwarning("Result", f"Diabetic possibility (prob={prob:.2f})")
            else:
                messagebox.showinfo("Result", f"Non-Diabetic possibility (prob={prob:.2f})")
        except Exception as e:
            messagebox.showerror("Error", "Please enter valid numeric values")

    root = tk.Tk()
    root.title("Diabetes Prediction (FNN)")
    root.geometry("420x560")

    tk.Label(root, text="Diabetes Prediction (FNN)", font=("Arial", 16, "bold")).pack(pady=10)

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    labels = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
              "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    entries = []
    for i, lbl in enumerate(labels):
        tk.Label(frame, text=lbl, anchor="w", width=25).grid(row=i, column=0, pady=5)
        e = tk.Entry(frame, width=18)
        e.grid(row=i, column=1, pady=5)
        entries.append(e)

    entry_pregnancies, entry_glucose, entry_bp, entry_skin, entry_insulin, entry_bmi, entry_dpf, entry_age = entries

    tk.Button(root, text="Predict", command=predict_diabetes, width=20).pack(pady=15)

    root.mainloop()

def main():
    csv_path = "diabetes.csv"
    df = load_csv(csv_path)

    df = replace_zero_values_by_mean(df)

    X, y = splitting_vertically(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.3, random_state=42)

    scaler, X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    model = build_fnn_model(X_train_scaled.shape[1])

    model, history = train_fnn_model(model, X_train_scaled, y_train,
                                     X_val=X_test_scaled, y_val=y_test,
                                     epochs=100, batch_size=16, model_path="diabetes_fnn.keras")

    loss, acc = evaluate_model(model, X_test_scaled, y_test)
    print(f"Test loss: {loss:.4f}, Test accuracy: {acc*100:.2f}%")

    model_path, scaler_path = save_model_and_scaler(model, scaler, "diabetes_fnn.keras", "scaler.pkl")

    loaded_model, loaded_scaler = load_model_and_scaler(model_path, scaler_path)

    sample = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 80,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.35,
        "Age": 35
    }
    prob = predict_new_data(loaded_model, loaded_scaler, sample)
    print("Sample prediction probability:", prob)

    run_gui(loaded_model, loaded_scaler)

if __name__ == "__main__":
    main()
