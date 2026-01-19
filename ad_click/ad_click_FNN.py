#################################################################################
#   AD CLICK PREDICTOR USING FEEDFORWARD NEURAL NETWORK (FNN)
#################################################################################

# ==================== REQUIRED LIBRARIES ====================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ============================================================


#################################################################################
#   FUNCTION 1 : load_data()
#################################################################################

def load_data(dataset):
    df = pd.read_csv(dataset)
    return df


#################################################################################
#   FUNCTION 2 : header_of_dataset()
#################################################################################

def header_of_dataset(df):
    headers = df.columns[:]
    return headers


#################################################################################
#   FUNCTION 3 : feature_headers_of_dataset()
#################################################################################

def feature_headers_of_dataset(df):
    feature_headers = df.columns[0:8]
    return feature_headers


#################################################################################
#   FUNCTION 4 : target_header_of_dataset()
#################################################################################

def target_header_of_dataset(df):
    target_header = df.columns[8:]
    return target_header


#################################################################################
#   FUNCTION 5 : datatypes_of_headers()
#################################################################################

def datatypes_of_headers(df):
    datatypes = df.dtypes
    return datatypes


#################################################################################
#   FUNCTION 6 : drop_unnecessary_columns()
#################################################################################

def drop_unnecessary_columns(df):
    df = df.drop(columns=['id', 'full_name'], inplace=False)
    return df


#################################################################################
#   FUNCTION 7 : replece_null_values()
#################################################################################

def replece_null_values(df):
    print(df.isnull().sum())
    df = df.replace(np.nan, 'missing')
    return df


#################################################################################
#   FUNCTION 8 : encoding_by_map()
#################################################################################

def encoding_by_map(df):

    df['age'] = df['age'].replace("missing", -1)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    df['gender'] = df['gender'].map({'Male': 1, 'Female': 2, 'Non-Binary': 3, 'missing': -1})
    df['device_type'] = df['device_type'].map({'Desktop': 1, 'Mobile': 2, 'Tablet': 3, 'missing': -1})
    df['ad_position'] = df['ad_position'].map({'Top': 1, 'Side': 2, 'Bottom': 3, 'missing': -1})
    df['browsing_history'] = df['browsing_history'].map(
        {'Shopping': 1, 'Education': 2, 'Entertainment': 3, 'Social Media': 4, 'News': 5, 'missing': -1})
    df['time_of_day'] = df['time_of_day'].map({'Afternoon': 1, 'Night': 2, 'Evening': 3, 'Morning': 4, 'missing': -1})

    return df


#################################################################################
#   FUNCTION 9 : boxplot_visualisation()
#################################################################################

def boxplot_visualisation(df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(df)
    plt.title("Boxplot of ad_click")
    plt.savefig("boxplot_age.png")
    plt.close()


#################################################################################
#   FUNCTION 10 : split_data_vertically()
#################################################################################

def split_data_vertically(df, target_header):
    X = df.drop(columns=target_header, axis=1)
    Y = df[target_header].values.ravel()
    return X, Y


#################################################################################
#   FUNCTION 11 : split_data_into_four_parts()
#################################################################################

def split_data_into_four_parts(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.7, random_state=42)
    return X_train, X_test, Y_train, Y_test


#################################################################################
#   FUNCTION 12 : using_standerd_scaler()
#################################################################################

def using_standerd_scaler(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


#################################################################################
#   FUNCTION 13 : fnn_classifier()  --> Neural Network
#################################################################################

def fnn_classifier(X_train_scaled, X_test_scaled, Y_train, Y_test, input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # binary classification
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("\nTraining Feedforward Neural Network...\n")
    model.fit(X_train_scaled, Y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    Y_pred_probs = model.predict(X_test_scaled)
    Y_pred = (Y_pred_probs > 0.5).astype(int)

    model_accuracy = accuracy_score(Y_test, Y_pred)
    print(f"\nModel Accuracy: {model_accuracy * 100:.2f}%")

    return model, model_accuracy


#################################################################################
#   FUNCTION 14 : save_fnn_model()
#################################################################################

def save_fnn_model(model, filename):
    model.save(filename)
    print(f"Model saved successfully as {filename}")
    return filename


#################################################################################
#   FUNCTION 15 : load_fnn_model()
#################################################################################

def load_fnn_model(filename):
    model = load_model(filename)
    print(f"Model loaded successfully from {filename}")
    return model


#################################################################################
#   FUNCTION 16 : predict_new_data_fnn()
#################################################################################

def predict_new_data_fnn(scaler, loaded_model):
    new_data = pd.DataFrame({
        'age': [28],
        'gender': [1],
        'device_type': [2],
        'ad_position': [1],
        'browsing_history': [3],
        'time_of_day': [2]
    })

    new_scaled = scaler.transform(new_data)
    pred_prob = loaded_model.predict(new_scaled)
    prediction = (pred_prob > 0.5).astype(int)
    print("Predicted Ad Click for new user:", int(prediction[0][0]))
    return prediction


#################################################################################
#   MAIN FUNCTION
#################################################################################

def main():
    print("CASE STUDY: AD CLICK PREDICTOR (Feedforward Neural Network)")

    df = load_data("ad_click_dataset.csv")
    print("Dataset loaded successfully.")
    print(df.head())
    print(df.shape)

    headers = header_of_dataset(df)
    print("Headers:", headers)

    feature_headers = feature_headers_of_dataset(df)
    print("Feature Headers:", feature_headers)

    target_header = target_header_of_dataset(df)
    print("Target Header:", target_header)

    datatypes = datatypes_of_headers(df)
    print("Data Types:\n", datatypes)

    df = drop_unnecessary_columns(df)
    print("Unnecessary columns removed successfully.")

    df = replece_null_values(df)
    print("Null values replaced successfully.")

    df = encoding_by_map(df)
    print("Encoding completed successfully.")

    boxplot_visualisation(df)
    print("Boxplot saved to current directory.")

    X, Y = split_data_vertically(df, target_header)
    print("Dataset split into features (X) and target (Y).")

    X_train, X_test, Y_train, Y_test = split_data_into_four_parts(X, Y)
    print("Data split into train/test sets.")

    scaler, X_train_scaled, X_test_scaled = using_standerd_scaler(X_train, X_test)
    print("Data scaled successfully.")

    model, model_accuracy = fnn_classifier(X_train_scaled, X_test_scaled, Y_train, Y_test,
                                           input_dim=X_train_scaled.shape[1])

    save_fnn_model(model, "adclick_fnn.h5")

    loaded_model = load_fnn_model("adclick_fnn.h5")

    predict_new_data_fnn(scaler, loaded_model)


#################################################################################
#   ENTRY POINT
#################################################################################

if __name__ == "__main__":
    main()
