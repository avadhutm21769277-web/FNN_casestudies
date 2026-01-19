################################################################################
#   REQUIRED LIBRARIES
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense


################################################################################
#   FUNCTION 1 : load_data()
################################################################################

def load_data(dataset):
    df = pd.read_csv(dataset)
    return df


################################################################################
#   FUNCTION 2 : name_of_headers()
################################################################################

def name_of_headers(df):
    return df.columns[:]


################################################################################
#   FUNCTION 3 : feature_header_of_dataset()
################################################################################

def feature_header_of_dataset(df):
    return df.columns[:3]


################################################################################
#   FUNCTION 4 : target_header_of_dataset()
################################################################################

def target_header_of_dataset(df):
    return df.columns[3:]


################################################################################
#   FUNCTION 5 : datatypes_of_headers()
################################################################################

def datatypes_of_headers(df):
    return df.dtypes


################################################################################
#   FUNCTION 6 : checking_null_values()
################################################################################

def checking_null_values(df):
    return df.isnull().sum()


################################################################################
#   FUNCTION 7 : statistical_info()
################################################################################

def statistical_info(df):
    return df.describe()


################################################################################
#   FUNCTION 8 : boxplot_of_dataset()
################################################################################

def boxplot_of_dataset(df):
    plt.figure(figsize=(8, 6))
    df.plot(kind="box", subplots=True, layout=(4, 4), sharex=False, sharey=False, figsize=(15, 10))
    plt.suptitle("Boxplots of Features")
    plt.savefig("boxplot.png")
    plt.close()
    print("Boxplot of dataset created successfully")


################################################################################
#   FUNCTION 9 : remove_unnecessary_columns()
################################################################################

def remove_unnecessary_columns(df):
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df


################################################################################
#   FUNCTION 10 : split_into_feature_and_target()
################################################################################

def split_into_feature_and_target(df):
    X = df.drop(columns='sales')
    Y = df['sales']
    return X, Y


################################################################################
#   FUNCTION 11 : split_dataset()
################################################################################

def split_dataset(X, Y):
    return train_test_split(X, Y, test_size=0.3, random_state=42)


################################################################################
#   FUNCTION 12 : scaled_data()
################################################################################

def scaled_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


################################################################################
#   FUNCTION 13 : build_fnn_model()
################################################################################

def build_fnn_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


################################################################################
#   FUNCTION 14 : train_and_evaluate_fnn()
################################################################################

def train_and_evaluate_fnn(X_train_scaled, X_test_scaled, Y_train, Y_test):
    model = build_fnn_model(X_train_scaled.shape[1])
    model.fit(X_train_scaled, Y_train, epochs=100, batch_size=10, verbose=0)

    Y_pred = model.predict(X_test_scaled)
    r2 = r2_score(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)

    return model, r2, mse


################################################################################
#   FUNCTION 15 : save_fnn_model()
################################################################################

def save_fnn_model(model, filename):
    model.save(filename)
    print(f"Model saved successfully as {filename}")


################################################################################
#   FUNCTION 16 : load_fnn_model()
################################################################################

def load_fnn_model(filename):
    return load_model(filename)


################################################################################
#   FUNCTION 17 : predict_new_data()
################################################################################

def predict_new_data(scaler, loaded_model):
    new_data = pd.DataFrame({
        'TV': [261.3],
        'radio': [42.7],
        'newspaper': [54.7]
    })
    new_scaled_data = scaler.transform(new_data)
    prediction = loaded_model.predict(new_scaled_data)
    return prediction


################################################################################
#   FUNCTION 18 : main()
################################################################################

def main():
    print("ADVERTISEMENT PREDICTOR MODEL (FNN BASED)\n")

    dataset = 'Advertising.csv'

    # 1. Load dataset
    df = load_data(dataset)
    print("DataFrame created successfully\n", df.head(), "\n")

    # 2. Basic info
    print("Headers:", name_of_headers(df))
    print("Feature Headers:", feature_header_of_dataset(df))
    print("Target Header:", target_header_of_dataset(df))
    print("Data Types:\n", datatypes_of_headers(df))
    print("Null Values:\n", checking_null_values(df))
    print("Statistical Info:\n", statistical_info(df), "\n")

    # 3. Visualization
    boxplot_of_dataset(df)

    # 4. Data cleanup
    df = remove_unnecessary_columns(df)

    # 5. Feature-target split
    X, Y = split_into_feature_and_target(df)

    # 6. Train-test split
    X_train, X_test, Y_train, Y_test = split_dataset(X, Y)

    # 7. Scaling
    scaler, X_train_scaled, X_test_scaled = scaled_data(X_train, X_test)

    # 8. Model training
    model, r2, mse = train_and_evaluate_fnn(X_train_scaled, X_test_scaled, Y_train, Y_test)
    print(f"Model R2 Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

    # 9. Save and load model
    filename = "advertising_fnn.keras"
    save_fnn_model(model, filename)

    loaded_model = load_fnn_model(filename)
    print("Model loaded successfully!")

    # 10. Predict new data
    prediction = predict_new_data(scaler, loaded_model)
    print("Predicted sales value for new data:", prediction[0][0])


################################################################################
#   CALLING MAIN FUNCTION
################################################################################

if __name__ == "__main__":
    main()
