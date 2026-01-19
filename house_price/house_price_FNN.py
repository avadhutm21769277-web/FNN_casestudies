################################################################################
#   REQUIRED LIBRARIES
################################################################################

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

####################################################################################
#       function 1  :   load_data()
####################################################################################
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df

##################################################################################
#       function 2  :   name_of_headers()
##################################################################################
def name_of_headers(df):
    return df.columns[:]

#########################################################################################
#       function 3  :   feature_header_of_dataset()
##########################################################################################
def feature_header_of_dataset(df):
    feature_headers = df.columns[1:11]
    return feature_headers

#########################################################################################
#       function 4  :   target_header_of_dataset()
##########################################################################################
def target_header_of_dataset(df):
    target_header = df.columns[0:1]
    return target_header

#########################################################################################
#       function 5  :   preprocessing_pipeline()
##########################################################################################
def preprocessing_pipeline(df):
    feature_headers = feature_header_of_dataset(df)
    target_header = target_header_of_dataset(df)
    X = df[feature_headers]
    Y = df[target_header]

    numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    print("numeric_features :", numeric_features)
    print("categorical_features :", categorical_features)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    X_transformed = preprocessor.fit_transform(X)
    encoded_columns = list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))
    all_columns = numeric_features + encoded_columns

    X_transformed_df = pd.DataFrame(X_transformed, columns=all_columns)
    return X_transformed_df, Y, preprocessor

#########################################################################################
#       function 6  :   fnn_model_pipeline()
##########################################################################################
def fnn_model_pipeline(X_transformed_df, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X_transformed_df, Y, test_size=0.2, random_state=42)

    input_dim = X_train.shape[1]

    # Define Feedforward Neural Network
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Output layer (regression)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    Y_pred = model.predict(X_test)
    r2_value = r2_score(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)

    print("R² value:", r2_value)
    print("MSE:", mse)

    return model, r2_value

############################################################################
#   function 7       :   save_model()
############################################################################
def save_model(model, filename):
    model.save(filename)
    return filename

##############################################################################
#   function 8   :   load_model()
##############################################################################
def load_model(filename):
    loaded_model = tf.keras.models.load_model(filename)
    return loaded_model

##############################################################################
#   function 9   :   predict_new_data()
##############################################################################
def predict_new_data(loaded_model, preprocessor):
    new_data = pd.DataFrame({
        'area': [9960],
        'bedrooms': [3],
        'bathrooms': [2],
        'stories': [2],
        'mainroad': ['yes'],
        'guestroom': ['no'],
        'basement': ['yes'],
        'hotwaterheating': ['no'],
        'airconditioning': ['yes'],
        'parking': [2],
        'prefarea': ['yes'],
        'furnishingstatus': ['semi-furnished']
    })

    X_new_transformed = preprocessor.transform(new_data)
    predicted_price = loaded_model.predict(X_new_transformed)
    print("Predicted Price:", predicted_price[0][0])

##################################################################################
#       function 10   :   main()
##################################################################################
def main():
    print("HOUSE PRICE PREDICTOR — FEEDFORWARD NEURAL NETWORK (FNN)")

    dataset = 'Housing.csv'

    df = load_data(dataset)
    print("Data loaded successfully.")
    print(df.head())

    headers = name_of_headers(df)
    print("Headers:", headers)

    feature_headers = feature_header_of_dataset(df)
    print("Feature Headers:", feature_headers)

    target_header = target_header_of_dataset(df)
    print("Target Header:", target_header)

    X_transformed_df, Y, preprocessor = preprocessing_pipeline(df)

    model, r2value = fnn_model_pipeline(X_transformed_df, Y)
    print("FNN Model trained successfully with R²:", r2value)

    model_filename = 'houseprice_fnn_model.keras'
    save_model(model, model_filename)
    print("Model saved as:", model_filename)

    loaded_model = load_model(model_filename)
    print("Model loaded successfully!")

    predict_new_data(loaded_model, preprocessor)

if __name__ == "__main__":
    main()
