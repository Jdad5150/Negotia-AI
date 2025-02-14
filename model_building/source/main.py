"""
Main script for creating and training a machine learning model using Keras, encoding categorical features,
and saving the model for inference with TensorFlow.js.

The script performs the following tasks:
1. Creates a synthetic dataset with job titles, states, salaries, work type, and experience.
2. Encodes categorical features ('title', 'state', 'work_type') using LabelEncoder and saves the encoding maps as JSON files.
3. Splits the data into features (X) and target (y), scales the features using StandardScaler.
4. Builds a neural network model using Keras to predict job salaries based on encoded features.
5. Trains the model, performs early stopping to avoid overfitting, and saves the trained model.

Dependencies:
- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib
- seaborn

Functions included:
- create_date: Generates a synthetic dataset for job salary prediction.
- encode_and_save: Encodes a specified column in the DataFrame using LabelEncoder and saves the encoding map to a JSON file.

Author: Jesse Little
Date: 02/06/2025
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout# type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

seed = 42
data_name = "cleaned_data.parquet"
data_path = os.path.join("data", data_name)

def load_data():
    """
    Load the dataset from the specified path.

    Returns:
    - df: pandas DataFrame containing the data.
    """
    df = pd.read_parquet(data_path)
    return df


def encode_and_save(
    df, column_name, save_dir="../shared", file_name="encoding.json"
):
    """
    Encodes the specified column in the DataFrame using LabelEncoder,
    then saves the encoding map to a JSON file.

    Parameters:
    - df: pandas DataFrame containing the data.
    - column_name: the name of the column to encode.
    - save_dir: directory to save the encoding map JSON file (default is '../../shared').
    - file_name: name of the JSON file to save the encoding map (default is 'encoding.json').

    Returns:
    - df: DataFrame with the encoded column.
    """
    # Initialize the LabelEncoder
    le = LabelEncoder()

    # Fit and transform the specified column
    df[column_name] = le.fit_transform(df[column_name])

    # Create the encoding map (labels to encoding)
    encoding_map = dict(zip(le.classes_, le.transform(le.classes_)))

    # Convert numpy.int64 to regular Python int
    encoding_map = {k: int(v) for k, v in encoding_map.items()}

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define the file path to save the encoding map
    file_path = os.path.join(save_dir, file_name)

    # Write the encoding map to a JSON file
    with open(file_path, "w") as f:
        json.dump(encoding_map, f)

    # Optionally, print to confirm the file is saved
    print(f"Encoding map saved to: {file_path}")

    # Return the DataFrame with the encoded column
    return df



if __name__ == "__main__":
    # Load the data
    df = load_data()
    print(df.head(5))
    print(df.info())
    
    # # Clean data
    df = df.dropna()


    # Encode the categorical columns
    df = encode_and_save(df, "job_title", file_name="title_encoding.json")
    df = encode_and_save(df, "state", file_name="state_encoding.json")
    df = encode_and_save(df, "experience_level", file_name="exp_level_encoding.json")
    df = encode_and_save(df, "work_type", file_name="work_type_encoding.json")


    # Split the data
    X = df.drop("salary", axis=1)
    y = df["salary"]

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit and transform the features
    X = scaler.fit_transform(X)

    # Save the scaler for inference
    joblib.dump(scaler, "../shared/scaler.pkl")
    print("Scaler saved to: ../shared/scaler.pkl")


    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    # Hyperparameters
    n_neurons = 128
    n_layers = 3  # Number of hidden layers
    embedding_dim = 10  # Size of embedding vectors
    n_epochs = 100
    batch_size = 32

    # Number of unique categories for embeddings
    n_job_titles = df["job_title"].nunique() + 1  # Add 1 for unseen values
    n_states = df["state"].nunique() + 1

    # Define model inputs
    job_title_input = Input(shape=(1,), name="job_title_input")
    state_input = Input(shape=(1,), name="state_input")
    numeric_inputs = Input(shape=(X_train.shape[1] - 2,), name="numeric_inputs")  # Exclude categorical features

    # Create embedding layers
    job_title_embedded = Embedding(input_dim=n_job_titles, output_dim=embedding_dim)(job_title_input)
    state_embedded = Embedding(input_dim=n_states, output_dim=embedding_dim)(state_input)

    # Flatten embeddings
    job_title_embedded = Flatten()(job_title_embedded)
    state_embedded = Flatten()(state_embedded)

    # Concatenate all features
    x = Concatenate()([job_title_embedded, state_embedded, numeric_inputs])

    # Dynamically add hidden layers
    for i in range(1, n_layers + 1):
        x = Dense(units=n_neurons, activation="relu")(x)
        x = Dropout(0.2)(x)

    # Output layer
    output = Dense(units=1)(x)

    # Create the model
    model = Model(inputs=[job_title_input, state_input, numeric_inputs], outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error", metrics=["mse"])

    # Print model summary
    model.summary()

    print("Job Title Shape:", X_train["job_title"].values.shape, "Type:", type(X_train["job_title"].values))
    print("State Shape:", X_train["state"].values.shape, "Type:", type(X_train["state"].values))
    print("Numerical Features Shape:", X_train.drop(columns=["job_title", "state"]).values.shape, "Type:", type(X_train.drop(columns=["job_title", "state"]).values))
    print("Target Shape:", y_train.values.shape, "Type:", type(y_train.values))


    # Train the model
    history = model.fit(
        [X_train[:, 0], X_train[:,1], X_train[:,2:]],
        y_train,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_data=(
            [X_test[:, 0], X_test[:, 1], X_test[:, 2:]],
            y_test,
        ),
        verbose=1,
    )

    # ## Save the model
    # model.save("../../shared/demo_model.keras")

    # # import tensorflowjs as tfjs
    # # tfjs.converters.save_keras_model(model, '../../shared/demo_model')
