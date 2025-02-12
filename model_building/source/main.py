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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

seed = 42


def create_date():
    """
    Create a synthetic dataset for job salary prediction.
    """
    # Define the job titles
    job_titles = [
        "Software Engineer",
        "Data Scientist",
        "Project Manager",
        "Machine Learning Engineer",
        "UI/UX Designer",
        "Product Manager",
        "DevOps Engineer",
        "System Administrator",
        "Cloud Architect",
        "Database Administrator",
    ]

    # Create the data
    data = pd.DataFrame(
        {
            "title": np.random.choice(job_titles, 1000),
            "state": np.random.choice(["CA", "NY", "TX", "WA", "FL"], 1000),
            "salary": np.random.randint(50000, 200000, 1000),
            "experience": np.random.randint(0, 6, 1000),
            "work_type": np.random.choice(
                ["Full-time", "Part-time", "Contract", "Remote"], 1000
            ),
        }
    )

    return data


def encode_and_save(
    df, column_name, save_dir="../../shared", file_name="encoding.json"
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
    df = create_date()
    print(df.head(5))

    # Clean data
    df = df.dropna()

    # Encode the categorical columns
    df = encode_and_save(df, "title", file_name="title_encoding.json")
    df = encode_and_save(df, "state", file_name="state_encoding.json")
    df = encode_and_save(df, "experience", file_name="exp_level_encoding.json")
    df = encode_and_save(df, "work_type", file_name="work_type_encoding.json")

    # Split the data
    X = df.drop("salary", axis=1)
    y = df["salary"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    # Create the model
    model = Sequential()
    # Input layer
    model.add(Dense(64, input_shape=(4,)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Hidden layers
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Output layer (regression)
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0,
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.summary()

    ## Save the model
    model.save("../../shared/demo_model.keras")

    # import tensorflowjs as tfjs
    # tfjs.converters.save_keras_model(model, '../../shared/demo_model')
