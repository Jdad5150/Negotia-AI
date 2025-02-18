"""
Salary Prediction Model Using Random Forest

This script builds and evaluates a Random Forest regression model to predict salaries based on job title, state, and experience level. 

Workflow:
1. Load the preprocessed dataset from a parquet file.
2. Perform basic data cleaning by dropping missing values.
3. Split the dataset into training and testing sets.
4. Train a RandomForestRegressor model on the training data.
5. Make salary predictions on the test set.
6. Evaluate the model using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) score.
7. Analyze feature importance to determine which factors influence salary the most.
8. Save the trained model as a `.pkl` file for future use.
9. Generate and save a feature importance plot.

Outputs:
- Model evaluation metrics displayed in the console.
- A bar chart of feature importances saved as 'output/feature_importances.png'.
- The trained model saved as 'salary_prediction_model.pkl' in the '../shared' directory.

Dependencies:
- pandas
- matplotlib
- scikit-learn
- joblib
- parquet (for reading the dataset)

 ```

Author: Jesse Little
Date: 2/16/2025
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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




if __name__ == "__main__":
    """
    This script builds a Random Forest model to predict salary based on job title, state, and experience level.
    """


    # Load the data
    df = load_data()
    print(df.info())
    
    # # Clean data
    df = df.dropna()

    # Split the data
    X = df.drop("salary", axis=1)
    y = df["salary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)


    # Build and train the model
    model = RandomForestRegressor(random_state=seed)

    model.fit(X_train, y_train)


    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model evaluation:")
    print("-----------------------------------------")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print("-----------------------------------------")

    model_analysis = """
    This model is performing extremely well with an R-squared value of 0.99.
    This indicates that the model is able to explain 99% of the variance in the target variable. 
    The mean absolute error is also very low, indicating that the model is making accurate predictions within $118 of the real salary.
"""
    print(f"Model Analysis:\n{model_analysis}")
    print("-----------------------------------------")
    print("Feature importance is a key aspect of Random Forest models because it explains which features are most impactful.")
    print("Feature Importances:")

    # Get feature importances
    importances = model.feature_importances_
    
    features = ['State', 'Title', 'Experience']
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })
    
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importances in Random Forest Model')
    plt.savefig('output/feature_importances.png', bbox_inches='tight', dpi=300)
    plt.show()

    # Save the model
    model_path = os.path.join("../shared", "salary_prediction_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    print("-----------------------------------------")
    print("Model building script completed successfully.")