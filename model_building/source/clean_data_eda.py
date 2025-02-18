"""
This script preprocesses salary data, performs encoding, augments data, and saves the cleaned dataset.

Key Steps:
1. Load raw salary data from a CSV file.
2. Handle missing values and clean salary values.
3. Encode categorical columns (job title, state, experience level) using Label Encoding.
4. Save encoding mappings as JSON files for reference.
5. Generate exploratory visualizations:
   - Salary distribution histogram
   - Correlation heatmap
   - Pairplot of all features
6. Augment the dataset by applying small random variations to salary values.
7. Save the processed dataset as a Parquet file for model training.

Outputs:
- Encoded feature mappings in '../shared/' (e.g., 'title_encoding.json').
- Visualization images in 'output/' directory.
- Cleaned and augmented dataset saved as 'data/cleaned_data.parquet'.

Dependencies:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- json
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import json
from sklearn.preprocessing import LabelEncoder

file_name = 'all_data_M_2023.csv'
file_path = os.path.join('data', file_name)

def load_data():
    """
    Load the dataset from the specified path.

    Returns:
    - df: pandas DataFrame containing the data.
    """
    df = pd.read_csv(file_path)
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



if __name__ == '__main__':
    df = load_data()
    
    input_features = ['title', 'state', 'experience']
    target = 'salary'

    # Drop rows with missing values
    df = df.dropna(subset=input_features + [target])

    # Remove the dollar sign and commas from the target variable and convert it to float
    df[target] = df[target].apply(lambda x: x.replace('$', '').replace(',', '')).astype(float)

    # Encode the categorical columns
    df = encode_and_save(df, "title", file_name="title_encoding.json")
    df = encode_and_save(df, "state", file_name="state_encoding.json")
    df = encode_and_save(df, "experience", file_name="experience_encoding.json")


    # Plot the distribution of the target variable
    plt.figure(figsize=(16, 9))
    sns.displot(df[target], kde=True, color='blue')
    plt.title("Distribution of Salary")  
    plt.xlabel("Salary (USD)")
    plt.ylabel("Frequency")  
    plt.savefig('output/salary_distribution.png', bbox_inches='tight', dpi=300)
    plt.show()
    print('Distribution Analysis:\nThis distribution is right-skewed, which is expected for salary data.')

    # Correlation plot
    plt.figure(figsize=(9,9))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig('output/correlation_matrix.png', bbox_inches='tight', dpi=300)
    plt.show()
    print('Correlation Analysis:\nThere is a positive correlation between salary and experience. The rest of the features have a weak correlation with salary, this is due to label encoding')

    # Pairplot
    sns.pairplot(df)
    plt.savefig('output/pairplot.png', bbox_inches='tight', dpi=300)
    plt.show()
    print('Pairplot Analysis:\nThis pairplot shows the distribution of the features with respect to each other. The diagonal shows the distribution of each feature.')


    # Augment the data
    n_augmented = 50
    augmented_data = []

    # Loop through each row of data
    for index, row in df.iterrows():
        title = row['title']
        state = row['state']
        experience = row['experience']
        salary = row['salary']
        
        # Create n_augmented augmented samples for each row
        for _ in range(n_augmented):
            random_percentage = np.random.uniform(0.997, 1.003)
            augmented_salary = salary * random_percentage

            augmented_title = title
            augmented_state = state
            augmented_experience = experience

            # Append the augmented data to the list
            augmented_data.append({
                'state': augmented_state,
                'title': augmented_title,                
                'experience': augmented_experience,
                'salary': augmented_salary
            })

    # Convert to a dataframe and add to the original data
    augmented_df = pd.DataFrame(augmented_data)      
    combined_df = pd.concat([df, augmented_df], ignore_index=True)  

    # Save as parquet file
    combined_df.to_parquet("data/cleaned_data.parquet", index=False)    