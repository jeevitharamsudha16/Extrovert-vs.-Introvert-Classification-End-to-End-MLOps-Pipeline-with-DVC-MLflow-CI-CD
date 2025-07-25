#data_loader.py
# Import the pandas library for working with data

import pandas as pd

# Define a function to load the heart.csv file
def load_data(path='data/personality_dataset.csv'):

    try:
        # Try to read the CSV file from the given path
        df = pd.read_csv(path)

        # If successful, print a confirmation message
        print("✅ Data loaded successfully!")

        # Return the loaded DataFrame (table of data)
        return df

    # If the file is not found at the given path, show an error
    except FileNotFoundError:
        print(f"❌ File not found at: {path}")


## DVC set up
'''
pip install dvc
git init
dvc init
git add .dvc .gitignore
git commit -m "Initialize DVC"
dvc add data/personality_datasert.csv
git add data/personality_datasert.csv.dvc .gitignore
git commit -m "Track dataset using DVC"
git branch -M main
git remote add origin https://github.com/jeevitharamsudha16/Extrovert-vs.-Introvert-Classification-End-to-End-MLOps-Pipeline-with-DVC-MLflow-CI-CD.git
git pull origin main --allow-unrelated-histories
git push -u origin main
'''