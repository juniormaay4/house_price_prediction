# src/data_preparation.py

import os
import sys

# *** CES LIGNES DOIVENT ÊTRE LES TOUTES PREMIÈRES APRÈS LES IMPORTS DE OS ET SYS ***
# Add the parent directory (project root) to sys.path
# This ensures 'config.py' can be found when imported by this module.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ********************************************************************************

import pandas as pd
#import config # Maintenant, cet import devrait fonctionner
from src import config

def load_data(train_path=config.TRAIN_DATA_PATH, test_path=config.TEST_DATA_PATH, target_column=config.TARGET_COLUMN):
    """
    Loads training and testing data from the specified paths.

    Args:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the testing CSV file.
        target_column (str): The name of the target column.

    Returns:
        tuple: X_train, y_train, X_test, y_test DataFrames.
                y_test might be None if the test set doesn't contain the target.
    """
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading data: {e}. Make sure '{train_path}' and '{test_path}' exist.")
    except Exception as e:
        raise Exception(f"An error occurred while reading CSV files: {e}")

    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data. Available columns: {train_df.columns.tolist()}")

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # Assume test_df may or may not have the target column
    X_test = test_df.drop(columns=[target_column]) if target_column in test_df.columns else test_df
    y_test = test_df[target_column] if target_column in test_df.columns else None

    print(f"Loaded data:")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   X_test shape: {X_test.shape}")
    print(f"   y_test shape: {y_test.shape if y_test is not None else 'N/A'}")

    return X_train, y_train, X_test, y_test