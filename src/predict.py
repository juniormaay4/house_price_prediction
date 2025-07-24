# src/predict.py
import joblib
import pandas as pd
import numpy as np
import os
import logging
import sys

# Ajoutez le répertoire parent au sys.path pour permettre les imports depuis src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features_engineering import feature_engineer_data
from src import config as config
from src.data_preparation import load_data # <--- C'EST CETTE LIGNE QUI DOIT ÊTRE MODIFIÉE/AJOUTÉE

# Define the log file path for predictions
PREDICTION_LOG_FILE = 'logs/prediction_logs.log'

# Ensure the log directory exists
log_dir = os.path.dirname(PREDICTION_LOG_FILE)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Setup logging for the prediction script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(PREDICTION_LOG_FILE), logging.StreamHandler()])

def load_model(model_path=config.MODEL_SAVE_PATH):
    """
    Loads the trained machine learning model from the specified path.
    Args:
        model_path (str): The path to the saved model file.
    Returns:
        sklearn.pipeline.Pipeline: The loaded machine learning model.
    """
    if not os.path.exists(model_path):
        logging.error(f"Le fichier modèle n'a pas été trouvé à {model_path}. Veuillez vous assurer que le modèle est entraîné et sauvegardé.")
        raise FileNotFoundError(f"Le fichier modèle n'a pas été trouvé à {model_path}. Veuillez vous assurer que le modèle est entraîné et sauvegardé.")
    logging.info(f"Chargement du modèle depuis {model_path}...")
    try:
        model = joblib.load(model_path)
        logging.info("Modèle chargé avec succès.")
        return model
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle depuis {model_path}: {e}")
        raise

def make_prediction(model, input_data: pd.DataFrame, target_log_transformed: bool = True):
    """
    Makes predictions using the loaded model.
    Args:
        model: The trained machine learning model (Pipeline).
        input_data (pd.DataFrame): A DataFrame containing the raw features for prediction.
                                   It must have the same column names as the original dataset
                                   before feature engineering.
        target_log_transformed (bool): True if the model was trained on a log-transformed target,
                                       False otherwise. Defaults to True.
    Returns:
        np.array: An array of predicted values.
    """
    if input_data.empty:
        logging.warning("Les données d'entrée pour la prédiction sont vides.")
        raise ValueError("Les données d'entrée pour la prédiction ne peuvent pas être vides.")

    logging.info("Application de l'ingénierie des caractéristiques aux données d'entrée...")
    try:
        input_data_fe = feature_engineer_data(input_data.copy())
    except Exception as e:
        logging.error(f"Erreur lors de l'ingénierie des caractéristiques pour la prédiction : {e}")
        raise ValueError(f"L'ingénierie des caractéristiques a échoué pour les données d'entrée : {e}")

    logging.info("Réalisation de la prédiction...")
    try:
        predictions_transformed = model.predict(input_data_fe)
        logging.info("Prédiction terminée.")

        if target_log_transformed:
            logging.info("Inversion de la transformation logarithmique des prédictions...")
            predictions = np.expm1(predictions_transformed)
        else:
            predictions = predictions_transformed
        
        predictions[predictions < 0] = 0

        return predictions
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction du modèle : {e}")
        raise ValueError(f"La prédiction du modèle a échoué : {e}")

if __name__ == "__main__":
    try:
        loaded_model = load_model()
    except (FileNotFoundError, Exception) as e:
        logging.critical(f"Échec du chargement du modèle, impossible de poursuivre la prédiction : {e}")
        exit()

    logging.info("Préparation d'un échantillon de données d'entrée pour la démonstration.")
    try:
        # Charger les données de test pour simuler de nouvelles données non vues
        # ATTENTION : La fonction load_data() dans src/data_preparation.py retourne 4 valeurs:
        # X_train, y_train, X_test, y_test.
        # Ici, nous n'avons besoin que de X_test pour simuler les données d'entrée.
        # Assurez-vous que le 3ème élément retourné par load_data est bien votre X_test_raw.
        _, _, sample_input_data_raw, _ = load_data() # <--- C'EST CETTE LIGNE QUI DOIT ÊTRE MODIFIÉE

        # Prendre les 5 premières lignes pour une démo rapide, et retirer la colonne 'price' si présente
        sample_input_data = sample_input_data_raw.head(5).copy()
        if 'price' in sample_input_data.columns:
            sample_input_data = sample_input_data.drop(columns=['price'])
        
        logging.info(f"Échantillon de données d'entrée chargé avec la forme : {sample_input_data.shape}")
        logging.debug(f"Aperçu des données d'entrée :\n{sample_input_data.head()}")

    except Exception as e:
        logging.critical(f"Échec du chargement des données d'échantillon pour la prédiction : {e}")
        exit()

    try:
        predicted_prices = make_prediction(loaded_model, sample_input_data, target_log_transformed=True)
        logging.info(f"Prix prédits : {predicted_prices}")

        predictions_df = pd.DataFrame({
            'id': sample_input_data['id'] if 'id' in sample_input_data.columns else range(len(predicted_prices)),
            'predicted_price': predicted_prices
        })
        output_csv_path = "predictions/latest_predictions.csv"
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        predictions_df.to_csv(output_csv_path, index=False)
        logging.info(f"Prédictions sauvegardées dans {output_csv_path}")
        print(f"Prédictions sauvegardées dans {output_csv_path}")

    except ValueError as e:
        logging.error(f"La prédiction a échoué : {e}")
    except Exception as e:
        logging.error(f"Une erreur inattendue est survenue pendant la prédiction : {e}")