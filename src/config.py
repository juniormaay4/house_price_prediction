# src/config.py

import os

# --- Chemins des Données ---
# Le répertoire actuel de config.py est 'src/'
# Pour atteindre 'data/processed' qui est au même niveau que 'src/',
# nous remontons d'un niveau (..) puis redescendons dans 'data/processed'.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Chemin vers le dossier 'src'
PROJECT_ROOT = os.path.join(BASE_DIR, '..') # Chemin vers la racine du projet

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw') # Si vous avez un dossier pour les données brutes

TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'test.csv')

# --- Chemin du Modèle Sauvegardé ---
# Le dossier 'models' est également à la racine du projet
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', 'xgboost_model.pkl') # <--- MODIFIEZ CECI

# --- Colonne Cible ---
TARGET_COLUMN = 'price' # Nom de la colonne que vous voulez prédire

# --- Caractéristiques pour l'Ingénierie ---
# Ces listes sont des bases; elles seront ajustées dans features_engineering.py
# en fonction des colonnes réellement disponibles et créées.

# Caractéristiques numériques initiales du dataset brut
INITIAL_NUMERICAL_FEATURES = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
    'yr_built', 'yr_renovated'
]

# Caractéristiques catégorielles initiales du dataset brut
INITIAL_CATEGORICAL_FEATURES = [
    # 'street', 'city', 'statezip', 'country' - Ces colonnes sont souvent droppées
    # après ingénierie ou nécessitent un traitement spécifique comme l'embedding.
    # Pour l'instant, si elles sont utilisées comme catégorielles, listez-les ici.
    # Sinon, assurez-vous qu'elles sont dans FEATURES_TO_DROP_AFTER_ENGINEERING
]

# Log file path
APP_LOG_FILE = 'logs/app.log'

# Colonnes à retirer après l'ingénierie des caractéristiques
# (par exemple, 'date' après extraction de l'année/mois, ou identifiants)
FEATURES_TO_DROP_AFTER_ENGINEERING = [
    'date', # La colonne de date brute
    'street', 'city', 'statezip', 'country' # Adresses détaillées souvent non utilisées directement
]

# --- Stratégies d'Imputation ---
NUM_IMPUTER_STRATEGY = 'mean' # 'mean', 'median', 'most_frequent'
CAT_IMPUTER_STRATEGY = 'most_frequent' # 'most_frequent', 'constant'

# --- Listes des Caractéristiques Finales (Mises à jour par features_engineering.py) ---
# Ces listes sont utilisées par model.py pour définir le preprocessor.
# Elles seront remplies dynamiquement par la fonction feature_engineer_data
# en fonction des caractéristiques finales sélectionnées/créées.
_final_numerical_features = []
_final_categorical_features = []