import pandas as pd
import numpy as np
import datetime

from src import config as config

def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait les caractéristiques de date pour XGBoost :
    - days_since_ref (numérique, pour la tendance)
    - sale_year (catégorielle, pour l'effet annuel)
    - sale_month (catégorielle, pour l'effet saisonnier)
    """
    if 'date' in df.columns:
        df['date_datetime'] = pd.to_datetime(df['date'], errors='coerce')
        initial_rows = len(df)
        df.dropna(subset=['date_datetime'], inplace=True)
        if len(df) < initial_rows:
            print(f"Attention : {initial_rows - len(df)} lignes ont été supprimées en raison de formats de date invalides.")

        if not df.empty:
            # Caractéristique temporelle continue: jours depuis un point de référence
            REFERENCE_DATE = datetime.datetime(2014, 5, 1) # Choisissez une date de référence pertinente
            df['days_since_ref'] = (df['date_datetime'] - REFERENCE_DATE).dt.days

            # Année et mois comme caractéristiques catégorielles (str) pour XGBoost
            df['sale_year'] = df['date_datetime'].dt.year.astype(str)
            df['sale_month'] = df['date_datetime'].dt.month.astype(str)
        else:
            print("Attention : Le DataFrame est vide après suppression des dates invalides. Aucune caractéristique de date extraite.")
    else:
        print("Attention : La colonne 'date' n'a pas été trouvée pour l'extraction de caractéristiques.")
    return df

def create_house_age_and_renovation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les caractéristiques 'house_age' et 'renovation_age'.
    Dépend de 'yr_built', 'yr_renovated' et 'sale_year' (qui doit être convertible en int pour le calcul).
    """
    # Convertir temporairement 'sale_year' en numérique pour les calculs d'âge
    # (elle redeviendra string dans la liste finale des catégories)
    temp_sale_year = pd.to_numeric(df['sale_year'], errors='coerce') if 'sale_year' in df.columns else None

    if 'yr_built' in df.columns and temp_sale_year is not None:
        df['house_age'] = temp_sale_year - df['yr_built']
        df['house_age'] = df['house_age'].apply(lambda x: max(x, 0)) # L'âge ne peut pas être négatif

        if 'yr_renovated' in df.columns:
            df['is_renovated'] = ((df['yr_renovated'] > df['yr_built']) & (df['yr_renovated'] > 0)).astype(int)
            df['renovation_age'] = temp_sale_year - df['yr_renovated']
            df['renovation_age'] = df['renovation_age'].apply(lambda x: max(x, 0) if x > 0 else 0)
        else:
            print("Attention : La colonne 'yr_renovated' n'a pas été trouvée pour les caractéristiques de rénovation.")
    else:
        print("Attention : 'yr_built' ou 'sale_year' non trouvées pour les caractéristiques d'âge.")
    return df

def combine_area_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des caractéristiques combinées et des ratios de surface.
    """
    if 'sqft_living' in df.columns and 'sqft_lot' in df.columns:
        df['sqft_ratio_living_lot'] = df['sqft_living'] / (df['sqft_lot'] + 1e-6)
    else:
        print("Attention : Manque 'sqft_living' ou 'sqft_lot' pour 'sqft_ratio_living_lot'.")

    if 'sqft_above' in df.columns and 'sqft_basement' in df.columns:
        df['building_total_sqft'] = df['sqft_above'] + df['sqft_basement']
    else:
        print("Attention : Manque 'sqft_above' ou 'sqft_basement' pour 'building_total_sqft'.")

    if 'bedrooms' in df.columns and 'sqft_living' in df.columns:
        df['sqft_living_per_bedroom'] = df['sqft_living'] / (df['bedrooms'] + 1e-6)
    if 'bathrooms' in df.columns and 'sqft_living' in df.columns:
        df['sqft_living_per_bathroom'] = df['sqft_living'] / (df['bathrooms'] + 1e-6)
    
    return df

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des caractéristiques d'interaction clés que XGBoost pourrait bénéficier de voir explicitement.
    Moins d'interactions sont nécessaires qu'avec la Régression Linéaire.
    """
    if 'sqft_living' in df.columns and 'grade' in df.columns:
        df['sqft_living_x_grade'] = df['sqft_living'] * df['grade']
    if 'waterfront' in df.columns and 'sqft_living' in df.columns:
        df['waterfront_x_sqft_living'] = df['waterfront'] * df['sqft_living']
    if 'lat' in df.columns and 'long' in df.columns:
        df['lat_x_long'] = df['lat'] * df['long']
    
    return df

def handle_categorical_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les caractéristiques numériques "similaires à des catégories" en chaînes de caractères.
    """
    for col in ['waterfront', 'view', 'condition', 'floors', 'bedrooms']:
        if col in df.columns:
            df[f'{col}_cat'] = df[col].astype(str)
        else:
            print(f"Attention : La colonne '{col}' n'a pas été trouvée pour la transformation catégorielle.")
    return df


def feature_engineer_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique toutes les étapes d'ingénierie de caractéristiques au DataFrame pour XGBoost.
    """
    df_engineered = df.copy()

    # Appliquer les fonctions d'ingénierie de caractéristiques
    df_engineered = extract_date_features(df_engineered)
    df_engineered = create_house_age_and_renovation_features(df_engineered)
    df_engineered = combine_area_features(df_engineered)
    df_engineered = handle_categorical_transformations(df_engineered)
    df_engineered = create_interaction_features(df_engineered)

    # --- Mise à jour dynamique de la configuration pour les caractéristiques ---
    current_numerical_features = config.INITIAL_NUMERICAL_FEATURES[:]
    current_categorical_features = config.INITIAL_CATEGORICAL_FEATURES[:]

    # Supprimer les caractéristiques numériques initiales qui sont maintenant transformées ou catégorielles
    features_to_remove_from_initial_num = ['date', 'yr_built', 'yr_renovated', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']
    for col in ['waterfront', 'view', 'condition', 'floors', 'bedrooms']:
        features_to_remove_from_initial_num.append(col)

    current_numerical_features = [f for f in current_numerical_features if f not in features_to_remove_from_initial_num]

    # Collecter TOUTES les nouvelles caractéristiques numériques ingéniérées
    new_engineered_numerical_features = [
        'days_since_ref', 'house_age', 'is_renovated', 'renovation_age',
        'sqft_ratio_living_lot', 'building_total_sqft', 'sqft_living_per_bedroom', 'sqft_living_per_bathroom',
        'sqft_living_x_grade', 'waterfront_x_sqft_living', 'lat_x_long'
    ]
    for feat in new_engineered_numerical_features:
        if feat in df_engineered.columns and feat not in current_numerical_features:
            current_numerical_features.append(feat)

    # Collecter TOUTES les nouvelles caractéristiques catégorielles ingéniérées
    new_engineered_categorical_features = [
        'waterfront_cat', 'view_cat', 'condition_cat', 'floors_cat', 'bedrooms_cat',
        'sale_year', 'sale_month'
    ]
    for feat in new_engineered_categorical_features:
        if feat in df_engineered.columns and feat not in current_categorical_features:
            current_categorical_features.append(feat)

    # Listes finales pour le ColumnTransformer
    config._final_numerical_features = [f for f in current_numerical_features if f in df_engineered.columns]
    config._final_categorical_features = [f for f in current_categorical_features if f in df_engineered.columns]

    # --- Supprimer les colonnes originales qui ne sont plus nécessaires ---
    columns_to_drop_now = config.FEATURES_TO_DROP_AFTER_ENGINEERING + ['date_datetime', 'date']
    for col in ['waterfront', 'view', 'condition', 'floors', 'bedrooms']:
        if f'{col}_cat' in config._final_categorical_features and col in df_engineered.columns and col not in columns_to_drop_now:
            columns_to_drop_now.append(col)
    
    if 'id' in df_engineered.columns and 'id' not in columns_to_drop_now:
        columns_to_drop_now.append('id')

    df_processed = df_engineered.drop(columns=columns_to_drop_now, errors='ignore')

    return df_processed