import os
import pandas as pd
import numpy as np
import xgboost as xgb # type: ignore
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib
import logging

from src import config
from src.data_preparation import load_data
from src.features_engineering import feature_engineer_data

# Setup logging
if not os.path.exists(os.path.dirname(config.APP_LOG_FILE)):
    os.makedirs(os.path.dirname(config.APP_LOG_FILE))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(config.APP_LOG_FILE), logging.StreamHandler()])


def get_preprocessor(numerical_features, categorical_features):
    """
    Définit et retourne le pipeline de prétraitement utilisant ColumnTransformer.
    """
    numerical_transformers = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=config.NUM_IMPUTER_STRATEGY)),
        ('scaler', StandardScaler())
    ])

    categorical_transformers = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=config.CAT_IMPUTER_STRATEGY)),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_pipeline', numerical_transformers, numerical_features),
            ('cat_pipeline', categorical_transformers, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor

def train_and_save_model(model_path=config.MODEL_SAVE_PATH):
    logging.info("--- Démarrage de l'entraînement du modèle ---")
    logging.info("Chargement des données...")
    X_train_raw, y_train, X_test_raw, y_test = load_data()
    logging.info(f"Forme de X_train_raw : {X_train_raw.shape}, y_train : {y_train.shape}")
    logging.info(f"Forme de X_test_raw : {X_test_raw.shape}, y_test : {y_test.shape}")


    # --- Application de la transformation logarithmique à la variable cible ---
    logging.info("Application de la transformation logarithmique à la variable cible (prix)...")
    y_train_transformed = np.log1p(y_train)
    y_test_transformed = np.log1p(y_test)

    logging.info("Application de l'ingénierie des caractéristiques aux données d'entraînement...")
    X_train_fe = feature_engineer_data(X_train_raw)
    logging.info(f"Forme de X_train_fe après ingénierie (avant préprocesseur) : {X_train_fe.shape}")

    final_numerical_features = config._final_numerical_features
    final_categorical_features = config._final_categorical_features

    logging.info(f"Caractéristiques Numériques Finales pour le Préprocesseur ({len(final_numerical_features)}): {final_numerical_features}")
    logging.info(f"Caractéristiques Catégorielles Finales pour le Préprocesseur ({len(final_categorical_features)}): {final_categorical_features}")

    preprocessor = get_preprocessor(final_numerical_features, final_categorical_features)

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(objective='reg:squarederror',
                                       n_estimators=500,
                                       learning_rate=0.05,
                                       max_depth=5,
                                       subsample=0.8,
                                       colsample_bytree=0.8,
                                       random_state=42,
                                       n_jobs=-1))
    ])

    logging.info("Entraînement du modèle XGBoost sur la cible transformée...")
    model_pipeline.fit(X_train_fe, y_train_transformed)
    logging.info("Entraînement du modèle XGBoost terminé.")

    logging.info("\nApplication de l'ingénierie des caractéristiques aux données de test...")
    X_test_fe = feature_engineer_data(X_test_raw)
    logging.info(f"Forme de X_test_fe après ingénierie (avant préprocesseur) : {X_test_fe.shape}")

    logging.info("\nPrédiction sur l'ensemble de test...")
    y_pred_transformed = model_pipeline.predict(X_test_fe)

    logging.info("Inversion de la transformation logarithmique pour obtenir les prédictions réelles...")
    y_pred = np.expm1(y_pred_transformed)

    logging.info("\nÉvaluation du modèle sur l'ensemble de test (avec les prix réels)...")
    if y_test is not None:
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        logging.info(f"Erreur Absolue Moyenne (MAE) : {mae:.2f}")
        logging.info(f"Erreur Quadratique Moyenne (MSE) : {mse:.2f}")
        logging.info(f"Racine Carrée de l'Erreur Quadratique Moyenne (RMSE) : {rmse:.2f}")
        logging.info(f"R-carré (R2) : {r2:.2f}")
    else:
        logging.warning("y_test non disponible pour l'évaluation.")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_pipeline, model_path)
    logging.info(f"Modèle sauvegardé avec succès dans {model_path}")
    logging.info("--- Entraînement du modèle terminé ---")

if __name__ == "__main__":
    train_and_save_model()