# api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import load_model, make_prediction
from src import config as config 


try:
    model = load_model()
except FileNotFoundError as e:
    print(f"Error loading model: {e}. Please ensure the model is trained by running 'python -m src.model'.")
    model = None
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")
    model = None

app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices based on various features.",
    version="1.0.0",
)

origins = [
    "http://localhost:3000",
    "http://localhost:8000",
  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HouseFeatures(BaseModel):
   
    date: str = Field(..., example="2023-01-15", description="Date de vente (AAAA-MM-JJ)")
    bedrooms: float = Field(..., example=3.0, description="Nombre de chambres")
    bathrooms: float = Field(..., example=2.5, description="Nombre de salles de bain")
    sqft_living: int = Field(..., example=1800, description="Superficie habitable en pieds carrés")
    sqft_lot: int = Field(..., example=6000, description="Superficie du terrain en pieds carrés")
    floors: float = Field(..., example=2.0, description="Nombre d'étages")
    waterfront: int = Field(..., example=0, description="1 si en bord de mer/lac, 0 sinon")
    view: int = Field(..., example=1, description="Qualité de la vue (0-4)")
    condition: int = Field(..., example=3, description="État de la maison (1-5)")
    sqft_above: int = Field(..., example=1500, description="Superficie au-dessus du sous-sol en pieds carrés")
    sqft_basement: int = Field(..., example=300, description="Superficie du sous-sol en pieds carrés")
    yr_built: int = Field(..., example=2005, description="Année de construction")
    yr_renovated: int = Field(..., example=0, description="Année de rénovation (0 si non rénovée)")
    street: str = Field(..., example="Sample St", description="Adresse de la rue")
    city: str = Field(..., example="Seattle", description="Ville")
    statezip: str = Field(..., example="WA 98101", description="État et code postal")
    country: Optional[str] = Field(None, example="USA", description="Pays (optionnel, sera supprimé si présent)")

   
    grade: int = Field(..., example=7, description="Niveau de qualité de la construction et du design (1-13)")
    lat: float = Field(..., example=47.6062, description="Latitude géographique")
    long: float = Field(..., example=-122.3321, description="Longitude géographique")
    zipcode: str = Field(..., example="98101", description="Code postal (peut être un entier ou une chaîne)")
 

@app.get("/")
async def read_root():
    return {"message": "Welcome to the House Price Prediction API!"}

@app.post("/predict/")
async def predict_price(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please contact administrator.")

  
    input_df = pd.DataFrame([features.dict()])

    try:
        prediction = make_prediction(model, input_df)
        
        return {"predicted_price": float(prediction[0])}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal error: {e}")