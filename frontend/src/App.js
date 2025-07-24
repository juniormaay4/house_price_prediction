import React, { useState } from 'react';
import './App.css';

function App() {
  // Définir l'état pour chaque champ du formulaire
  const [date, setDate] = useState('2025-01-15');
  const [street, setStreet] = useState('123 Main St');
  const [city, setCity] = useState('Seattle');
  const [statezip, setStatezip] = useState('WA 98101');
  const [sqft_living, setSqftLiving] = useState('');
  const [bedrooms, setBedrooms] = useState('');
  const [bathrooms, setBathrooms] = useState('');
  const [grade, setGrade] = useState('');
  const [lat, setLat] = useState('');
  const [long, setLong] = useState('');
  const [yr_built, setYrBuilt] = useState('');
  const [waterfront, setWaterfront] = useState(0);
  const [view, setView] = useState(0);
  const [condition, setCondition] = useState(1);
  const [floors, setFloors] = useState('');
  const [zipcode, setZipcode] = useState('');
  const [yr_renovated, setYrRenovated] = useState('');
  const [sqft_lot, setSqftLot] = useState('');
  const [sqft_above, setSqftAbove] = useState('');
  const [sqft_basement, setSqftBasement] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    const formData = {
      date: date,
      street: street,
      city: city,
      statezip: statezip,
      sqft_living: parseInt(sqft_living),
      bedrooms: parseFloat(bedrooms),
      bathrooms: parseFloat(bathrooms),
      grade: parseInt(grade),
      lat: parseFloat(lat),
      long: parseFloat(long),
      yr_built: parseInt(yr_built),
      waterfront: parseInt(waterfront),
      view: parseInt(view),
      condition: parseInt(condition),
      floors: parseFloat(floors),
      zipcode: zipcode,
      yr_renovated: parseInt(yr_renovated),
      sqft_lot: parseInt(sqft_lot),
      sqft_above: parseInt(sqft_above),
      sqft_basement: parseInt(sqft_basement),
    };

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Erreur lors de la prédiction');
      }

      const data = await response.json();
      setPrediction(data.predicted_price);
    } catch (err) {
      console.error("Erreur de prédiction :", err);
      setError(err.message || 'Une erreur inattendue est survenue.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="project-header">
        <h1>Projet MLOps - Groupe ISI</h1>
        <h2>Dublond Junior, Mayembo Miyeke</h2>
      </div>
      
      <div className="main-container">
        <header className="App-header">
          <div className="title-container">
            <h1>Prédiction du Prix Immobilier 🏡</h1>
            <p className="subtitle">Estimez la valeur d'une propriété en fonction de ses caractéristiques</p>
          </div>
          
          <form onSubmit={handleSubmit} className="prediction-form">
            <div className="form-section">
              <h3>Informations de localisation</h3>
              <div className="form-group">
                <label>Date (AAAA-MM-JJ):</label>
                <input type="text" value={date} onChange={(e) => setDate(e.target.value)} required />
              </div>
              <div className="form-group">
                <label>Rue:</label>
                <input type="text" value={street} onChange={(e) => setStreet(e.target.value)} required />
              </div>
              <div className="form-group">
                <label>Ville:</label>
                <input type="text" value={city} onChange={(e) => setCity(e.target.value)} required />
              </div>
              <div className="form-group">
                <label>État et Code Postal:</label>
                <input type="text" value={statezip} onChange={(e) => setStatezip(e.target.value)} required />
              </div>
              <div className="form-group">
                <label>Code Postal:</label>
                <input type="text" value={zipcode} onChange={(e) => setZipcode(e.target.value)} required />
              </div>
              <div className="form-group">
                <label>Latitude:</label>
                <input type="number" step="any" value={lat} onChange={(e) => setLat(e.target.value)} required />
              </div>
              <div className="form-group">
                <label>Longitude:</label>
                <input type="number" step="any" value={long} onChange={(e) => setLong(e.target.value)} required />
              </div>
            </div>

            <div className="form-section">
              <h3>Caractéristiques principales</h3>
              <div className="form-group">
                <label>Surface habitable (pieds carrés):</label>
                <input type="number" value={sqft_living} onChange={(e) => setSqftLiving(e.target.value)} required />
              </div>
              <div className="form-group">
                <label>Chambres:</label>
                <input type="number" value={bedrooms} onChange={(e) => setBedrooms(e.target.value)} required />
              </div>
              <div className="form-group">
                <label>Salles de bain:</label>
                <input type="number" step="0.5" value={bathrooms} onChange={(e) => setBathrooms(e.target.value)} required />
              </div>
              <div className="form-group">
                <label>Niveau de qualité (1-13):</label>
                <input type="number" value={grade} onChange={(e) => setGrade(e.target.value)} min="1" max="13" required />
              </div>
              <div className="form-group">
                <label>Étages:</label>
                <input type="number" step="0.5" value={floors} onChange={(e) => setFloors(e.target.value)} required />
              </div>
            </div>

            <div className="form-section">
              <h3>Détails supplémentaires</h3>
              <div className="form-group">
                <label>Année de construction:</label>
                <input type="number" value={yr_built} onChange={(e) => setYrBuilt(e.target.value)} required />
              </div>
              <div className="form-group">
                <label>Bord de l'eau:</label>
                <select value={waterfront} onChange={(e) => setWaterfront(e.target.value)}>
                  <option value="0">Non</option>
                  <option value="1">Oui</option>
                </select>
              </div>
              <div className="form-group">
                <label>Qualité de la vue (0-4):</label>
                <input type="number" value={view} onChange={(e) => setView(e.target.value)} min="0" max="4" />
              </div>
              <div className="form-group">
                <label>État général (1-5):</label>
                <input type="number" value={condition} onChange={(e) => setCondition(e.target.value)} min="1" max="5" />
              </div>
              <div className="form-group">
                <label>Année de rénovation (0 si jamais):</label>
                <input type="number" value={yr_renovated} onChange={(e) => setYrRenovated(e.target.value)} />
              </div>
            </div>

            <div className="form-section">
              <h3>Surfaces</h3>
              <div className="form-group">
                <label>Taille du terrain (pieds carrés):</label>
                <input type="number" value={sqft_lot} onChange={(e) => setSqftLot(e.target.value)} required />
              </div>
              <div className="form-group">
                <label>Surface au-dessus du sol:</label>
                <input type="number" value={sqft_above} onChange={(e) => setSqftAbove(e.target.value)} required />
              </div>
              <div className="form-group">
                <label>Surface du sous-sol:</label>
                <input type="number" value={sqft_basement} onChange={(e) => setSqftBasement(e.target.value)} required />
              </div>
            </div>

            <div className="form-actions">
              <button type="submit" disabled={loading}>
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Calcul en cours...
                  </>
                ) : 'Estimer le prix'}
              </button>
            </div>
          </form>

          {error && (
            <div className="alert error">
              <span className="alert-icon">⚠️</span>
              <p>{error}</p>
            </div>
          )}

          {prediction !== null && (
            <div className="prediction-result">
              <div className="result-card">
                <h3>Estimation de prix</h3>
                <div className="price">
                  {new Intl.NumberFormat('fr-FR', { style: 'currency', currency: 'USD' }).format(prediction)}
                </div>
                <p className="disclaimer">* Cette estimation est basée sur notre modèle MLOps et peut varier selon les conditions du marché.</p>
              </div>
            </div>
          )}
        </header>
      </div>
    </div>
  );
}

export default App;