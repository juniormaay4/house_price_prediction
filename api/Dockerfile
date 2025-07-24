# api/Dockerfile

# Utilise une image de base Python légère (par exemple, Python 3.9 Alpine pour une petite taille)
FROM python:3.9-slim-buster

# Définit le répertoire de travail à l'intérieur du conteneur
# Tous les chemins relatifs dans le Dockerfile seront par rapport à WORKDIR
WORKDIR /app

# Copie le fichier requirements.txt de l'API dans le conteneur
# Ceci doit être fait avant d'installer les dépendances pour profiter du cache Docker
COPY api/requirements.txt ./requirements.txt

# Installe les dépendances Python spécifiées dans requirements.txt
# --no-cache-dir : n'enregistre pas les fichiers intermédiaires du cache pip
# -U : met à jour les paquets déjà installés
RUN pip install --no-cache-dir -U -r requirements.txt

# Copie tout le reste du code de l'API dans le conteneur
# Le dossier 'api' du contexte de build sera copié dans /app
COPY api/ .

# Copie le dossier 'src' entier (contenant data_preparation, features_engineering, model, predict)
# C'est crucial car votre API dépend de ces modules
COPY src/ ../src/

# Copie le dossier 'models' entier (contenant le modèle entraîné)
# C'est nécessaire pour que l'API puisse charger le modèle
COPY models/ ../models/

# Copie le dossier 'config.py' depuis la racine du projet (où se trouve src)
# config.py est essentiel car il contient tous les chemins et configurations partagés
COPY config.py ../config.py

# Expose le port sur lequel l'API FastAPI s'exécute par défaut
# Ceci informe Docker que le conteneur écoutera sur ce port
EXPOSE 8000

# Commande à exécuter lorsque le conteneur démarre
# 'uvicorn' est le serveur ASGI pour FastAPI
# 'main:app' fait référence à l'instance 'app' dans 'main.py'
# --host 0.0.0.0 rend l'API accessible depuis l'extérieur du conteneur
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]