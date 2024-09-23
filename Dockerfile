# Utiliser l'image de base python 3.10
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système si nécessaire
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cloner le projet depuis le dépôt GitHub
RUN apt-get update && apt-get install -y git

# Cloner le dépôt GitHub
RUN git clone https://github.com/QuentinVillotta/audit_anomaly_detection_ETL.git .

# Installer les dépendances Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exposer le port utilisé par Streamlit
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Définir la commande par défaut pour lancer l'application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]