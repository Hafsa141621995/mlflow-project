# mlflow-project

## Version Complète et Académique
### Objectif global de la formation

La formation avait pour objectif de comprendre comment :

Développer un modèle de Machine Learning

L’industrialiser

Le containeriser avec Docker

Le déployer sur différents cloud providers

Mettre en place une logique MLOps

On ne faisait pas juste du code.
On apprenait à rendre un projet IA déployable dans un environnement réel.

1. Azure Machine Learning – Orchestration

On a appris à :

Créer un environnement custom (conda + pip)

Définir des composants (preprocess, train, evaluate)

Utiliser command()

Construire un pipeline avec @pipeline

Exécuter sur un compute cluster

Gérer les datasets versionnés

Donc on a appris l’orchestration d’un workflow ML dans le cloud.

Ce n’était plus un script local.
C’était un pipeline industrialisé.

2. Séparation des responsabilités

On a appris à séparer :

Prétraitement

Entraînement

Évaluation

Serving

Tracking

Chaque étape est indépendante.
C’est une architecture modulaire.

3. MLflow – Tracking & Monitoring

On a appris :

Lancer un MLflow server

Configurer un backend store

Comprendre le suivi des expériences

Gérer les artefacts

Donc on a compris la notion de MLOps et traçabilité des modèles.

4. API ML – Model Serving

On a appris à :

Charger un modèle entraîné

Exposer une API FastAPI

Créer un endpoint /predict

Structurer les données avec Pydantic

Donc on a compris comment transformer un modèle en service exploitable.

5. Docker – Containerisation

On a appris :

Écrire un Dockerfile propre

Optimiser les layers

Gérer les ports dynamiques

Adapter selon le provider (Azure, GCP, Heroku, AWS)

Donc on a appris la portabilité.

6. Multi-Cloud

On a appris les différences entre :

Provider	Logique
Azure	Orchestration ML
GCP	Cloud Run serverless
AWS	VM / ECS
Heroku	PaaS dyno

On a compris que le Dockerfile doit s’adapter au provider.

7. Architecture globale

À la fin, on obtient :

Pipeline Azure ML (training)

MLflow (tracking)

API ML (serving)

Streamlit (frontend)

LLM service (Ollama)

Déploiement multi-cloud

Donc on a construit une architecture IA complète.