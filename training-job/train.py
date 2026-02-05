import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Connexion au serveur MLflow
mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment("Iris-Experiment")

with mlflow.start_run():
    # Data
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modèle
    n_estimators = 100
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Logs MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    registered_model_name="iris_model"
)

    print("Training terminé. Accuracy =", accuracy)
