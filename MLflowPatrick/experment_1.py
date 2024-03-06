import mlflow

def run_experiment(experiment_name):
    # Démarrez une expérience MLflow
    with mlflow.start_run(run_name=experiment_name):
        # Enregistrez des métriques
        mlflow.log_metric("accuracy", 0.85)
        mlflow.log_metric("precision", 0.90)
        
        # Enregistrez des paramètres
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("batch_size", 32)
        
        # Enregistrez un artefact (par exemple, un fichier modèle)
        with open("model.pkl", "wb") as f:
            # Écrivez du contenu fictif dans le fichier modèle
            f.write(b"model_content")
        mlflow.log_artifact("model.pkl")

def main():
    # Créez deux expériences avec du contenu
    run_experiment("experment 1")
    run_experiment("experment 2")

if __name__ == "__main__":
    main()
