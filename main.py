import os
import time
import mlflow
from data_loader import load_data
from data_preprocessing import preprocess_data
from model_training import train_models
from model_evaluation import evaluate_models

if __name__ == "__main__":
    start_time = time.time()

    # ✅ Set MLflow credentials for DagsHub
    os.environ["MLFLOW_TRACKING_USERNAME"] = "jeevitharamsudha16"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

    # ✅ Set MLflow tracking URI (replace with your DagsHub MLflow URL)
    mlflow.set_tracking_uri(
        "https://dagshub.com/jeevitharamsudha16/Extrovert-vs.-Introvert-Classification-End-to-End-MLOps-Pipeline-with-DVC-MLflow-CI-CD.mlflow"
    )

    print(f"📡 MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    try:
        # Step 1: Load data
        df = load_data("data/personality_dataset.csv")
        print("\n📊 First 5 rows of raw dataset:")
        print(df.head())

        # Step 2: Preprocess data
        X_train, X_test, y_train, y_test, label_encoders = preprocess_data(df)

        # Step 3: Train models and log to MLflow
        train_models(X_train, y_train)

        # Step 4: Evaluate models and log to MLflow
        evaluate_models(X_test, y_test, model_dir="artifacts/models", log_to_mlflow=True)

        print("\n✅ Pipeline completed successfully!")

    except Exception as e:
        print(f"\n❌ Pipeline failed due to: {e}")

    print(f"\n⏱️ Total time taken: {time.time() - start_time:.2f} seconds")
