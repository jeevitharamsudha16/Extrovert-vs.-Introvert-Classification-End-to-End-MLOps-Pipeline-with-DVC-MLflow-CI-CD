import os
import time
import mlflow
from dagshub import init
from data_loader import load_data
from data_preprocessing import preprocess_data 
from model_training import train_models
from model_evaluation import evaluate_models

if __name__ == "__main__":
    start_time = time.time()

    # ✅ Set MLflow credentials for DagsHub
    os.environ["MLFLOW_TRACKING_USERNAME"] = "jeevitharamsudha16"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

    # ✅ Initialize DagsHub + MLflow tracking
    init(
        repo_owner="jeevitharamsudha16",
        repo_name="Extrovert-vs.-Introvert-Classification-End-to-End-MLOps-Pipeline-with-DVC-MLflow-CI-CD",
        mlflow=True,
        token=os.getenv("DAGSHUB_TOKEN")
    )

    print(f"📡 MLflow Tracking URI (dagshub): {mlflow.get_tracking_uri()}")

    try:
        df = load_data("data/personality_dataset.csv")
        print("\n📊 First 5 rows of raw dataset:")
        print(df.head())

        X_train, X_test, y_train, y_test, label_encoders = preprocess_data(df)

        train_models(X_train, y_train)
        evaluate_models(X_test, y_test, model_dir="artifacts/models", log_to_mlflow=True)

        print("\n✅ Pipeline completed successfully!")

    except Exception as e:
        print(f"\n❌ Pipeline failed due to: {e}")

    print(f"\n⏱️ Total time taken: {time.time() - start_time:.2f} seconds")

