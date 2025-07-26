import os
import time
import mlflow

from data_loader import load_data 
from data_preprocessing import preprocess_data
from model_training import train_models
from model_evaluation import evaluate_models

if __name__ == "__main__":
    start_time = time.time()

    # ✅ Set remote MLflow tracking URI for DagsHub
    mlflow.set_tracking_uri("https://dagshub.com/jeevitharamsudha16/Extrovert-vs.-Introvert-Classification-End-to-End-MLOps-Pipeline-with-DVC-MLflow-CI-CD.mlflow")

    # ✅ Use your experiment name (must match or pre-create in DagsHub UI)
    mlflow.set_experiment("Personality_Classification")

    # ✅ Authenticate using your DagsHub token as MLflow username
    #os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_TOKEN")
    #os.environ["MLFLOW_TRACKING_PASSWORD"] = ""

    # Debug: confirm tracking URI
    print(f"📡 MLflow Tracking URI (from script): {mlflow.get_tracking_uri()}")

    try:
        # 🚀 Step 1: Load Raw Data
        df = load_data("data/personality_dataset.csv")
        print("\n📊 First 5 rows of raw dataset:")
        print(df.head())

        # 🧹 Step 2: Preprocess Data
        X_train, X_test, y_train, y_test, label_encoders = preprocess_data(df)

        # 🤖 Step 3: Train Models
        train_models(X_train, y_train)

        # 🧪 Step 4: Evaluate Models and log to remote MLflow
        evaluate_models(X_test, y_test, model_dir="artifacts/models", log_to_mlflow=True)

        print("\n✅ Pipeline completed successfully!")

    except Exception as e:
        print(f"\n❌ Pipeline failed due to: {e}")

    end_time = time.time()
    print(f"\n⏱️ Total time taken: {end_time - start_time:.2f} seconds")
