import os
import joblib
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dagshub import init
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, RocCurveDisplay
)
import datetime

def evaluate_models(X_test, y_test, model_dir="artifacts/models", log_to_mlflow=True):
    # ✅ Connect to DagsHub MLflow
    init(
        repo_owner="jeevitharamsudha16",
        repo_name="Extrovert-vs.-Introvert-Classification-End-to-End-MLOps-Pipeline-with-DVC-MLflow-CI-CD",
        mlflow=True
    )
    mlflow.set_experiment("Personality_Classification")
    print("✅ Connected to MLflow on DagsHub")

    # ✅ Filter model files excluding scalers, best_model, or comparison
    model_files = [
        f for f in os.listdir(model_dir)
        if f.endswith(".pkl") and all(x not in f for x in ["scaler", "best_model", "comparison"])
    ]
    print(f"🔍 Found {len(model_files)} models to evaluate in: {model_dir}")

    if not model_files:
        print("❌ No valid models found to evaluate.")
        return

    best_f1 = 0
    best_model = None
    best_model_name = ""
    prediction_log = {}
    eval_results = []

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        model_name = model_file.replace(".pkl", "")

        try:
            loaded_obj = joblib.load(model_path)
            model = loaded_obj["model"] if isinstance(loaded_obj, dict) and "model" in loaded_obj else loaded_obj
            y_pred = model.predict(X_test)
        except Exception as e:
            print(f"❌ Skipping {model_name}: Failed to predict. Error: {e}")
            continue

        # 📊 Evaluation Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\n📌 {model_name.upper()} Evaluation:")
        print(f"✅ Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

        prediction_log[model_name] = y_pred
        eval_results.append({
            "model": model_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        # 🏆 Track best model
        is_best = False
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = model_name
            is_best = True

        if log_to_mlflow:
            run_name = f"{model_name}_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags({
                    "stage": "evaluation",
                    "model_alias": f"{model_name}_Classifier"
                })

                mlflow.log_params({
                    "model_name": model_name,
                    "model_file": model_file
                })

                mlflow.log_metrics({
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1
                })

                mlflow.sklearn.log_model(model, artifact_path="model")

                if is_best:
                    mlflow.set_tag("best_model", "true")

                # 📊 Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title(f"Confusion Matrix - {model_name}")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                cm_path = os.path.join(model_dir, f"{model_name}_conf_matrix.png")
                plt.savefig(cm_path)
                plt.close()
                mlflow.log_artifact(cm_path)

                # 📈 ROC Curve
                if hasattr(model, "predict_proba"):
                    try:
                        y_probs = model.predict_proba(X_test)[:, 1]
                        RocCurveDisplay.from_predictions(y_test, y_probs)
                        roc_path = os.path.join(model_dir, f"{model_name}_roc_curve.png")
                        plt.savefig(roc_path)
                        plt.close()
                        mlflow.log_artifact(roc_path)
                    except Exception as e:
                        print(f"⚠️ Could not generate ROC for {model_name}: {e}")

    # ✅ Save best model
    if best_model:
        best_path = os.path.join(model_dir, "best_model.pkl")
        joblib.dump(best_model, best_path)
        print(f"\n🏆 Best model '{best_model_name}' saved at: {best_path}")

    # 📄 Save predictions
    if prediction_log:
        pred_df = pd.DataFrame(prediction_log)
        comparison_path = os.path.join(model_dir, "prediction_comparison.csv")
        pred_df.to_csv(comparison_path, index=False)
        print(f"📄 Saved prediction comparison at: {comparison_path}")

    # 📈 Save evaluation summary
    if eval_results:
        summary_df = pd.DataFrame(eval_results)
        summary_path = os.path.join(model_dir, "evaluation_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"📊 Saved evaluation summary at: {summary_path}")
