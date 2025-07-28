import os
import joblib
import pandas as pd
import mlflow
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import datetime

def train_models(X_train, y_train, model_dir="artifacts/models"):
    os.environ["MLFLOW_TRACKING_USERNAME"] = "jeevitharamsudha16"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

    mlflow.set_tracking_uri(
        "https://dagshub.com/jeevitharamsudha16/Extrovert-vs.-Introvert-Classification-End-to-End-MLOps-Pipeline-with-DVC-MLflow-CI-CD.mlflow"
    )
    mlflow.set_experiment("Personality_Classification")
    print("✅ Connected to MLflow (DagsHub)")

    os.makedirs(model_dir, exist_ok=True)

    models = {
        'dt': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5, 10]}
        },
        'rf': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
        },
        'gb': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {'n_estimators': [50, 100, 150], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
        },
        'xgb': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5, 7]}
        }
    }

    summary_results = []

    for name, config in models.items():
        print(f"\n🚀 Training {name.upper()}...")
        clf = RandomizedSearchCV(
            estimator=config['model'],
            param_distributions=config['params'],
            n_iter=10,
            cv=4,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_

        model_path = os.path.join(model_dir, f"{name}_model.pkl")
        joblib.dump(best_model, model_path)

        run_name = f"{name}_train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({"developer": "jeevitharamsudha16", "stage": "training", "model_type": name.upper()})
            mlflow.log_params(clf.best_params_)
            mlflow.log_metric("cv_accuracy", clf.best_score_)
            mlflow.log_artifact(model_path)  # ✅ Log manually instead of log_model()

        print(f"✅ Saved {name.upper()} model to: {model_path}")
        summary_results.append({
            "Model": name.upper(),
            "Best Accuracy": round(clf.best_score_, 4),
            "Best Params": clf.best_params_
        })

    summary_df = pd.DataFrame(summary_results)
    summary_csv = os.path.join(model_dir, "model_comparison.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n📊 Model training summary saved at: {summary_csv}")
