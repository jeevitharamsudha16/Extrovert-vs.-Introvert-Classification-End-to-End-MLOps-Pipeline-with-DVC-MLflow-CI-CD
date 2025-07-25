import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

def preprocess_data(data, output_dir="artifacts"):
    target_column = "Personality"
    
    numeric_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                       'Friends_circle_size', 'Post_frequency']
    categorical_columns = ['Stage_fear', 'Drained_after_socializing']

    # 🔧 Handle missing values
    data[numeric_columns] = SimpleImputer(strategy='median').fit_transform(data[numeric_columns])
    data[categorical_columns] = SimpleImputer(strategy='most_frequent').fit_transform(data[categorical_columns])

    # 🏷️ Label encode categorical + target columns
    label_encoders = {}
    for col in categorical_columns + [target_column]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # ✂️ Split data
    X = data.drop(columns=[target_column])
    y = data[target_column]
  

    # 📤 Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # ⚖️ Apply SMOTE only on training data
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # 💾 Save each encoder in a dedicated subfolder
    encoder_dir = os.path.join(output_dir, "encoders")
    os.makedirs(encoder_dir, exist_ok=True)
    for col, le in label_encoders.items():
        joblib.dump(le, os.path.join(encoder_dir, f"{col}_encoder.pkl"))
    print(f"✅ All encoders saved individually to {encoder_dir}/")

    return X_train, X_test, y_train, y_test, label_encoders
