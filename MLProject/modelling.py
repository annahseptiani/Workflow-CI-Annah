import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# Ambil token dan konfigurasi langsung dari environment GitHub Actions
dagshub_token = os.getenv("DAGSHUB_TOKEN_ENV")
repo_owner = "annahseptiani14"
repo_name = "membangun_model"

if dagshub_token:
    # Mengonfigurasi MLflow Tracking secara langsung tanpa menggunakan dagshub.init()
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    print("✓ Berhasil mengonfigurasi MLflow Tracking URI ke DagsHub secara remote.")
else:
    print("⚠️ Peringatan: DAGSHUB_TOKEN_ENV tidak ditemukan.")

# Set nama eksperimen yang dituju
mlflow.set_experiment("Heart_Disease_CI_Automation")

print("=== Mengambil dataset hasil preprocessing ===")
df = pd.read_csv('heart_preprocessed.csv')

X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("=== Menjalankan Re-Training Model via MLflow Project ===")
with mlflow.start_run(run_name="CI_Automated_Run") as run:
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # Mencatat Parameter dan Metrik Utama
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    
    # Daftarkan Model Utama Scikit-Learn
    mlflow.sklearn.log_model(model, "model")
    print(f"Sukses! Model dilatih dengan Akurasi Test: {test_acc:.4f}")