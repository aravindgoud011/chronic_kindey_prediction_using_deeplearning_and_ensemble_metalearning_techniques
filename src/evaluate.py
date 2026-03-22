import numpy as np
import joblib
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from .preprocess import PREPROCESSED_PATH


def evaluate():
    data = np.load(PREPROCESSED_PATH)
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Load models
    encoder = keras.models.load_model("artifacts/models/encoder_model.keras")
    tabtransformer = keras.models.load_model("artifacts/models/tabtransformer_model.keras")
    meta_model = joblib.load("artifacts/models/meta_model_lr.pkl")
    meta_scaler = joblib.load("artifacts/scalers/meta_scaler.pkl")

    # Feature extraction
    ae_feats = encoder.predict(X_test)
    tt_feats = tabtransformer.predict(X_test)
    X_meta = np.concatenate([ae_feats, tt_feats], axis=1)

    # Scale meta features
    X_meta_scaled = meta_scaler.transform(X_meta)

    # Predictions
    y_pred = meta_model.predict(X_meta_scaled)
    y_probs = meta_model.predict_proba(X_meta_scaled)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)

    print("\n✅ FINAL EVALUATION RESULTS ✅\n")
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print(f"AUC-ROC   : {auc * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # ==========================
    # 1️⃣ Confusion Matrix Plot
    # ==========================

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-CKD","CKD"],
                yticklabels=["Non-CKD","CKD"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # ==========================
    # 2️⃣ ROC Curve
    # ==========================

    fpr, tpr, _ = roc_curve(y_test, y_probs)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()

    # ==========================
    # 3️⃣ Metrics Bar Chart
    # ==========================

    metrics = ["Accuracy","Precision","Recall","F1 Score","AUC"]
    values = [acc,prec,rec,f1,auc]

    plt.figure(figsize=(7,5))
    sns.barplot(x=metrics, y=values)
    plt.title("Model Evaluation Metrics")
    plt.ylim(0,1)
    plt.savefig("evaluation_metrics.png")
    plt.close()


if __name__ == "__main__":
    evaluate()