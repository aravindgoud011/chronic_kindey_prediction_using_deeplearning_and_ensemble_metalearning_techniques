import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from .utils import ensure_dirs, save_joblib

DATA_PATH = "data/ckd_10000_raw.xlsx"

SCALER_PATH = "artifacts/scalers/scaler.pkl"
ENCODER_PATH = "artifacts/scalers/ordinal_encoder.pkl"
META_INFO_PATH = "artifacts/scalers/feature_info.pkl"
PREPROCESSED_PATH = "artifacts/preprocessed_data.npz"

ensure_dirs()

def _identify_target(df):
    for col in ["classification", "class", "Class", "target"]:
        if col in df.columns:
            return col
    return df.columns[-1]

def preprocess_save():
    # ---------- LOAD ----------
    df = pd.read_excel(DATA_PATH)
    df = df.replace("?", np.nan)
    df = df.drop_duplicates().dropna(how="all")

    target_col = _identify_target(df)

    # ---------- LABEL CLEANING ----------
    y_raw = df[target_col].astype(str).str.lower().str.strip()

    def map_label(v):
        if v == "ckd":
            return 1
        elif v in ["notckd", "no_ckd", "nockd"]:
            return 0
        else:
            return 0

    df["__target__"] = y_raw.apply(map_label)

    # ---------- SOFT MEDICAL CLEANING ----------
    df = df[~(
        (df["sc"] <= 1.1) &
        (df["bu"] <= 35) &
        (df["al"] == 0) &
        (df["hemo"] >= 12) &
        (df["__target__"] == 1)
    )].reset_index(drop=True)

    y = df["__target__"].values
    X = df.drop(columns=[target_col, "__target__"])

    # ---------- FEATURE TYPES ----------
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # ---------- MISSING VALUES ----------
    for c in numeric_cols:
        X[c] = X[c].fillna(X[c].median())

    for c in cat_cols:
        X[c] = X[c].fillna(X[c].mode()[0])

    # ---------- TRY NUMERIC CONVERSION ----------
    for c in cat_cols[:]:
        try:
            X[c] = pd.to_numeric(X[c])
            numeric_cols.append(c)
            cat_cols.remove(c)
        except:
            pass

    # ---------- TRAIN / TEST SPLIT ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---------- ENCODING ----------
    if cat_cols:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols].astype(str))
        X_test[cat_cols] = encoder.transform(X_test[cat_cols].astype(str))
        save_joblib(encoder, ENCODER_PATH)

    # ---------- SCALING ----------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    save_joblib(scaler, SCALER_PATH)

    # ---------- REMOVE EXTREME OUTLIERS ----------
    mask = np.abs(X_train_scaled).mean(axis=1) < 3.0
    X_train_scaled = X_train_scaled[mask]
    y_train = y_train[mask]

    # ---------- SAFE SMOTE ----------
    counter = Counter(y_train)
    print("Class distribution before SMOTE:", counter)

    if counter[0] != counter[1]:
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_final, y_train_final = smote.fit_resample(
            X_train_scaled, y_train
        )
        print("SMOTE applied")
    else:
        X_train_final, y_train_final = X_train_scaled, y_train
        print("SMOTE skipped (already balanced)")

    # ---------- SAVE ----------
    np.savez(
        PREPROCESSED_PATH,
        X_train=X_train_final,
        X_test=X_test_scaled,
        y_train=y_train_final,
        y_test=y_test
    )

    save_joblib({
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "all_cols": X.columns.tolist()
    }, META_INFO_PATH)

    print("✅ Preprocessing completed successfully")
    print("Train samples:", len(X_train_final))
    print("Test samples :", len(X_test_scaled))

if __name__ == "__main__":
    preprocess_save()
