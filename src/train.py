import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from .models import build_autoencoder, build_tabtransformer
from .preprocess import PREPROCESSED_PATH
from .utils import ensure_dirs, save_joblib

ensure_dirs()

LATENT_DIM = 16

def train():
    # ---------------- Load data ----------------
    arr = np.load(PREPROCESSED_PATH)
    X_train = arr["X_train"]
    X_test = arr["X_test"]
    y_train = arr["y_train"]
    y_test = arr["y_test"]

    input_dim = X_train.shape[1]
    print("Input dim:", input_dim)

    # ---------------- Autoencoder ----------------
    autoencoder, encoder = build_autoencoder(
        input_dim=input_dim,
        latent_dim=LATENT_DIM
    )

    autoencoder.fit(
        X_train, X_train,
        validation_data=(X_test, X_test),
        epochs=50,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=7,
                restore_best_weights=True
            )
        ],
        verbose=1
    )

    # ---------------- TabTransformer ----------------
    tabtransformer = build_tabtransformer(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        embed_dim=64,
        num_heads=4
    )

    inp = tabtransformer.input
    feat = tabtransformer.output
    recon = keras.layers.Dense(
        input_dim,
        activation="linear",
        name="recon_head"
    )(feat)

    train_model = keras.Model(inputs=inp, outputs=recon)
    train_model.compile(optimizer="adam", loss="mse")

    train_model.fit(
        X_train, X_train,
        validation_data=(X_test, X_test),
        epochs=40,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=6,
                restore_best_weights=True
            )
        ],
        verbose=1
    )

    # ---------------- Feature Extraction ----------------
    ae_feats = encoder.predict(X_train)
    tt_feats = tabtransformer.predict(X_train)

    print("AE features shape:", ae_feats.shape)
    print("TT features shape:", tt_feats.shape)

    X_meta = np.concatenate([ae_feats, tt_feats], axis=1)

    # ---------------- META FEATURE SCALING ----------------
    meta_scaler = StandardScaler()
    X_meta_scaled = meta_scaler.fit_transform(X_meta)
    save_joblib(meta_scaler, "artifacts/scalers/meta_scaler.pkl")

    # ---------------- Logistic Meta Learner ----------------
    meta_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        C=1.0,
        random_state=42
    )

    meta_model.fit(X_meta_scaled, y_train)

    # ---------------- Save Models ----------------
    save_joblib(meta_model, "artifacts/models/meta_model_lr.pkl")
    encoder.save("artifacts/models/encoder_model.keras")
    tabtransformer.save("artifacts/models/tabtransformer_model.keras")

    print("✅ Training completed successfully")

if __name__ == "__main__":
    train()
