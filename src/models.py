"""
Model builders:
- build_autoencoder(input_dim, latent_dim=16) -> returns (autoencoder, encoder)
- build_tabtransformer(input_dim, latent_dim=16) -> returns model which maps input -> latent_dim vector
"""

from tensorflow import keras
from tensorflow.keras import layers

def build_autoencoder(input_dim, latent_dim=16):
    inp = keras.Input(shape=(input_dim,))

    x = layers.Dense(128, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    x = layers.Dense(32, activation="relu")(latent)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    out = layers.Dense(input_dim, activation="linear")(x)

    autoencoder = keras.Model(inputs=inp, outputs=out, name="autoencoder")
    encoder = keras.Model(inputs=inp, outputs=latent, name="encoder")

    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def build_tabtransformer(input_dim, latent_dim=16, embed_dim=64, num_heads=4):
    inp = keras.Input(shape=(input_dim,))

    # Define sequence length safely
    seq_len = min(max(4, input_dim // 2), 16)

    x = layers.Dense(seq_len * embed_dim, activation="relu")(inp)
    x = layers.Reshape((seq_len, embed_dim))(x)

    key_dim = embed_dim // num_heads

    # Multi-head attention
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim
    )(x, x)

    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.1)(x)

    # Feed-forward block
    ff = layers.Dense(embed_dim * 2, activation="relu")(x)
    ff = layers.Dense(embed_dim, activation="relu")(ff)

    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.1)(x)

    # Pool and project
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(latent_dim, activation="relu")(x)

    model = keras.Model(inputs=inp, outputs=out, name="tabtransformer_like")
    model.compile(optimizer="adam", loss="mse")

    return model
