import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Disable mixed precision (for debugging, re-enable later if needed)
policy = mixed_precision.Policy('float32')  # Change 'mixed_float16' to 'float32'
mixed_precision.set_global_policy(policy)

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def check_data_for_nans(data_folder, mode="train"):
    """Checks for NaN values in dataset chunks."""
    X_files = sorted([f for f in os.listdir(data_folder) if f.startswith(f"X_{mode}_chunk_")])
    y_files = sorted([f for f in os.listdir(data_folder) if f.startswith(f"y_{mode}_chunk_")])

    for X_file, y_file in zip(X_files, y_files):
        X_chunk = np.load(os.path.join(data_folder, X_file))
        y_chunk = np.load(os.path.join(data_folder, y_file))

        if np.isnan(X_chunk).any():
            print(f"NaNs detected in {X_file}")
        if np.isnan(y_chunk).any():
            print(f"NaNs detected in {y_file}")


def data_generator(data_folder, batch_size=8, mode="train"):
    """Yields batches of data from chunk files."""
    if mode == "train":
        X_files = sorted([f for f in os.listdir(data_folder) if f.startswith("X_train_chunk_")])
        y_files = sorted([f for f in os.listdir(data_folder) if f.startswith("y_train_chunk_")])
    elif mode == "test":
        X_files = sorted([f for f in os.listdir(data_folder) if f.startswith("X_test_chunk_")])
        y_files = sorted([f for f in os.listdir(data_folder) if f.startswith("y_test_chunk_")])
    else:
        raise ValueError("Invalid mode. Use 'train' or 'test'.")

    for X_file, y_file in zip(X_files, y_files):
        X_chunk = np.load(os.path.join(data_folder, X_file)).astype(np.float32)
        y_chunk = np.load(os.path.join(data_folder, y_file)).astype(np.float32)
        X_chunk = np.nan_to_num(X_chunk, nan=0.0)
        y_chunk = np.nan_to_num(y_chunk, nan=0.0)

        for i in range(0, len(X_chunk), batch_size):
            yield X_chunk[i:i + batch_size], y_chunk[i:i + batch_size]


def tf_data_generator(data_folder, batch_size=8, mode="train"):
    """TensorFlow-compatible data generator with prefetching."""

    def gen():
        for X_batch, y_batch in data_generator(data_folder, batch_size, mode):
            yield X_batch, y_batch

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 30, 64, 64, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)


class LossLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            print(f"Epoch {epoch + 1}: Loss = {logs['loss']:.4f}, Val Loss = {logs.get('val_loss', 'N/A'):.4f}")


if __name__ == "__main__":
    data_folder = "preprocessed_data"
    check_data_for_nans(data_folder, mode="train")
    check_data_for_nans(data_folder, mode="test")

    train_dataset = tf_data_generator(data_folder, batch_size=8, mode="train")
    test_dataset = tf_data_generator(data_folder, batch_size=8, mode="test")

    model = tf.keras.Sequential([
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            input_shape=(30, 64, 64, 3)
        ),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(2, 2)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.25)),

        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        ),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(2, 2)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.3)),

        tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D()),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, dtype='float32')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)  # Reduced LR + Clipping
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=test_dataset,
        callbacks=[
            LossLogger(),
            tf.keras.callbacks.EarlyStopping(patience=5),
            tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)
        ]
    )
    model.save("final_model.h5")
