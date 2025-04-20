import os
import numpy as np
import tensorflow as tf
import datetime

def data_generator(data_folder, batch_size=32, mode="train"):
    """
    A generator that yields batches of data from chunk files.
    """
    # List all chunk files for the specified mode (train or test)
    if mode == "train":
        X_files = sorted([f for f in os.listdir(data_folder) if f.startswith("X_train_chunk_")])
        y_files = sorted([f for f in os.listdir(data_folder) if f.startswith("y_train_chunk_")])
    elif mode == "test":
        X_files = sorted([f for f in os.listdir(data_folder) if f.startswith("X_test_chunk_")])
        y_files = sorted([f for f in os.listdir(data_folder) if f.startswith("y_test_chunk_")])
    else:
        raise ValueError("Invalid mode. Use 'train' or 'test'.")

    # Iterate through each chunk file
    for X_file, y_file in zip(X_files, y_files):
        # Load X and y chunks
        X_chunk = np.load(os.path.join(data_folder, X_file))
        y_chunk = np.load(os.path.join(data_folder, y_file))

        # Yield data in smaller batches
        for i in range(0, len(X_chunk), batch_size):
            yield X_chunk[i:i + batch_size], y_chunk[i:i + batch_size]

def tf_data_generator(data_folder, batch_size=32, mode="train"):
    """
    A TensorFlow-compatible data generator.
    """
    def generator():
        for X_batch, y_batch in data_generator(data_folder, batch_size, mode):
            yield X_batch, y_batch

    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 30, 64, 64, 3), dtype=tf.float32),  # Adjust shape as needed
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )
    return dataset

# Custom callback to log losses
class LossLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            print(f"Epoch {epoch + 1}: Training Loss = {logs['loss']:.4f}, Validation Loss = {logs.get('val_loss', 'N/A'):.4f}")

# Main function
if __name__ == "__main__":
    # Path to the folder containing preprocessed data
    data_folder = "preprocessed_data"

    # Create TensorFlow datasets
    train_dataset = tf_data_generator(data_folder, batch_size=32, mode="train")
    test_dataset = tf_data_generator(data_folder, batch_size=32, mode="test")

    # Build and compile your CNN-LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), input_shape=(30, 64, 64, 3)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mse')













    # Train the model with the custom callback
    model.fit(
        train_dataset,
        epochs=500,
        validation_data=test_dataset,
        callbacks=[LossLogger()]
    )
    model.save("modele500.h5")