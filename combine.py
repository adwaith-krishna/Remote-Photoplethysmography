import os
import numpy as np

def combine_preprocessed_data(data_folder):
    """
    Combines preprocessed data from all batch files into a single dataset using memory-mapped arrays.
    Saves the combined data in smaller chunks.
    """
    # Path to the batches folder
    batches_folder = os.path.join(data_folder, "batches")

    # Check if the batches folder exists
    if not os.path.exists(batches_folder):
        print(f"Error: The folder '{batches_folder}' does not exist.")
        return np.array([]), np.array([])

    # List the contents of the batches folder
    print(f"Contents of '{batches_folder}':")
    print(os.listdir(batches_folder))

    # Calculate the total number of sequences
    total_sequences = 0
    for file in os.listdir(batches_folder):
        if file.startswith("X_batch_") and file.endswith(".npy"):
            X_path = os.path.join(batches_folder, file)
            X_batch = np.load(X_path)
            total_sequences += len(X_batch)

    # Create memory-mapped arrays for combined data
    X_combined = np.memmap(
        os.path.join(data_folder, "X_combined.dat"),  # Save as a memory-mapped file
        dtype='float32',
        mode='w+',  # Read/write mode
        shape=(total_sequences, 30, 64, 64, 3)  # Adjust shape as needed
    )
    y_combined = np.memmap(
        os.path.join(data_folder, "y_combined.dat"),  # Save as a memory-mapped file
        dtype='float32',
        mode='w+',  # Read/write mode
        shape=(total_sequences,)
    )

    # Combine data from all batches
    start_idx = 0
    for file in os.listdir(batches_folder):
        if file.startswith("X_batch_") and file.endswith(".npy"):
            # Load X batch
            X_path = os.path.join(batches_folder, file)
            X_batch = np.load(X_path)
            X_combined[start_idx:start_idx + len(X_batch)] = X_batch

            # Load corresponding y batch
            y_file = file.replace("X_batch_", "y_batch_")
            y_path = os.path.join(batches_folder, y_file)
            if os.path.exists(y_path):
                y_batch = np.load(y_path)
                y_combined[start_idx:start_idx + len(y_batch)] = y_batch
            else:
                print(f"Skipping {file}: Corresponding y batch not found.")

            start_idx += len(X_batch)

    print(f"Combined data shape: X = {X_combined.shape}, y = {y_combined.shape}")

    # Save the combined data in smaller chunks
    chunk_size = 500  # Reduce chunk size further
    num_chunks = (total_sequences + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_sequences)

        # Save X chunk
        X_chunk = X_combined[start:end]
        np.save(os.path.join(data_folder, f"X_combined_chunk_{i}.npy"), X_chunk)

        # Save y chunk
        y_chunk = y_combined[start:end]
        np.save(os.path.join(data_folder, f"y_combined_chunk_{i}.npy"), y_chunk)

    print(f"Combined data saved in {num_chunks} chunks.")

    return X_combined, y_combined

# Main function
if __name__ == "__main__":
    # Path to the folder containing preprocessed data
    data_folder = "preprocessed_data"

    # Combine preprocessed data
    X, y = combine_preprocessed_data(data_folder)