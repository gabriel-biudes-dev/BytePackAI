import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib


def extract_histogram(file_path):
    """
    Extracts the byte histogram from an executable file.

    Args:
        file_path (str): Path to the file.

    Returns:
        numpy.ndarray: Byte histogram (256 bins) as integers, or None if an error occurs.
    """
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        histogram = np.histogram(list(data), bins=256, range=(0, 255))[0]
        return histogram.astype(int)  # Ensure consistent data type
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def collect_data(directory, label):
    """
    Collects histogram data and corresponding labels from files in a directory.

    Args:
        directory (str): Path to the directory containing files.
        label (int): Label to assign to all files in this directory (e.g., 1 for packed).

    Returns:
        tuple: A list of histograms and a list of labels.
    """
    data = []
    labels = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            histogram = extract_histogram(file_path)
            if histogram is not None:
                data.append(histogram)
                labels.append(label)
    return data, labels


def train_model(packed_dir, not_packed_dir):
    """
    Trains a Random Forest model to classify executables as packed or not packed.

    Args:
        packed_dir (str): Path to the directory containing packed files.
        not_packed_dir (str): Path to the directory containing not packed files.
    """
    print("Collecting data for packed files...")
    packed_data, packed_labels = collect_data(packed_dir, 1)

    print("Collecting data for not packed files...")
    not_packed_data, not_packed_labels = collect_data(not_packed_dir, 0)

    # Combine data and labels
    X = np.array(packed_data + not_packed_data)
    y = np.array(packed_labels + not_packed_labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model
    print("Training the model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    # Save the model
    joblib.dump(model, "packed_detector.pkl")
    print("Model saved as 'packed_detector.pkl'.")


def classify_executable(file_path):
    """
    Classifies an executable file as packed or not packed.

    Args:
        file_path (str): Path to the file to classify.
    """
    if not os.path.exists("packed_detector.pkl"):
        print("Model not found. Train the model first.")
        return

    # Load the model
    model = joblib.load("packed_detector.pkl")

    # Extract features from the file
    histogram = extract_histogram(file_path)
    if histogram is None:
        print(f"Could not process file: {file_path}")
        return

    # Predict the label
    prediction = model.predict([histogram])
    result = "PACKED" if prediction[0] == 1 else "NOT PACKED"
    print(f"The file '{file_path}' is classified as: {result}")


if __name__ == "__main__":
    """
    Entry point for the script. Supports training and classification modes.

    Usage:
        Train the model:
            python bytepackai.py train <packed_dir> <unpacked_dir>
        Classify an executable:
            python bytepackai.py classify <file_path>
    """
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Train: python bytepackai.py train <packed_dir> <unpacked_dir>")
        print("  Classify: python bytepackai.py classify <file_path>")
    else:
        command = sys.argv[1].lower()
        if command == "train" and len(sys.argv) == 4:
            packed_dir = sys.argv[2]
            not_packed_dir = sys.argv[3]
            train_model(packed_dir, not_packed_dir)
        elif command == "classify" and len(sys.argv) == 3:
            file_path = sys.argv[2]
            classify_executable(file_path)
        else:
            print("Invalid arguments. See usage.")
