# BytePackAI: Executable File Packing Detection with Machine Learning

BytePackAI is a machine learning-based tool designed to classify executable files as packed or not packed by analyzing byte histograms. It uses a Random Forest Classifier to detect packing, an obfuscation technique often used in malware to evade detection. The project offers a simple command-line interface for both training and classifying files.

## Key Features:
- **Byte Histogram Extraction**: Converts executable files into byte histograms (256 bins) to represent file contents.
- **Machine Learning Model**: Uses a Random Forest Classifier to classify files as packed or not packed.
- **Model Training**: Allows training of the model using custom datasets of packed and unpacked executable files.
- **Pre-trained Model**: A pre-trained model (`packed_detector.pkl`) is included, so users can start classifying files immediately without needing to train the model.
- **File Classification**: Classifies individual files as packed or not packed using the trained or pre-trained model.
- **Performance Metrics**: Provides classification reports and accuracy metrics after training.

## Installation

To get started with BytePackAI, follow these steps:

1. **Clone the repository**:

   git clone https://github.com/gabriel-biudes-dev/BytePackAI.git
   cd BytePackAI

    Install dependencies: BytePackAI requires Python 3.x and several Python libraries. Install them using pip:

    pip install -r requirements.txt

    The required dependencies include:
        numpy
        scikit-learn
        joblib

Train the Model

To train the model, you need two directories: one containing packed executable files and another with unpacked files. Run the following command:

python bytepackai.py train <packed_dir> <unpacked_dir>

    Replace <packed_dir> with the path to the directory containing packed executable files.
    Replace <unpacked_dir> with the path to the directory containing unpacked executable files.

This will extract byte histograms from the files, train the model using the Random Forest Classifier, and save the trained model as packed_detector.pkl.

Classify an Executable File

Once the model is trained, or if you wish to use the pre-trained model, you can use it to classify a single executable file as packed or not packed:

python bytepackai.py classify <file_path>

    Replace <file_path> with the path to the executable file you want to classify.

The output will indicate whether the file is "PACKED" or "NOT PACKED".
Example

    Training the model:

python bytepackai.py train ./packed_files ./unpacked_files

Classifying a file:

python bytepackai.py classify ./example.exe

Output:

    The file './example.exe' is classified as: PACKED

Model Evaluation

After training, the tool evaluates the model's performance on a test set and prints:

    Accuracy: The percentage of correct predictions.
    Classification Report: Detailed precision, recall, and F1-score for both packed and unpacked classes.

Requirements

    Python 3.x
    numpy
    scikit-learn
    joblib

You can install the dependencies using the following command:

pip install -r requirements.txt
