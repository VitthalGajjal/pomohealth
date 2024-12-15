Image Classification for Pomegranate Diseases

This project is an Image Classification Model designed to detect and classify pomegranate diseases. The model is built using TensorFlow and Keras, and it identifies the following categories of pomegranate conditions:

Alternaria

Anthracnose

Bacterial Blight

Cercospora

Healthy



The project includes two main components:

The model training script, which defines and trains the neural network.

A Streamlit-based web interface for predicting the disease category from an input image.



Installation and Setup

Prerequisites

Make sure you have the following installed:

Python (>= 3.8)

TensorFlow (>= 2.0)

Streamlit (>= 1.0)

NumPy

Pandas

Matplotlib



Directory Structure

Ensure your project has the following directory structure:

project-directory/
├── train/                # Training dataset
├── validation/           # Validation dataset
├── test/                 # Test dataset
├── model/                # Saved models
├── main.py               # Streamlit interface
├── train_model.py        # Model training script
├── README.md             # Project documentation




Installation Steps

Clone the repository or copy the project files.


Install the required dependencies:

pip install tensorflow streamlit numpy pandas matplotlib

Ensure that your dataset is organized into the train, validation, and test folders with subfolders for each category.

Model Training

Dataset Preparation


Path to Datasets:

Training: C:/Users/Rutuja/Downloads/Image_classification/train

Validation: C:/Users/Rutuja/Downloads/Image_classification/validation

Test: C:/Users/Rutuja/Downloads/Image_classification/test

The dataset should be organized into subfolders for each class (e.g., Alternaria/, Anthracnose/, etc.), with images stored inside these subfolders.


Training Script

The train_model.py script performs the following steps:

Loads the dataset using tf.keras.utils.image_dataset_from_directory().

Defines a CNN model using the Keras Sequential API:

-Convolutional Layers for feature extraction.

-MaxPooling Layers for dimensionality reduction.

-Dense Layers for classification.

Compiles the model with the Adam optimizer and sparse categorical cross-entropy loss.

Trains the model for 20 epochs, displaying training and validation accuracy and loss.

Saves the trained model to Image_classify.keras.



Training Command

Run the following command to train the model:

python train_model.py



Web Interface

Overview

The main.py script creates an interactive web interface using Streamlit. Users can upload an image, and the model predicts the disease category along with confidence levels.



Key Features

Image Upload: Users can input the path to the image file.

Prediction Output:

-The classified disease category.

-Prediction accuracy (confidence score).

Visualization: Displays the input image.



Running the Web Interface

To run the Streamlit app, execute the following command:

streamlit run main.py

Input Example

Enter the name of an image file (e.g., anthracnose.jpg) when prompted. Ensure the image is in the current working directory or provide the full path.



Results and Evaluation

Training Results

After training, the model achieved the following performance metrics:

Training Accuracy: Displayed per epoch.

Validation Accuracy: Monitored during training.

Model Testing

To evaluate the model on the test dataset:

Load the test dataset using image_dataset_from_directory().

Use the trained model to predict labels for the test images.

Compare predictions against ground truth labels.



Example Prediction

disease in image is Anthracnose with accuracy of 92.35%



File Descriptions

train_model.py

Script for training the image classification model.

Defines the CNN architecture and trains it on the provided dataset.

Saves the model to Image_classify.keras.

main.py

Streamlit script for deploying the model as a web application.

Loads the trained model and provides an interface for predictions.

Image_classify.keras

The saved trained model.




Troubleshooting

Common Issues

Model file not found:

Ensure the path to Image_classify.keras in main.py is correct.

Invalid image input:

Verify the image path and ensure the image is in a supported format (e.g., .jpg, .png).

Dependencies not installed:

Run pip install -r requirements.txt to install all dependencies.

Future Enhancements

Add support for more disease categories.

Optimize model performance for faster predictions.

Implement a drag-and-drop interface for image uploads.
