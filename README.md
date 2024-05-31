# graph-ml-FER

## Table of Contents
- [File Explanations](#file-explanations)
- [How it Works](#how-it-works)
  - [Landmark Extraction and Normalization](#landmark-extraction-and-normalization)
  - [Dataset Generation](#dataset-generation)
  - [Basic Graph Convolutional Network (GCN)](#basic-graph-convolutional-network-gcn)
  - [Model Evaluation](#model-evaluation)
  - [Inference](#inference)

## File Explanations 
The repository contains the following files and directories:

### ck_data/
- **ck_landmarks.pkl**: A pickle file containing landmarks and bounding box coordinates for facial images. Each row represents an image, and columns include filename, label, bounding box, and facial landmarks.
- **train_data_70_20_10.pkl**: A pickle file containing the training dataset, which is 70% of the total data. This file stores the PyTorch Geometric Data objects for training the model.
- **val_data_70_20_10.pkl**: A pickle file containing the validation dataset, which is 20% of the total data. This file stores the PyTorch Geometric Data objects for validating the model during training.
- **test_data_70_20_10.pkl**: A pickle file containing the testing dataset, which is 10% of the total data. This file stores the PyTorch Geometric Data objects for evaluating the model performance after training.

### fer2013_data/
- **fer2013_landmarks.pkl**: A pickle file containing landmarks and bounding box coordinates for facial images from the FER2013 dataset. Each row represents an image, and columns include filename, label, bounding box, and facial landmarks.
- **train_data_fer2013.pkl**: A pickle file containing the training dataset for the FER2013 dataset. This file stores the PyTorch Geometric Data objects for training the model.
- **val_data_fer2013.pkl**: A pickle file containing the validation dataset for the FER2013 dataset. This file stores the PyTorch Geometric Data objects for validating the model during training.
- **test_data_fer2013.pkl**: A pickle file containing the testing dataset for the FER2013 dataset. This file stores the PyTorch Geometric Data objects for evaluating the model performance after training.

### scripts/
- **__init__.py**: An empty file that indicates to Python that the directory should be considered a package.
- **ck_GINConvBN.pt**: A PyTorch model checkpoint file for the Graph Isomorphism Network (GIN) model trained on the CK+ dataset.
- **image_landmarks_generation.py**: A Python script that contains functions for detecting facial landmarks, normalizing them, and saving the preprocessed data.
- **model_inference.py**: A Python script that loads a trained model and performs inference on landmarks.
- **live_demo.py**: A Python script that uses a webcam feed to detect facial landmarks and predict facial expressions in real-time.
- **plotting_utils.py**: A Python script that contains helper functions for displaying landmarks on faces.
- **reference_image.jpeg**: A reference image used for aligning facial landmarks during normalization.

### test_results/
- **ALL:** Images of the test results including confusion matrix, classification report, training and validation loss and accuracy plots.

### .gitattributes
- **.gitattributes**: A Git configuration file that specifies attributes for pathnames to determine how Git should treat them. Contains a filter for Jupyter Notebook files to strip output cells when committing, to prevent merge conflicts.

### .gitignore
- **.gitignore**: A Git configuration file that specifies files and directories that should be ignored by Git.

### Architecture_tests.md
- **Architecture_tests.md**: A Markdown file that contains the results of testing different Graph Convolutional Network (GCN) architectures on the CK+ dataset. It includes details about the models, training parameters, and performance metrics.

### Basic_GCN.ipynb
- **Basic_GCN.ipynb**: A Jupyter Notebook file that contains the code for building a basic Graph Convolutional Network (GCN) model using PyTorch Geometric. It includes the model architecture, training loop, and evaluation steps.

### ck_dataset_generation.ipynb
- **ck_dataset_generation.ipynb**: A Jupyter Notebook file that contains the code for generating the training, validation, and test datasets from the `ck_landmarks.csv` file. It includes steps for data preprocessing, splitting, and saving the datasets into pickle files.

### fer2013_dataset_generation.ipynb
- **fer2013_dataset_generation.ipynb**: A Jupyter Notebook file that contains the code for generating the training, validation, and test datasets from the `fer2013_landmarks.csv` file. It includes steps for data preprocessing, splitting, and saving the datasets into pickle files.

### standard_mesh_adj_matrix.csv
- **standard_mesh_adj_matrix.csv**: A CSV file that represents the adjacency matrix for a standard mesh. This file is used to define the connections (edges) between nodes (landmarks) in the facial images for graph-based processing.


## How it Works
### Landmark Extraction and Normalization
1. **Face Detection and Landmark Extraction:** Use MediaPipe's Face Mesh to detect facial landmarks in an input image. This step extracts 468 landmarks for each detected face.
2. **Bounding Box Calculation:** Compute the bounding box around the detected face using the extracted landmarks. This helps in isolating the face from the rest of the image.
3. **Landmark Centering:** Center the extracted landmarks to the origin by calculating their centroid and adjusting all landmark coordinates accordingly.
4. **Scaling Landmarks:** Scale the centered landmarks so that their values fit within a range of 0 to 1. This normalization step ensures consistency in landmark values.
5. **Landmark Alignment:** Align the landmarks to a set of reference landmarks using Procrustes analysis. This step standardizes the orientation and position of the face based on reference landmarks extracted from a reference image.
6. **Normalization:** Combine centering, scaling, and alignment steps to fully normalize the landmarks. This results in a consistent representation of facial landmarks across different images.
7. **Data Storage:** Put the image name, expression label, bounding box coordinates, and list of normalized landmarks for each image in a dataframe.
8. **Save Preprocessed Data:** Save the DataFrame to a file using the pickle module for later use.

### Dataset Generation
1. **Load Landmark Data:** Load the preprocessed landmark data from the saved file.
2. **Create Graph Data Objects:** Convert the landmark data into PyTorch Geometric Data objects for graph-based processing. This involves creating nodes for each landmark and defining edges based on the adjacency matrix.
3. **Split Dataset:** Split the dataset into training, validation, and test sets based on the specified ratios (e.g., 70% training, 20% validation, 10% test).
4. **Verify Dataset:** Check the distribution of labels in each dataset to ensure a balanced split.
4. **Save Datasets:** Save the training, validation, and test datasets as pickle files for easy access during model training and evaluation.

### Basic Graph Convolutional Network (GCN)
1. **Define GCN Model:** Implement a basic Graph Convolutional Network (GCN) model using PyTorch Geometric. This model consists of multiple GINConv layers followed by a linear layer for classification.
2. **Class Weights:** Calculate class weights to handle class imbalance in the dataset during training.
3. **Loss Function:** Define the loss function (e.g., Cross Entropy Loss) to optimize the model parameters during training.
4. **Optimizer:** Choose an optimizer (e.g., Adam) to update the model parameters based on the computed gradients.
5. **Early Stopping:** Implement early stopping to prevent overfitting by monitoring the validation loss and stopping training when it starts to increase.
6. **Training Loop:** Define the training loop to optimize the model parameters using backpropagation and gradient descent. This loop includes forward pass, loss calculation, backward pass, and optimizer step.
7. **Save Model:** Save the trained model with the lowest validation loss as a checkpoint file for later use.

### Model Evaluation
1. **Plotting Loss and Accuracy:** Visualize the training and validation loss and accuracy over epochs to monitor the model's learning progress.
2. **Confusion Matrix:** Visualize the confusion matrix to understand the distribution of predicted labels compared to ground truth labels.
3. **Metrics:** Calculate classification metrics such as accuracy, precision, recall, and F1 score to quantify the model's performance.

### Inference
1. **Load Trained Model:** Load the trained GCN model from a saved checkpoint file.
2. **Preprocess Input Data:** Preprocess the input data (e.g., live webcam feed or test images) by detecting facial landmarks, normalizing them, and converting them into PyTorch Geometric Data objects.
2. **Inference on Data:** Perform inference on data to predict facial expressions using the trained model.

## Results
The following first results were obtained by training a Graph Isomorphism Network (GIN) model on the CK+ dataset using the PyTorch Geometric library. The model achieved an accuracy of 80% on the test set.
![](/test_results/GINConvBN.png)