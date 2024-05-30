# graph-ml-FER



## File Explanations

### ck_data/
- **ck_landmarks.pkl**: A pickle file containing landmarks and bounding box coordinates for facial images. Each row represents an image, and columns include filename, label, bounding box, and facial landmarks.
- **train_data_70_20_10.pkl**: A pickle file containing the training dataset, which is 70% of the total data. This file stores the PyTorch Geometric Data objects for training the model.
- **val_data_70_20_10.pkl**: A pickle file containing the validation dataset, which is 20% of the total data. This file stores the PyTorch Geometric Data objects for validating the model during training.
- **test_data_70_20_10.pkl**: A pickle file containing the testing dataset, which is 10% of the total data. This file stores the PyTorch Geometric Data objects for evaluating the model performance after training.

### Dataset_Generation.ipynb
- **Dataset_Generation.ipynb**: A Jupyter Notebook file that contains the code for generating the training, validation, and test datasets from the `ck_landmarks.csv` file. It includes steps for data preprocessing, splitting, and saving the datasets into pickle files.

### standard_mesh_adj_matrix.csv
- **standard_mesh_adj_matrix.csv**: A CSV file that represents the adjacency matrix for a standard mesh. This file is used to define the connections (edges) between nodes (landmarks) in the facial images for graph-based processing.

### Basic_GCN.ipynb
- **Basic_GCN.ipynb**: A Jupyter Notebook file that contains the code for building a basic Graph Convolutional Network (GCN) model using PyTorch Geometric. It includes the model architecture, training loop, and evaluation steps.