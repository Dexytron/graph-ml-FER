import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool, BatchNorm
import scripts


class SimpleGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = BatchNorm(hidden_dim)

        nn2 = torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = BatchNorm(hidden_dim)

        self.lin = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x.to(torch.float), data.edge_index.to(torch.int64), data.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)


dataset = 'ck'
# dataset = 'fer2013'
if dataset == 'ck':
    output_dim = 8
    label_mapping = {'neutral': 0, 'happiness': 1, 'sadness': 2, 'surprise': 3, 'fear': 4, 'disgust': 5, 'anger': 6, 'contempt': 7}
    model_path = scripts.__path__[0] + '/ck_GINConvBN.pt'
elif dataset == 'fer2013':
    output_dim = 7
    label_mapping = {'neutral': 0, 'happy': 1, 'sad': 2, 'surprise': 3, 'fear': 4, 'disgust': 5, 'angry': 6}
    model_path = scripts.__path__[0] + '/fer2013_GINConvBN.pt'
else:
    raise ValueError('Invalid dataset')
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleGCN(input_dim=3, hidden_dim=128, output_dim=output_dim)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Create edge_index from the graph once
adjacency_matrix = np.loadtxt(scripts.__path__[0] + '/../standard_mesh_adj_matrix.csv', delimiter=',')
graph = nx.from_numpy_array(adjacency_matrix)
edge_index = []
for edge in graph.edges:
    edge_index.append([edge[0], edge[1]])
    edge_index.append([edge[1], edge[0]])
edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous()


# Inference function
def infer_model(landmarks) -> str:
    """"
    Function to perform inference on the trained model
    :param landmarks: normalized landmarks
    :return: predicted expression
    """
    model.eval()

    with torch.no_grad():
        # Prepare the data object
        x = torch.tensor(landmarks, dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index, batch=torch.zeros(x.size(0), dtype=torch.long))
        data = data.to(device)

        # Perform inference
        out = model(data)
        pred = out.argmax(dim=1)

    return inverse_label_mapping[pred.cpu().numpy()[0]]


if __name__ == '__main__':
    import image_landmarks_generation as ilg
    test_predictions = infer_model(ilg.reference_landmarks)
    print(test_predictions)
