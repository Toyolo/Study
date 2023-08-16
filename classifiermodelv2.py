import torch
import torch.nn as nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

n_samples = 1000

X, y = make_circles(n_samples, noise=0.03, random_state=42)

if not torch.is_tensor(X):
    X = torch.from_numpy(X).type(torch.float)
if not torch.is_tensor(y):
    y = torch.from_numpy(y).type(torch.float)



device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.network_stack = nn.Sequential(
            nn.Linear(in_features=2, out_features=10),
            nn.Linear(in_features=10, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
    def forward(self, x):
        return self.network_stack(x)

model = CircleModelV2()
untrained_preds = 