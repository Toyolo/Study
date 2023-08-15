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

X_train, y_train, X_test, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

device = "cuda" if torch.cuda.is_available() else "cpu"

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=5)
        self.layer2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer2(self.layer1(x))
    
model = CircleModelV0()
untrained_preds = model(X_test.to(device))

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

y_logits = model(X_test.to(device))
