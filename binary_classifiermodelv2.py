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

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


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

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params= model.parameters(), lr=0.1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

logits = model(X_test.to(device))
pred_probs = torch.sigmoid(logits)

torch.manual_seed(42)

epochs = 1000
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

for epoch in range(epochs):
    model.train()
    logits = model(X_train).squeeze()
    predictions = torch.round(torch.sigmoid(logits))
    loss = loss_fn(logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=predictions)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_predictions = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_predictions)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
    
