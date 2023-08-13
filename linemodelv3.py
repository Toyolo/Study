import torch
import torch.nn as nn
from pathlib import Path

#1 data and data processing
weight, bias = 0.3, 0.9


#create data
start, end, step = 0, 100, 1
X = torch.arange(start, end, step)
y = weight * X + bias
train_split = int(0.8 * len(X))
X_train, X_test = X[:train_split], X[train_split:]
y_train, y_test = y[:train_split], y[train_split:]




#2 model
class LinearRegressionModelv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


torch.manual_seed(42)
model = LinearRegressionModelv3()
list(model.parameters())
model.state_dict()

#3 loss and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
#4 training loop
torch.manual_seed(42)
epochs = 400
trainlossvals = []
testlossvals = []
epochcount = []

for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #5 testing loop
    model.eval()
    with torch.inference_mode():
        testpred = model(X_test)
        testloss = loss_fn(testpred, y_test.type(torch.float))
        testlossvals.append(testloss.detach().numpy())
        print(f'epoch:{epoch} | train loss: {loss}, test loss: {testloss}')

# Find our model's learned parameters
print("The model learned the following values for weights and bias:")
print(model.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

#6 save model
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "pytorch_linear_model_3.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
print(f"Saving model state dict at: {MODEL_SAVE_PATH}")
torch.save(model.state_dict(), MODEL_SAVE_PATH) 
#7 load model

