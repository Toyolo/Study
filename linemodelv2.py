#imports
import torch
import torch.nn as nn

torch.__version__

#make device agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#create weights and biases
weight = 0.7
bias = 0.3

#create range values
start = 0
end = 1
step = 0.02

#create X and y (features and labels)
X = torch.arange(start, end, step).unsqueeze(dim=1) #without unsqueeze errors will occur like "RuntimeError: Expected 2-dimensional input for 2-dimensional weight [1, 1], but got 1-dimensional input of size [40] instead"
y = weight * X + bias
X[:10], y[:10]

#split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print(len(X_train), len(y_train), len(X_test), len(y_test))

#subclass nn.Module to make the model
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        #Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1, 
                                      out_features=1)
    
    #define the forward computation (input data x flows through nn.Linear())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

#set the manual seed for reproducibility
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
model_1, model_1.state_dict()

#check model device
next(model_1.parameters()).device


model_1.to(device)
next(model_1.parameters()).device

#create loss function
loss_fn = nn.MSELoss()

#create optimizer
optimizer = torch.optim.Adam(model_1.parameters(), lr=0.01)

torch.manual_seed(42)

# Set the number of epochs 
epochs = 300000

# Put data on the available device
# Without this, error will happen (not all model/data on device)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    ### Training
    model_1.train() # train mode is on by default after construction

    # 1. Forward pass
    y_pred = model_1(X_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad optimizer
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Step the optimizer
    optimizer.step()

    ### Testing
    model_1.eval() # put the model in evaluation mode for testing (inference)
    # 1. Forward pass
    with torch.inference_mode():
        test_pred = model_1(X_test)
    
        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

#set model to evaluation mode
model_1.eval()

#make predictions on test data
with torch.inference_mode():
    y_preds = model_1(X_test)
y_preds

# Find our model's learned parameters
from pprint import pprint # pprint = pretty print, see: https://docs.python.org/3/library/pprint.html 
print("The model learned the following values for weights and bias:")
pprint(model_1.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")