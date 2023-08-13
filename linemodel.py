import torch
import torch.nn as nn
from sklearn.datasets import make_circles
# import matplotlib.pyplot as plt

what_im_building = {1: "data (prep and load)",
                    2: "build model",
                    3: "train",
                    4: "inference",
                    5: "saving and loading a model",
                    6: "put it together"
                    }

# 0. housekeeping
# check pytorch version
print(torch.__version__)

# 1. data (prep and load)
#creating known parameters
weight = 0.7
bias = 0.3

#create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
print(X.shape)

X[:10], y[:10]

#split the dataset
train_split = int(0.8 * len(X)) # 80% of the data for training
print(train_split)
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)

# # 1.5 plot function
# def plot_predictions(train_data=X_train,  
#                     train_labels=y_train,
#                     test_data=X_test,
#                     test_labels=y_test,
#                     predictions=None,):
#     """Plots training data, test data and compares predictions if provided."""
#     plt.figure(figsize=(10,7)) #makes a new plot (graph thingy in python) with a width of 10 and height of 7
#     # Plot training data in blue, the size of each point is set to 4 and the label training data is assigned
#     plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
#     # Plot test data in green, the size of each point is set to 4 and the label test data is assigned
#     plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
#     # If predictions are provided, plot them in red (predictions were made on the test data)
#     if predictions is not None:
#         plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

#     plt.legend(prop={"size": 14})


# 2. build model
class LinearRegressionModel(nn.Module): #almost everything in pytorch is an nn.Module its like staples for neural network construction
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True) #start with random weights (this will get adjusted as the model learns) PyTorch loves float32 by default
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True) #same as above, but for the bias
        
        #forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: #x is the input data (e.g. training/testing features)
        return self.weights * x + self.bias #this is y = mx + b

#set manual seed since nn.Parameter are randomly initialized
torch.manual_seed(42)

##create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
model_0 = LinearRegressionModel()

#check the nn.Parameter(s) within the nn.Module subclass i created
list(model_0.parameters())

#list the named parameters (what the model contains)
model_0.state_dict() 

#Make predictions with model
with torch.inference_mode(): 
    y_preds = model_0(X_test)
#Note: in older PyTorch code you might also see torch.no_grad()
#with torch.no_grad():
#   y_preds = model_0(X_test)

#to check predictions themselves
print(f'Number of testing samples: {len(X_test)}')
print(f'Number of testing samples: {len(y_preds)}')
print(f'Predicted values:\n {y_preds}')


print(y_test - y_preds)

#create the loss function
loss_fn = nn.L1Loss() #MAE loss is same as L1Loss

#create the optimizer 
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.001) #parameters of target model to optimize and the learning rate

torch.manual_seed(42)

# Set the number of epochs (how many times the model will pass over the training data)
epochs = 100

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    ### Training

    # Put model in training mode (this is the default state of a model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside 
    y_pred = model_0(X_train)
    # print(y_pred)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()

    ### Testing

    # Put the model in evaluation mode
    model_0.eval()

    with torch.inference_mode():
        # 1. Forward pass on test data
        test_pred = model_0(X_test)

        # 2. Caculate loss on test data
        test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

        # Print out what's happening
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

# # Plot the loss curves
# plt.plot(epoch_count, train_loss_values, label="Train loss")
# plt.plot(epoch_count, test_loss_values, label="Test loss")
# plt.title("Training and test loss curves")
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend();

# Find our model's learned parameters
print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

# Make 1000 samples 
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
                    noise=0.03, # a little bit of noise to the dots
                    random_state=42) # keep random state so we get the same values