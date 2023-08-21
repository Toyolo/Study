import torch
from torch import nn 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


training_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor(), target_transform=None)
testing_data = datasets.FashionMNIST('data', train=False, download=False, transform=ToTensor())
class_names = training_data.classes
BATCHSIZE = 32
training_dataloader = DataLoader(testing_data, batch_size=BATCHSIZE, shuffle=False)
testing_dataloader = DataLoader(testing_data, batch_size=BATCHSIZE, shuffle=False)

class CvModel(nn.Module):
    def __init__(self, input, hiddenput, output):
        super().__init__()
        self.network_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input, hiddenput),
            nn.ReLU(),
            nn.Linear(hiddenput, output),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network_stack(x)


model = CvModel(784, 10, len(class_names))
model.to('cpu')

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 0.1)

torch.manual_seed(42)
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"epoch:{epochs}\n----------")
    train_loss = 0
    for batch, (X,y) in enumerate(training_dataloader):
        model.train()
        prediction = model(X)
        loss = loss_fn(prediction, y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(training_dataloader.dataset)} samples")
    train_loss /= len(training_dataloader)
    test_loss, test_acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X, y in testing_dataloader:
            test_predictons = model(X)
            test_loss += loss_fn(test_predictons, y)
            test_acc += accuracy_fn(y, test_predictons.argmax(dim=1))
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")
    
