"""
Contains functionality for training and testing a pytorch model
"""

import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """trains a model for a single epoch

    turns a target model to training mode and then runs through
    all of the required training steps (forward pass, loss calculation, optimizer step)

    Args:
        model: A PyTorch model to train.
        dataloader: A PyTorch DataLoader with training data.
        loss_fn: A PyTorch loss function.
        optimizer: A PyTorch optimizer.
        device: A PyTorch device.


    returns:
        A tuple of (loss, accuracy) for the epoch.
        in the form (train_loss, train_accuracy) for example:
        (1.2345, 0.5678)
    """
    #set model to training mode
    model.train()

    #set train loss and train accuracy to 0
    train_loss == 0, train_acc == 0

    #loop through data loader data batches
    for batch, (X,y) in enumerate(dataloader):
        #send data to device
        X,y = X.to(device), y.to(device)
        #forward pass
        preds = model(X)
        #calculate loss
        loss = loss_fn(preds, y)
        train_loss += loss.item()
        #optimzer zero grad
        optimizer.zero_grad()
        #backward pass
        loss.backward()
        #optimizer step
        optimizer.step()
        #calculate accuracy
        pred_class = torch.argmax(sofmax(preds, dim=1), dim=1)
        train_acc += (pred_class == y).sum().item() / len(preds)

    #adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """tests a model for a single epoch

    Turns a target model to evaluation mode and then runs through
    a forward pass on a testing dataset

    Args:
        model: A PyTorch model to test.
        dataloader: A PyTorch DataLoader with testing data.
        loss_fn: A PyTorch loss function.
        device: A PyTorch device.

    returns:
        A tuple of (loss, accuracy) for the epoch.
        in the form (test_loss, test_accuracy) for example:
        (1.2345, 0.5678)
    """
    #set model to evaluation mode
    model.eval()

    #set test loss and test accuracy to 0
    test_loss == 0, test_acc == 0

    #turn on inference context manager
    with torch.inference_mode():
        #loop through data loader data batches
        for batch, (X,y) in enumerate(dataloader):
            #send data to device
            X,y = X.to(device), y.to(device)
            #forward pass
            preds = model(X)
            #calculate loss
            loss = loss_fn(preds, y)
            test_loss += loss.item()
            #calculate accuracy
            pred_class = torch.argmax(sofmax(preds, dim=1), dim=1)
            test_acc += (pred_class == y).sum().item() / len(preds)
            
