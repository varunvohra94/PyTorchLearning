import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim,
        loss_fn: torch.nn.Module,
        device: torch.device
) -> Tuple[float, float]:

    """
    Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:
        (0.1112, 0.8743)
    """

    # Put the model in train mode
    model.train()
    model.to(device)

    train_loss, train_acc = 0
    for X, y in dataloader:
        # Send data to the right device
        X, y = X.to(device), y.to(device)

        # 1. forward Pass
        y_logits = model(X)
        y_preds = torch.argmax(torch.softmax(X, dim=1), dim=1)

        # 2. Calculate the loss and accuracy
        loss = loss_fn(y_logits, y)
        train_loss = train_loss + loss.item()
        train_acc += (y_preds == y).sum().item()/len(y_preds)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
    
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
):
    # Put the model in eval mode and on the device
    model.eval()
    model.to(device)

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Calculate the logits and predictions
            y_logits = model(X)
            y_preds = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)

            # Calculate the loss and accuracy
            loss = loss_fn(y_logits, y)
            test_loss += loss.item()
            test_acc += (y_preds==y).sum().item()/len(y_preds)

        test_loss = test_loss/len(dataloader)
        test_acc = test_acc/len(dataloader)
    
    return test_loss, test_acc

def train(
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        epochs: int,
) -> Dict[str, List]:
    """
    Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]} 
        For example if training for epochs=2: 
                    {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]} 
    """

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_loss": []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        print(
            f"Epoch: {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: {test_acc:.4f} | "
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    
    return results