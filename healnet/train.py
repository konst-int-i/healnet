# Implement train loop
import torch
import torch.optim as optim
import torch.nn as nn
from typing import *
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.dummy import DummyClassifier
import numpy as np


def train_loop(
    preprocess_fn: Callable,
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    epochs: int=1,
    verbose:bool=True,
    max_lr: float = 0.05,
    lr: float = 0.005,
    momentum=0.9,
    **kwargs
) -> None:

    n_batches = len(trainloader)
    majority_val_acc = majority_classifier_acc(y_true=testloader.dataset.targets)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # set efficient OneCycle scheduler, significantly reduces required training iters
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=n_batches)

    criterion = nn.CrossEntropyLoss()
    logs_per_epoch = 10

    if n_batches < logs_per_epoch:
        logs_per_epoch = n_batches

    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader)):

            inputs, labels = data
            if preprocess_fn is not None:
                inputs = preprocess_fn(inputs)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # print statistics
            running_loss += loss.item()

            if verbose and i % (int(n_batches/logs_per_epoch)) == 0:
                print(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] train_loss: {running_loss / 2000:.3f}')

            running_loss = 0.0
    print('Finished Training')
    accuracy, f1, precision, recall = validate(model, testloader, majority_val_acc)


def validate(model: nn.Module, data_loader: torch.utils.data.DataLoader, majority_val_acc: float):

    print("Running validation...")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        y_true = []
        y_pred = []
        for batch in data_loader:
            inputs, labels = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.to(device).numpy())
            y_pred.extend(predicted.to(device).numpy())

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        # auc = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")

    print(f"n_classes: {len(set(y_true))}")
    print(f"accuracy: {accuracy:.4f}, f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, "
          f"majority_val_acc: {majority_val_acc:.4f}")

    # resume train mode
    model.train()
    return accuracy, f1, precision, recall

def majority_classifier_acc(y_true: List):
    """
    Returns accuracy of majority class classifier
    """
    X = np.ones(len(y_true))
    dummy = DummyClassifier(strategy="most_frequent").fit(X=X, y=y_true)
    y_majority = dummy.predict(X)
    return accuracy_score(y_true, y_majority)
