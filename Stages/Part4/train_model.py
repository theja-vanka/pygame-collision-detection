from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def train_model(no_epochs):
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"

    device = torch.device(dev)
    print(device)
    batch_size = 256
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    model.to(device)
    loss_function = nn.BCEWithLogitsLoss()
    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch_i in range(no_epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        # sample['input'] and sample['label']
        for idx, sample in enumerate(data_loaders.train_loader):
            inpt = sample['input'].to(device)
            labels = sample['label'].to(device)
            labels = labels.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inpt)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'| Epoch: {epoch_i+1}', end=" | ")
        print(f'Loss: {epoch_loss/len(data_loaders.train_loader):.4f} |')
        #print(f'Acc: {epoch_acc/len(data_loaders.train_loader):.3f}')
        model.eval()
        with torch.no_grad():
            test_loss = model.evaluate(
                    model,
                    data_loaders.test_loader,
                    loss_function
                )
            print(f'------- Test Loss: {test_loss:.4f} -------')
            losses.append(test_loss)
            PATH = f"saved/weights/weights_{test_loss:.3f}.pkl"
            torch.save(model.state_dict(), PATH, _use_new_zipfile_serialization=False)

if __name__ == '__main__':
    no_epochs = 1000
    train_model(no_epochs)
