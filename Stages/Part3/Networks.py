import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        # __init__() must initiatize nn.Module and define your network's
        super(Action_Conditioned_FF, self).__init__()
        # custom architecture
        self.layer_1 = nn.Linear(6, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_4 = nn.Linear(64, 16)
        self.layer_out = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        # forward() must complete a single forward pass through your network
        # and return the output which should be a tensor
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))
        x = self.layer_out(x)
        x = self.sigmoid(x)
        return x

    def evaluate(self, model, test_loader, loss_function):
        # evaluate() must return the loss (a value, not a tensor)
        # over your testing dataset.
        # Keep in mind that we do not need to keep track of
        # any gradients while evaluating the
        # model. loss_function will be a PyTorch loss function
        # which takes as argument the model's
        # output and the desired output.
        loss = 0
        index = 0
        for x in test_loader:
            x_input = x['input']
            y_true = x['label']
            y_true = y_true.unsqueeze(1)
            y_pred = model(x_input)
            simpleloss = loss_function(y_pred, y_true)
            loss = loss + simpleloss.item()
            index = index + 1
        return loss/index


def main():
    model = Action_Conditioned_FF()


if __name__ == '__main__':
    main()
