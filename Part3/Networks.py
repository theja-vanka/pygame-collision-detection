import torch
import torch.nn as nn


class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        # __init__() must initiatize nn.Module and define your network's
        super(Action_Conditioned_FF, self).__init__()
        # custom architecture
        self.l1 = nn.Linear(6, 784)
        self.l2 = nn.Linear(784, 128)
        self.l3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, input):
        # forward() must complete a single forward pass through your network
        # and return the output which should be a tensor
        x = self.l1(input)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x

    def evaluate(self, model, test_loader, loss_function):
        # evaluate() must return the loss (a value, not a tensor)
        # over your testing dataset.
        # Keep in mind that we do not need to keep track of
        # any gradients while evaluating the
        # model. loss_function will be a PyTorch loss function
        # which takes as argument the model's
        # output and the desired output.
        loss=0
        index = 0
        for x in test_loader:
            x_input = x['input']
            y_true = x['label']
            y_pred = model(x_input)
            simpleloss = loss_function(y_pred, y_true)
            loss = loss + simpleloss.item()
            index = index + 1
        return loss/index


def main():
    model = Action_Conditioned_FF()


if __name__ == '__main__':
    main()
