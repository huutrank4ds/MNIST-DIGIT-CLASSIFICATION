import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2) # (1 x 5 x 5 + 1) x 20 = 520 params
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2) # output (batch_size, 20, 14, 14)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3920, 10) # (20 x 14 x 14  + 1) x 10 = 39210 params
        self.softmax = nn.Softmax(dim=1)
        self.model = nn.Sequential(
            self.conv2d,
            self.relu, 
            self.maxpool,
            self.flatten,
            self.fc,
            self.softmax
        )

    @staticmethod
    def reshape(inputs):
        return inputs.unsqueeze(dim=1)
    
    # cross entropy loss
    def loss(self, outputs, labels):
        log_outputs = torch.log(outputs)
        loss = -torch.sum(labels * log_outputs, dim=1)
        return torch.mean(loss)
    
    def forward(self, inputs, labels=None):
        inputs = self.reshape(inputs)
        outputs = self.model(inputs)
        if labels is not None:
            loss = self.loss(outputs, labels)
            return outputs, loss
        return outputs