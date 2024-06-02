import torch.nn as nn
import torch.nn.functional as F


class InceptModel(nn.Module):
    def __init__(self):
        super(InceptModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4096 * 4, 2048)
        self.fc_agg = nn.Linear(2048, 1024)  # Adjust depending on aggregation method
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 4)

    def forward(self, x):
        x = x.view(-1, 4096 * 4)  # Adjust view based on aggregation method
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc_agg(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class CLIPModel(nn.Module):
    def __init__(self):
        super(CLIPModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 4, 2048)
        self.fc_agg = nn.Linear(2048, 1024)  # Adjust depending on aggregation method
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 4)

    def forward(self, x):
        x = x.view(-1, 512 * 4)  # Adjust view based on aggregation method
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc_agg(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
