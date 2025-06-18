import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        weights = F.softmax(self.fc2(x), dim=-1)  #softmax across experts
        return weights
