import pandas as pd
import numpy as np
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout_rate=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)                  
        out = out[:, -1, :]                   
        out = self.dropout(out)                
        out = self.fc(out)                    
        return out
