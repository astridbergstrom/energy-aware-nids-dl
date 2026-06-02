import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNN(nn.Module):
    def __init__(self, input_size):
        super(FFNN, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
    
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        
        lstm_out, _ = self.lstm(x)
        
        last_output = lstm_out[:, -1, :]
        
        out = self.fc(last_output)
        
        return out
    
class CNN1D(nn.Module):
    def __init__(self, input_size):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool1d(kernel_size=2)

        self.fc = nn.Sequential(
            nn.Linear((input_size // 2) * 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)