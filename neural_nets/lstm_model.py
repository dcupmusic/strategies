import torch
import torch.nn as nn
import torch.nn.functional as F




class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        weights = torch.tanh(self.linear(lstm_output))
        weights = F.softmax(weights, dim=1)
        
        context = weights * lstm_output
        context = torch.sum(context, dim=1)
        return context, weights

class BiLSTMClassifierWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate):
        super(BiLSTMClassifierWithAttention, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)        
        self.bn_conv1 = nn.BatchNorm1d(hidden_dim)        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)        
        self.attention = Attention(hidden_dim * 2)        
        self.dropout = nn.Dropout(dropout_rate)        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)        
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)        
        self.fc2 = nn.Linear(hidden_dim, output_dim)        
        self.dropout_fc = nn.Dropout(dropout_rate / 2)

    def forward(self, x):
        if x is None:
            raise ValueError("Input to the model's forward method is None")
        x = x.permute(0, 2, 1)        
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.relu(x)        
        x = x.permute(0, 2, 1)        
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)        
        out, _ = self.lstm(x, (h0, c0))        
        context, _ = self.attention(out)        
        out = self.fc1(context)
        out = self.bn_fc1(out)
        out = F.relu(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        
        return out

