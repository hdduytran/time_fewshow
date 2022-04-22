from torch import nn 

class ResidualBlock(nn.Module):
  def __init__(self,input_channels,num_channel = 32,kernel_size = 5, padding = 2,strides = 1):
    super().__init__()
    self.a = 1
    self.conv1 = nn.Conv1d(input_channels,num_channel,kernel_size = kernel_size, padding = padding, stride = strides)
    self.conv2 = nn.Conv1d(num_channel,num_channel,kernel_size = kernel_size, padding = padding, stride = strides)
    self.pool = nn.MaxPool1d(kernel_size,strides,padding)
    self.relu = nn.ReLU()
  def forward(self,X):
    Y = self.conv1(X)
    Y = self.relu(Y)
    Y = self.conv2(Y)
    Y += X
    Y = self.relu(Y)
    Y = self.pool(Y)
    return Y