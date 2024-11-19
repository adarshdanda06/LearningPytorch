import torch
from torch.autograd import Variable
import numpy as np

# Data preparation
x_values = [i for i in range(20)]
x_train = np.array(x_values, dtype=np.float32).reshape(-1, 1)
y_values = [4 * i + 14 for i in x_values]
y_train = np.array(y_values, dtype=np.float32).reshape(-1, 1)


print(4 * 4 + 14)
# Convert to PyTorch tensors
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
print(x_train)
print(y_train)


# Normalize data
#x_train = (x_train - x_train.mean()) / x_train.std()
#y_train = (y_train - y_train.mean()) / y_train.std()

#print(x_train)
#print(y_train)

# Model definition
class LinRegModel(torch.nn.Module):
    def __init__(self):
        super(LinRegModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinRegModel()

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Training loop
for epoch in range(200):
    pred_y = model(x_train)
    loss = criterion(pred_y, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Test prediction
new_var = Variable(torch.Tensor([[4.0]]))
pred_y = model(new_var)
print(pred_y)
