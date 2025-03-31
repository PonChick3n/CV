import torch
import torchvision.transforms.v2 as tfs
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from model import Unet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DriveDataset(data.Dataset):

    def __init__(self, images, masks):
        self.images = images
        self.masks = masks
        self.length = len(images)
        self.transform = tfs.ToDtype(torch.float32, scale=True)
    
    def __getitem__(self, item):
        image = self.transform(self.images[item]).to(device)
        mask = self.transform(self.masks[item]).to(device)
        return (image, mask)
    
    def __len__(self):
        return self.length


train_images = torch.load('datasets/train_images.pt', weights_only=False)
train_masks = torch.load('datasets/train_masks.pt', weights_only=False)
test_images = torch.load('datasets/test_images.pt', weights_only=False)
test_masks = torch.load('datasets/test_masks.pt', weights_only=False)

d_train = DriveDataset(train_images, train_masks)
d_test = DriveDataset(test_images, test_masks)

train_data = data.DataLoader(d_train, batch_size=4, shuffle=True)
test_data = data.DataLoader(d_test, batch_size=len(d_test), shuffle=False)

model = Unet().to(device)
model.train()

optimizer = optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.001)
loss_func = nn.BCELoss()
n_epochs = 5

for _ in range(n_epochs):

    for x_train, y_train in train_data:
        predict = model(x_train).squeeze()
        loss = loss_func(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    for x_test, y_test in test_data:
        predict = model(x_test).squeeze()
        predict = torch.where(predict >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
        score = (predict == y_test).float().mean()
        
print(score)
