import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
from model import Unet
from dataprep import DriveDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_images = torch.load('datasets/train_images.pt', weights_only=False)
train_masks = torch.load('datasets/train_masks.pt', weights_only=False)
test_images = torch.load('datasets/test_images.pt', weights_only=False)
test_masks = torch.load('datasets/test_masks.pt', weights_only=False)

d_train = DriveDataset(train_images, train_masks)
d_test = DriveDataset(test_images, test_masks)

if __name__ == '__main__':

    train_data = data.DataLoader(d_train, batch_size=4, shuffle=True)
    test_data = data.DataLoader(d_test, batch_size=len(d_test), shuffle=False)

    model = Unet().to(device)
    model.train()

    optimizer = optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.001)
    loss_func = nn.BCEWithLogitsLoss()
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
        
    print(score.item())

    st = model.state_dict()
    torch.save(st, 'model_unet_seg.tar')
