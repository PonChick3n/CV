import deeplake
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as tfs
import torch.utils.data as data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resize(x):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = F.interpolate(x, size=(576, 576), mode='bilinear', align_corners=False)
    return x.squeeze()


class DriveDataset(data.Dataset):

    def __init__(self, images, masks):
        self.images = images
        self.masks = masks
        self.length = len(images)
        self.transform = tfs.ToDtype(torch.float32, scale=True)
        self.resize = resize
    
    def __getitem__(self, item):
        image = self.resize(self.transform(self.images[item]).to(device))
        mask = self.resize(self.transform(self.masks[item]).to(device))
        return (image, mask)
    
    def __len__(self):
        return self.length

if __name__ == '__main__':
    
    d_train = deeplake.load("hub://activeloop/drive-train")
    d_test = deeplake.load("hub://activeloop/drive-test")

    train_images = torch.tensor(d_train['rgb_images'].numpy()).permute(0, 3, 1, 2)
    train_masks = torch.tensor(d_train['masks/mask'].numpy()).permute(0, 3, 1, 2)
    test_images = torch.tensor(d_test['rgb_images'].numpy()).permute(0, 3, 1, 2)
    test_masks = torch.tensor(d_test['masks'].numpy()).permute(0, 3, 1, 2)

    torch.save(train_images, 'datasets/train_images.pt')
    torch.save(train_masks, 'datasets/train_masks.pt')
    torch.save(test_images, 'datasets/test_images.pt')
    torch.save(test_masks, 'datasets/test_masks.pt')
