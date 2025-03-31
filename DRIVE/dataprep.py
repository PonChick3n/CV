import deeplake
import torch

d_train = deeplake.load("hub://activeloop/drive-train")
d_test = deeplake.load("hub://activeloop/drive-test")

train_images = torch.tensor(d_train['rgb_images'].numpy()).permute(0, 3, 1, 2)
train_masks = torch.tensor(d_train['masks/mask'].numpy()).permute(0, 3, 1, 2)
test_images = torch.tensor(d_test['rgb_images'].numpy()).permute(0, 3, 1, 2)
test_masks = torch.tensor(d_test['masks'].numpy()).permute(0, 3, 1, 2)

torch.save(train_images, 'datasets/train_images.pt')
torch.save(train_masks, 'datasets/train_masks.pt')
torch.save(train_images, 'datasets/test_images.pt')
torch.save(train_images, 'datasets/test_masks.pt')
