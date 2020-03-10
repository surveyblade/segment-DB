import numpy as np
import cv2
import glob
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
import PIL.Image as Image
import os
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def make_dataset_test(root):
    imgs=[]
    n=len(os.listdir(root))//2
    for i in range(1,21):
        if i <10:
            ind='0'+str(i)
        else:
            ind = str(i)
        img=os.path.join(root,ind+"_test.tif")
        mask=os.path.join(root,ind+"_manual1.gif")
        imgs.append((img,mask))
    return imgs

def make_dataset_train(root):
    imgs=[]
    n=len(os.listdir(root))//2
    for i in range(21,41):
        if i <10:
            ind='0'+str(i)
        else:
            ind = str(i)
        img=os.path.join(root,ind+"_training.tif")
        mask=os.path.join(root,ind+"_manual1.gif")
        imgs.append((img,mask))
    return imgs

class LiverDataset(Dataset):
    def __init__(self, root, path, transform=None, target_transform=None):
        if path == 'train':
            imgs = make_dataset_train(root)
        else:
            imgs = make_dataset_test(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
#         .resize((1608,1068),Image.ANTIALIAS)
        img_x = Image.open(x_path).convert('RGB')
        img_y = Image.open(y_path).convert('L')
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
y_transforms = transforms.ToTensor()

def train_model(model, criterion, optimizer, dataload, num_epochs=21):
    epochloss = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
        epochloss.append(epoch_loss/step)
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    epochloss = np.array(epochloss)
    np.save('train_loss.npy',epochloss)
    return model

def train():
    model = UNet(3, 1).to(device)
    batch_size = 1
    criterion = nn.BCEWithLogitsLoss()
    optimizer=torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    liver_dataset = LiverDataset("/kaggle/input/train/train",'train', transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)
def test():
    model = UNet(3, 1)
    model.load_state_dict(torch.load('/kaggle/working/weights_39.pth',map_location='cpu'))
    liver_dataset = LiverDataset("/kaggle/input/test/test", 'test', transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    criterion = nn.BCEWithLogitsLoss()
    model.eval()
    plt.ion()
    imgs = []
    losses = []
    with torch.no_grad():
        for x, mask in dataloaders:
            y=model(x)
            loss = criterion(y, mask)
            losses.append(loss.item())
            print(loss.item())
            img_y=torch.squeeze(y).numpy()
            imgs.append(img_y)
            plt.imshow(img_y)
            plt.pause(0.01)
        plt.show()
        imgs = np.array(imgs)
        losses = np.array(losses)
        np.save('test_ex.npy',imgs)
        np.save('test_loss.npy',losses)
if __name__ == '__main__':
    #参数解析
    train()
test()
import os
os.listdir('/kaggle/input/train/train')