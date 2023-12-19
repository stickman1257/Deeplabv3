import cv2
import pandas as pd
import numpy as np
from multiprocessing import freeze_support

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Union
from joblib import Parallel, delayed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rle_decode(mask_rle: Union[str, int], shape=(224, 224)) -> np.array:
    
    if mask_rle == -1:
        return np.zeros(shape)
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def dice_score(prediction: np.array, ground_truth: np.array, smooth=1e-7) -> float:
    
    intersection = np.sum(prediction * ground_truth)
    return (2.0 * intersection + smooth) / (np.sum(prediction) + np.sum(ground_truth) + smooth)


def calculate_dice_scores(ground_truth_df, prediction_df, img_shape=(224, 224)) -> List[float]:
    
    prediction_df = prediction_df[prediction_df.iloc[:, 0].isin(ground_truth_df.iloc[:, 0])]
    prediction_df.index = range(prediction_df.shape[0])


    
    pred_mask_rle = prediction_df.iloc[:, 1]
    gt_mask_rle = ground_truth_df.iloc[:, 1]


    def calculate_dice(pred_rle, gt_rle):
        pred_mask = rle_decode(pred_rle, img_shape)
        gt_mask = rle_decode(gt_rle, img_shape)


        if np.sum(gt_mask) > 0 or np.sum(pred_mask) > 0:
            return dice_score(pred_mask, gt_mask)
        else:
            return None 


    dice_scores = Parallel(n_jobs=-1)(
        delayed(calculate_dice)(pred_rle, gt_rle) for pred_rle, gt_rle in zip(pred_mask_rle, gt_mask_rle)
    )


    dice_scores = [score for score in dice_scores if score is not None]  


    return np.mean(dice_scores)


class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
    
transform = A.Compose(
    [   
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ]
)


train_dataset = SatelliteDataset(csv_file='./train.csv', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)


train_size = int(0.9 * len(train_dataset))
valid_size = len(train_dataset) - train_size
train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)



deeplab = models.segmentation.deeplabv3_resnet101(pretrained=False)
model = deeplab.to(device)
model.classifier[-1] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

model = model.to(device)

def plot_losses(train_losses, valid_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    freeze_support()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    valid_losses = []
    train_losses = []
    for epoch in range(100): 
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(train_dataloader):
            images = images.float().to(device)
            masks = masks.float().to(device) 

            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs['out']
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(train_loss)
        
        model.eval()
        valid_loss = 0
        with torch.no_grad():        
            for images, masks in tqdm(valid_dataloader):
                images = images.float().to(device)
                masks = masks.float().to(device)
                outputs = model(images)
                outputs = outputs['out']
                loss = criterion(outputs, masks.unsqueeze(1))
                valid_loss += loss.item()
                
        valid_loss = valid_loss / len(valid_dataloader)
        valid_losses.append(valid_loss) 
    
        print(f'Train Epoch {epoch+1}, Loss: {train_loss}')
    
        print(f'Valid Epoch {epoch+1}, Loss: {valid_loss}')
        torch.save(model.state_dict(), f'./checkpoint/epoch{epoch}_{epoch_loss}.pth')
    plot_losses(train_losses, valid_losses)

 
