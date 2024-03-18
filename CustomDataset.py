import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, csv_path="/home/wdkang/code/CustomUnsupervised/test.csv", is_train=True):

        self.is_train = is_train

        df = pd.read_csv(csv_path)
        img0_paths = df["img0_paths"].tolist()
        img1_paths = df["img1_paths"].tolist()

        self.train_img0_ps, self.test_img0_ps, self.train_img1_ps, self.test_img1_ps = train_test_split(img0_paths, img1_paths, test_size=0.3, random_state=4715)

    def __len__(self):
        if self.is_train:
            return len(self.train_img0_ps)
        else:
            return len(self.test_img0_ps)
    def __getitem__(self, idx):
        if self.is_train:
            img0_path = self.train_img0_ps[idx]
            img1_path = self.train_img1_ps[idx]
            
            img0 = cv2.imread(img0_path)
            img1 = cv2.imread(img1_path)

            return img0, img1
        else:
            img0_path = self.test_img0_ps[idx]
            img1_path = self.test_img1_ps[idx]
            
            img0 = cv2.imread(img0_path)
            img1 = cv2.imread(img1_path)

            return img0, img1