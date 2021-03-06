import os

import numpy as np
from imutils import paths
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

class loadCustomDataset(Dataset):
    def __init__(self, path):
        self.path = path
        # self.transform = transform
        self.img_labels = list(paths.list_images(self.path))
        num_img = len(self.img_labels)
        z=0
        while z < num_img:
            img_path = os.path.join(self.path, self.img_labels[z]) 
            if img_path.split('.')[-1] != 'png' and img_path.split('.')[-1] != 'jpg':
                del self.img_labels[z]
                num_img -= 1
                z-=1
            z+=1


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        path_name = os.path.join(self.path, self.img_labels[item])
        # img_ext = path_name.split('.')[-1]
        # if img_ext == 'jpg' or img_ext == 'png':
        image = cv2.imread(path_name)
        label = path_name.split('\\')[-2]
        image = np.asarray(memoryview(image))
        image = cv2.resize(image, (224, 224))
        image = np.transpose(image, (2,0,1))
        # if self.transform:
        #     image = self.transform(torch.from_numpy(image))
        sample = {'image': torch.from_numpy(image), 'label': label}

        return sample


class Datasets:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def return_total_batches(self, seed):
        dataset = loadCustomDataset(path=self.dataset_path)
        train_set_samples = int(len(dataset.img_labels) - 0.3 * len(dataset.img_labels))  # train_test_split
        test_set_samples = len(dataset.img_labels) - train_set_samples

        trainset, testset = torch.utils.data.random_split(dataset, [train_set_samples, test_set_samples],
                                                          generator=torch.Generator().manual_seed(seed))

        return trainset, testset








