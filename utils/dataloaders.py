import os
import random
import cv2
import numpy as np
import torch.utils.data as data



def label_loader(label_path):
    label = cv2.imread(label_path, 0) / 255   # flag=0:  Return a grayscale image.
    return label



def full_path_loader(data_dir):
    train_data = [i for i in os.listdir(data_dir + 'train/A/') if not
    i.startswith('.')]
    train_data.sort()

    valid_data = [i for i in os.listdir(data_dir + 'val/A/') if not
    i.startswith('.')]
    valid_data.sort()

    train_label_paths = []
    val_label_paths = []
    for img in train_data:
        train_label_paths.append(data_dir + 'train/OUT/' + img)
    for img in valid_data:
        val_label_paths.append(data_dir + 'val/OUT/' + img)


    train_data_path = []
    val_data_path = []

    for img in train_data:
        train_data_path.append([data_dir + 'train/', img])
    for img in valid_data:
        val_data_path.append([data_dir + 'val/', img])

    train_dataset = {}
    val_dataset = {}
    for cp in range(len(train_data)):
        train_dataset[cp] = {'image': train_data_path[cp],
                         'label': train_label_paths[cp]}
    for cp in range(len(valid_data)):
        val_dataset[cp] = {'image': val_data_path[cp],
                         'label': val_label_paths[cp]}


    return train_dataset, val_dataset

def full_test_loader(data_dir):

    test_data = [i for i in os.listdir(data_dir + 'test/A/') if not
                    i.startswith('.')]
    test_data.sort()

    test_label_paths = []
    for img in test_data:
        test_label_paths.append(data_dir + 'test/OUT/' + img)

    test_data_path = []
    for img in test_data:
        test_data_path.append([data_dir + 'test/', img])

    test_dataset = {}
    for cp in range(len(test_data)):
        test_dataset[cp] = {'image': test_data_path[cp],
                           'label': test_label_paths[cp]}

    return test_dataset

def cdd_loader(img_path, label_path, aug):
    dir = img_path[0]
    name = img_path[1]

    bands_date1 = []
    bands_info1 = cv2.split(cv2.imread(dir + 'A/' + name))
    for i in range(len(bands_info1)):
        bands_date1.append(bands_info1[i])

    bands_date2 = []
    bands_info2 = cv2.split(cv2.imread(dir + 'B/' + name))
    for i in range(len(bands_info2)):
        bands_date2.append(bands_info2[i])

    out_img = np.stack((bands_date1, bands_date2))
    out_lbl = label_loader(label_path)

    if aug:
        rot_deg = random.randint(0, 3)
        out_img = np.rot90(out_img, rot_deg, [2, 3]).copy()
        out_lbl = np.rot90(out_lbl, rot_deg, [0, 1]).copy()

        if random.random() > 0.5:
            out_img = np.flip(out_img, axis=2).copy()
            out_lbl = np.flip(out_lbl, axis=0).copy()

        if random.random() > 0.5:
            out_img = np.flip(out_img, axis=3).copy()
            out_lbl = np.flip(out_lbl, axis=1).copy()

    return out_img[0], out_img[1], out_lbl


class CDDloader(data.Dataset):

    def __init__(self, full_load, aug=False):
        random.shuffle(full_load)

        self.full_load = full_load
        self.loader = cdd_loader
        self.aug = aug

    def __getitem__(self, index):

        img_path, label_path = self.full_load[index]['image'], self.full_load[index]['label']

        return self.loader(img_path,
                           label_path,
                           self.aug)

    def __len__(self):
        return len(self.full_load)
