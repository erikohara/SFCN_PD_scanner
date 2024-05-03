from torch.utils.data import Dataset
import torch
import os
import tifffile as tiff
import nibabel as nib

class CustomDataset3D(Dataset):
    def __init__(self, img_dir, df, device='cpu', label_col='Group_bin', label_col_2=None, file_name_col = 'Subject', transform=None):
        self.device = device
        self.df = df
        self.img_dir = img_dir
        self.len = len(self.df)
        self.label_col = label_col
        self.label_col_2 = label_col_2
        self.file_name_col = file_name_col
        self.transform = transform

    # print('data loaded on {}'.format(self.x.device))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        subject_id = str(self.df.loc[index, self.file_name_col])
        site = ''
        if self.label_col_2:
            site = self.df.loc[index, self.label_col_2]
        img_path = os.path.join(self.img_dir, str(subject_id) + '.nii.gz')
        #image = tiff.imread(img_path)
        image = nib.load(img_path).get_fdata()
        label = self.df.loc[index, self.label_col].astype('f4')
        if self.transform:
            image = self.transform(image)
        return image, label, subject_id, site

class CustomDataset(Dataset):
    def __init__(self, img_dir, df, device='cpu', label_col='Group_bin', file_name_col = 'Subject', transform=None):
        self.device = device
        self.df = df
        self.img_dir = img_dir
        self.len = len(self.df)
        self.label_col = label_col
        self.file_name_col = file_name_col
        self.transform = transform

    # print('data loaded on {}'.format(self.x.device))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        subject_id = str(self.df.loc[index, self.file_name_col])
        img_path = os.path.join(self.img_dir, subject_id + '.tiff')
        image = tiff.imread(img_path)
        label = self.df.loc[index, self.label_col].astype('f4')
        if self.transform:
            image = self.transform(image)
        return image, label, subject_id
