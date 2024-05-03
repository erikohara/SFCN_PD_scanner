import torch
import pickle
import pandas as pd
import numpy as np
from dataset import CustomDataset3D
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms 
from customTransforms import ToFloatUKBB, Crop3D
from models import SFCNModelScannerMONAI
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('one_site')
args = parser.parse_args()
one_site = int(args.one_site)

#images_path = '/work/forkert_lab/erik/mitacs_dataset'
images_path = '/work/forkert_lab/mitacs_dataset/affine_using_nifty_reg'
cf_image_path = '/work/forkert_lab/erik/MACAW/cf_images/mitacs_site_aside'
dataframe_path = '/home/erik.ohara/SFCN_PD_scanner'
output_path = '/home/erik.ohara/SFCN_PD_scanner/models_scanner'

batch_size = 4
crop_size = (100,150,150)
epochs = 100
lr = 0.0001
training = False
categories = 10

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print(device)

# Loading traning, validation and test image paths
#image_path_train = images_path + '/train_site'
#image_path_val = images_path + '/val_site'
#image_path_test = images_path + '/test_site'

# Loading dataframes (patient info, scanner, site)
#df_old = pd.read_csv(dataframe_path + '/new_train_df_2.csv',low_memory=False)
#df_train = pd.DataFrame(columns=df_old.columns)
df_train = pd.read_csv(dataframe_path + '/new_train_df_2.csv',low_memory=False)
df_val = pd.read_csv(dataframe_path + '/new_val_df_2.csv',low_memory=False)

df_all = pd.read_csv(dataframe_path + '/all_df.csv',low_memory=False)
df_test = df_all[df_all['Site_3'] == one_site].copy().reset_index()
study_aside = df_test['Study'].unique()[0]
print(study_aside)

dataset_train = CustomDataset3D(images_path, df_train ,device, 
                              label_col='Site_3', file_name_col = 'Subject', 
                              transform=transforms.Compose([ToFloatUKBB(),ToTensor(),Crop3D(crop_size)]))
dataset_val = CustomDataset3D(images_path, df_val ,device,
                            label_col='Site_3', file_name_col = 'Subject', 
                            transform=transforms.Compose([ToFloatUKBB(),ToTensor(),Crop3D(crop_size)]))
dataset_test = CustomDataset3D(cf_image_path, df_test ,device,
                             label_col='Site_3', file_name_col = 'Subject',
                             transform=transforms.Compose([ToFloatUKBB(),ToTensor(),Crop3D(crop_size)]))

train_loader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(dataset_val, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)

model = SFCNModelScannerMONAI().to(device)
opt = torch.optim.Adam(model.parameters(), lr)

BCELoss_fn = nn.CrossEntropyLoss()
best_metric= 9999
best_metric_epoch = -1
# Testing
print()
print("Testing")
model.load_state_dict(torch.load(output_path + f'/epoch_best_model.pt'))
model.eval()


with torch.no_grad():
    correct = 0
    total = 0
    pbar4 = tqdm(train_loader)
    for data in pbar4:
        test_X, test_Y = data[0].to(device) , data[1].to(device)
        test_Y_one_hot = F.one_hot(test_Y.to(torch.int64), num_classes=10)

        # Make a prediction
        test_X = test_X[:,None,:,:,:]
        pred = model(test_X)
        real_pred = pred.argmax(dim=1)


        # measure the accuracy
        total += test_Y.size(0)
        correct += torch.eq(real_pred.int(),test_Y.int()).sum().item()

    print('Accuracy of the network on the training images: %d %%' % (100 * correct / total))

    correct = 0
    total = 0
    pbar5 = tqdm(val_loader)
    for data in pbar5:
        test_X, test_Y = data[0].to(device) , data[1].to(device)
        test_Y_one_hot = F.one_hot(test_Y.to(torch.int64), num_classes=10)

        # Make a prediction
        test_X = test_X[:,None,:,:,:]
        pred = model(test_X)
        real_pred = pred.argmax(dim=1)

        # measure the accuracy
        total += test_Y.size(0)
        correct += torch.eq(real_pred.int(),test_Y.int()).sum().item()

    print('Accuracy of the network on the Validation images: %d %%' % (100 * correct / total))

    correct = 0
    total = 0
    pbar3 = tqdm(test_loader)
    df_predict = pd.DataFrame(columns=['Subject','Site_original','Prediction'])
    for data in pbar3:
        test_X, test_Y, subject_name = data[0].to(device) , data[1].to(device), data[2]
        test_Y_one_hot = F.one_hot(test_Y.to(torch.int64), num_classes=10)

        # Make a prediction
        test_X = test_X[:,None,:,:,:]
        pred = model(test_X)
        real_pred = pred.argmax(dim=1)

        # measure the accuracy
        total += test_Y.size(0)
        correct += torch.eq(real_pred.int(),test_Y.int()).sum().item()

        #saving in the dataframe
        for idx, prediction in enumerate(real_pred.cpu().int()):
            df_predict.loc[len(df_predict)] = {
                                                "Subject": subject_name[idx],
                                                "Site_original": test_Y.cpu().int()[idx].item(),
                                                "Prediction": prediction.item()
                                             }

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    df_predict.to_csv(output_path + f"/prediction_scanner_site_{one_site}.csv", index=False)





