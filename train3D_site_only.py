import torch
import pickle
import pandas as pd
import numpy as np
from dataset import CustomDataset3D
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms 
from customTransforms import ToFloatUKBB, Crop3D
from models import SFCNModelMONAI
from tqdm.auto import tqdm
import torch.nn as nn
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#images_path = '/work/forkert_lab/erik/mitacs_dataset'
images_path = '/work/forkert_lab/mitacs_dataset/affine_using_nifty_reg'
cf_image_path = '/work/forkert_lab/erik/MACAW/cf_images/mitacs_just_site'
dataframe_path = '/home/erik.ohara/SFCN_PD_scanner'
output_path = '/home/erik.ohara/SFCN_PD_scanner/site_aside_hamonized_0_site'
parser = argparse.ArgumentParser()
parser.add_argument('site_aside')
args = parser.parse_args()
site_aside = int(args.site_aside)

batch_size = 4
crop_size = (100,150,150)
epochs = 100
lr = 0.0001
training = True

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print(device)

# Loading traning, validation and test image paths
#image_path_train = images_path + '/train_site'
#image_path_val = images_path + '/val_site'
#image_path_test = images_path + '/test_site'

# Loading dataframes (patient info, scanner, site)
df_all = pd.read_csv(dataframe_path + '/all_df_2.csv',low_memory=False)
df_test = df_all[df_all['Site_3'] == site_aside].copy().reset_index()
study_aside = df_test['Study'].unique()[0]
print(study_aside)

if os.path.exists(output_path +f"/{study_aside}/df_train.csv"):
    df_train = pd.read_csv(output_path +f"/{study_aside}/df_train.csv",low_memory=False)
    df_val = pd.read_csv(output_path +f"/{study_aside}/df_val.csv",low_memory=False)
    df_test = pd.read_csv(output_path +f"/{study_aside}/df_test.csv",low_memory=False)
else:
    df_rest = df_all[df_all['Site_3'] != site_aside].copy()
    df_train, df_val = train_test_split(df_rest, test_size=0.2, random_state=1, stratify=df_rest[['Site_3']]) 
    df_train = df_train.reset_index()
    df_val = df_val.reset_index()
    df_train.to_csv(output_path +f"/{study_aside}/df_train.csv", index=False)  
    df_val.to_csv(output_path +f"/{study_aside}/df_val.csv", index=False)  
    df_test.to_csv(output_path +f"/{study_aside}/df_test.csv", index=False)  


dataset_train = CustomDataset3D(cf_image_path, df_train ,device, 
                              label_col='Group_bin',label_col_2='Site_3', file_name_col = 'Subject', 
                              transform=transforms.Compose([ToFloatUKBB(),ToTensor(),Crop3D(crop_size)]))
dataset_val = CustomDataset3D(cf_image_path, df_val ,device,
                            label_col='Group_bin',label_col_2='Site_3', file_name_col = 'Subject', 
                            transform=transforms.Compose([ToFloatUKBB(),ToTensor(),Crop3D(crop_size)]))
dataset_test = CustomDataset3D(cf_image_path, df_test ,device,
                             label_col='Group_bin',label_col_2='Site_3', file_name_col = 'Subject',
                             transform=transforms.Compose([ToFloatUKBB(),ToTensor(),Crop3D(crop_size)]))

train_loader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(dataset_val, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)

model = SFCNModelMONAI().to(device)
opt = torch.optim.Adam(model.parameters(), lr)

BCELoss_fn = nn.BCELoss()
best_metric= 9999
best_metric_epoch = -1

if training:
    ep_pbar = tqdm(range(epochs))
    print("start training")
    for epoch in ep_pbar:
        model.train()
        train_losses = []
        pbar = tqdm(train_loader)

        for data in pbar:
            train_X, train_Y = data[0].to(device) , data[1].to(device)
            counter = 1 - train_Y
            train_Y = torch.stack((counter, train_Y),dim=1)

            # Zero the gradient
            opt.zero_grad()

            # Make a prediction
            train_X = train_X[:,None,:,:,:]
            pred = model(train_X)
            
            # Calculate the loss and backpropagate
            loss = BCELoss_fn(pred, train_Y)
            loss.backward()

            # Adjust the learning weights
            opt.step()

            # Calculate stats
            train_losses.append(loss.item())
            pbar.set_description(f"######## Training Loss: {loss.item():<.6f} ")
        
        # Validation
        model.eval()

        BCE_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            pbar2 = tqdm(val_loader)
            for data in pbar2:
                # Extract the input and the labels
                val_X, val_Y = data[0].to(device) , data[1].to(device)
                counter = 1 - val_Y
                val_Y = torch.stack((counter, val_Y),dim=1)

                # Make a prediction
                val_X = val_X[:,None,:,:,:]
                pred = model(val_X)
                real_pred = pred.argmax(dim=1)

                # Calculate the loss
                BCEloss = BCELoss_fn(pred, val_Y)
                BCE_losses.append(BCEloss.item())

                # measure the accuracy
                total += val_Y.size(0)
                correct += torch.eq(real_pred.int(),val_Y[:,1].int()).sum().item()

                pbar2.set_description(f"Epoch {epoch+1:<2} BCE Loss: {BCEloss.item():<.4f} Accuracy: {100 * correct / total:<.4f}")
        print(F"Validation: Total = {total} ; Correct = {correct} ; Accuracy = {100 * correct / total:<.4f}")
        if BCEloss.item() < best_metric:
            best_metric = BCEloss.item()
            best_metric_epoch = epoch
            torch.save(model.state_dict(), output_path +f"/{study_aside}/epoch_best_model.pt")

    print()
    print("End of Training")
    print(f"best metric epoch: {best_metric_epoch}")
    print(f"best metric (MSE): {best_metric}")

    # Saving the model
    torch.save(model.state_dict(), output_path + f"/{study_aside}/end_model.pt")

# Testing
print()
print("Testing")
model.load_state_dict(torch.load(output_path +f"/{study_aside}/epoch_best_model.pt"))
model.eval()


with torch.no_grad():
    correct = 0
    total = 0
    pbar4 = tqdm(train_loader)
    for data in pbar4:
        test_X, test_Y = data[0].to(device) , data[1].to(device)
        counter = 1 - test_Y
        test_Y = torch.stack((counter, test_Y),dim=1)

        # Make a prediction
        test_X = test_X[:,None,:,:,:]
        pred = model(test_X)
        real_pred = pred.argmax(dim=1)

        # measure the accuracy
        total += test_Y.size(0)
        correct += torch.eq(real_pred.int(),test_Y[:,1].int()).sum().item()

    print('Accuracy of the network on the training images: %d %%' % (100 * correct / total))

    correct = 0
    total = 0
    pbar5 = tqdm(val_loader)
    df_predict = pd.DataFrame(columns=['Subject','Site_original','Group_bin','Prediction'])
    for data in pbar5:
        test_X, test_Y, subject_name, this_site = data[0].to(device) , data[1].to(device),  data[2], data[3]
        counter = 1 - test_Y
        test_Y = torch.stack((counter, test_Y),dim=1)

        # Make a prediction
        test_X = test_X[:,None,:,:,:]
        pred = model(test_X)
        real_pred = pred.argmax(dim=1)

        # measure the accuracy
        total += test_Y.size(0)
        correct += torch.eq(real_pred.int(),test_Y[:,1].int()).sum().item()

        #saving in the dataframe
        for idx, prediction in enumerate(real_pred.cpu().int()):
            df_predict.loc[len(df_predict)] = {
                                                "Subject": subject_name[idx],
                                                "Site_original": this_site.int()[idx].item(),
                                                "Group_bin": test_Y[:,1].int()[idx].item(),
                                                "Prediction": prediction.item()
                                             }

    print('Accuracy of the network on the Validation images: %d %%' % (100 * correct / total))
    df_predict.to_csv(output_path +f"/{study_aside}/prediction_val.csv", index=False)
    print(df_predict)
    # Getting the accuracy per site
    df_predict['correct_prediction'] = (df_predict['Group_bin'] == df_predict['Prediction']).astype(int)
    accuracy_df = df_predict.groupby('Site_original')['correct_prediction'].mean()
    print(accuracy_df)
    print(type(accuracy_df))
    accuracy_df.to_csv(output_path +f"/{study_aside}/accuracy_persite_val.csv", index=True)

    correct = 0
    total = 0
    print(type(test_loader))
    print(len(test_loader))
    df_predict_test = pd.DataFrame(columns=['Subject','Site_original','Group_bin','Prediction'])
    pbar3 = tqdm(test_loader)
    print(type(pbar3))
    for data in pbar3:
        test_X, test_Y, subject_name, this_site = data[0].to(device), data[1].to(device), data[2], data[3]
        counter = 1 - test_Y
        test_Y = torch.stack((counter, test_Y),dim=1)

        # Make a prediction
        test_X = test_X[:,None,:,:,:]
        pred = model(test_X)
        real_pred = pred.argmax(dim=1)

        # measure the accuracy
        total += test_Y.size(0)
        correct += torch.eq(real_pred.int(),test_Y[:,1].int()).sum().item()

        #saving in the dataframe
        for idx, prediction in enumerate(real_pred.cpu().int()):
            df_predict_test.loc[len(df_predict_test)] = {
                                                "Subject": subject_name[idx],
                                                "Site_original": this_site.int()[idx].item(),
                                                "Group_bin": test_Y[:,1].int()[idx].item(),
                                                "Prediction": prediction.item()
                                             }

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    df_predict_test.to_csv(output_path +f"/{study_aside}/prediction_test.csv", index=False)





