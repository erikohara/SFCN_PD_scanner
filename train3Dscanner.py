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

#images_path = '/work/forkert_lab/erik/mitacs_dataset'
images_path = '/work/forkert_lab/mitacs_dataset/affine_using_nifty_reg'
cf_image_path = '/work/forkert_lab/erik/MACAW/cf_images/mitacs'
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
df_test = pd.read_csv(dataframe_path + '/new_test_df_2.csv',low_memory=False)

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

if training:
    ep_pbar = tqdm(range(epochs))
    print("start training")
    for epoch in ep_pbar:
        model.train()
        train_losses = []
        pbar = tqdm(train_loader)

        for data in pbar:
            train_X, train_Y = data[0].to(device) , data[1].to(device)
            train_Y_one_hot =  F.one_hot(train_Y.to(torch.int64), num_classes=10).float()

            # Zero the gradient
            opt.zero_grad()

            # Make a prediction
            train_X = train_X[:,None,:,:,:]
            pred = model(train_X)
            
            # Calculate the loss and backpropagate
            loss = BCELoss_fn(pred, train_Y_one_hot)
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
                val_Y_one_hot = F.one_hot(val_Y.to(torch.int64), num_classes=10).float()

                # Make a prediction
                val_X = val_X[:,None,:,:,:]
                pred = model(val_X)
                real_pred = pred.argmax(dim=1)

                # Calculate the loss
                BCEloss = BCELoss_fn(pred, val_Y_one_hot)
                BCE_losses.append(BCEloss.item())

                # measure the accuracy
                total += val_Y.size(0)
                correct += torch.eq(real_pred.int(),val_Y.int()).sum().item()

                pbar2.set_description(f"Epoch {epoch+1:<2} BCE Loss: {BCEloss.item():<.4f} Accuracy: {100 * correct / total:<.4f}")
        print(F"Validation: Total = {total} ; Correct = {correct} ; Accuracy = {100 * correct / total:<.4f}")
        if BCEloss.item() < best_metric:
            best_metric = BCEloss.item()
            best_metric_epoch = epoch
            torch.save(model.state_dict(), output_path + f'/epoch_best_model.pt')

    print()
    print("End of Training")
    print(f"best metric epoch: {best_metric_epoch}")
    print(f"best metric (MSE): {best_metric}")

    # Saving the model
    torch.save(model.state_dict(), output_path + '/end_model.pt')

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
    df_predict.to_csv(output_path + '/prediction_scanner.csv', index=False)





