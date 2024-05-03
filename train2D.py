import torch
import pickle
import pandas as pd
import numpy as np
from dataset import CustomDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms 
from customTransforms import ToFloatUKBB
from models import SFCNModel
from tqdm.auto import tqdm
import torch.nn as nn

images_path = '/work/forkert_lab/erik/mitacs_dataset'
dataframe_path = '/home/erik.ohara/SFCN_PD_scanner'
output_path = '/home/erik.ohara/SFCN_PD_scanner/models'

batch_size = 4
epochs = 100
lr = 0.0001
training = False

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print(device)

# Loading traning, validation and test image paths
image_path_train = images_path + '/train_site'
image_path_val = images_path + '/val_site'
image_path_test = images_path + '/test_site'

# Loading dataframes (patient info, scanner, site)
df_old = pd.read_csv(dataframe_path + '/new_train_df.csv',low_memory=False)
df_train = pd.DataFrame(columns=df_old.columns)
df_train = pd.read_csv(dataframe_path + '/new_train_df.csv',low_memory=False)
df_val = pd.read_csv(dataframe_path + '/new_val_df.csv',low_memory=False)
df_test = pd.read_csv(dataframe_path + '/new_test_df.csv',low_memory=False)

dataset_train = CustomDataset(image_path_train, df_train ,device, 
                              label_col='Group_bin', file_name_col = 'Subject', 
                              transform=transforms.Compose([ToFloatUKBB(),ToTensor()]))
dataset_val = CustomDataset(image_path_val, df_val ,device,
                            label_col='Group_bin', file_name_col = 'Subject', 
                            transform=transforms.Compose([ToFloatUKBB(),ToTensor()]))
dataset_test = CustomDataset(image_path_test, df_test ,device,
                             label_col='Group_bin', file_name_col = 'Subject',
                             transform=transforms.Compose([ToFloatUKBB(),ToTensor()]))

train_loader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(dataset_val, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)

model = SFCNModel().to(device)
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
            torch.save(model.state_dict(), dataframe_path + f'/models/epoch_best_model.pt')

    print()
    print("End of Training")
    print(f"best metric epoch: {best_metric_epoch}")
    print(f"best metric (MSE): {best_metric}")

    # Saving the model
    torch.save(model.state_dict(), dataframe_path + '/models/end_model.pt')

# Testing
print()
print("Testing")
model.load_state_dict(torch.load(dataframe_path + f'/models/end_model.pt'))
model.eval()

BCE_losses_test = []
correct = 0
total = 0

with torch.no_grad():
    pbar3 = tqdm(test_loader)
    for data in pbar3:
        test_X, test_Y = data[0].to(device) , data[1].to(device)
        counter = 1 - test_Y
        test_Y = torch.stack((counter, test_Y),dim=1)

        # Make a prediction
        pred = model(test_X)
        real_pred = pred.argmax(dim=1)

        # Calculate the loss
        BCEloss = BCELoss_fn(pred, test_Y)
        BCE_losses_test.append(BCEloss.item())

        # measure the accuracy
        total += test_Y.size(0)
        correct += torch.eq(real_pred.int(),test_Y[:,1].int()).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))





