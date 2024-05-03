import torch
import pandas as pd
from dataset import CustomDataset3D
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms 
from customTransforms import ToFloatUKBB, Crop3D
from models import SFCNModelMONAI
from tqdm.auto import tqdm
import torch.nn as nn
import argparse

cf_image_path = '/work/forkert_lab/erik/MACAW/cf_images/mitacs_site_aside'
dataframe_path = '/home/erik.ohara/SFCN_PD_scanner'
output_path = '/home/erik.ohara/SFCN_PD_scanner/site_aside_hamonized'
parser = argparse.ArgumentParser()
parser.add_argument('site_aside')
args = parser.parse_args()
site_aside = int(args.site_aside)

batch_size = 4
crop_size = (100,150,150)
epochs = 100
lr = 0.0001

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print(device)

# Loading traning, validation and test image paths

# Loading dataframes (patient info, scanner, site)
df_all = pd.read_csv(dataframe_path + '/all_df.csv',low_memory=False)
df_test = df_all[df_all['Site_3'] == site_aside].copy().reset_index()
study_aside = df_test['Study'].unique()[0]
print(study_aside)

df_test = pd.read_csv(output_path +f"/{study_aside}/df_test.csv",low_memory=False)  

dataset_test = CustomDataset3D(cf_image_path, df_test ,device,
                             label_col='Group_bin',label_col_2='Site_3', file_name_col = 'Subject',
                             transform=transforms.Compose([ToFloatUKBB(),ToTensor(),Crop3D(crop_size)]))

test_loader = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)

model = SFCNModelMONAI().to(device)
opt = torch.optim.Adam(model.parameters(), lr)

BCELoss_fn = nn.BCELoss()

print("Testing on cf images")
model.load_state_dict(torch.load(output_path +f"/{study_aside}/epoch_best_model.pt"))
model.eval()


with torch.no_grad():
    correct = 0
    total = 0
    print(type(test_loader))
    print(len(test_loader))
    pbar3 = tqdm(test_loader)
    print(type(pbar3))
    for data in pbar3:
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

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))





