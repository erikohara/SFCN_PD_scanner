import torch
import pickle
import pandas as pd
import numpy as np
from dataset import CustomDataset
from torch.utils.data import DataLoader
from models import SFCNModel

reshaped_path = '/work/forkert_lab/erik/MACAW/reshaped/mitacs'
pca_path = '/work/forkert_lab/erik/PCA/mitacs'
dataframe_path = '/home/erik.ohara/SFCN_PD_scanner'
output_path = '/home/erik.ohara/SFCN_PD_scanner/models'

ncomps = 1500
batch_size = 4

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print(device)

# Loading traning, validation and test set
data_train = np.load(reshaped_path + '/reshaped_mitacs_train.npy')
data_val = np.load(reshaped_path + '/reshaped_mitacs_val.npy')
data_test = np.load(reshaped_path + '/reshaped_mitacs_test.npy')
print("Data loaded")

# Loading PCAs
with open(pca_path + '/evecs_slice_train.pkl','rb') as f:  
    evecs_train = pickle.load(f)
with open(pca_path + '/evecs_slice_val.pkl','rb') as f:  
    evecs_val = pickle.load(f)
with open(pca_path + '/evecs_slice_test.pkl','rb') as f:  
    evecs_test = pickle.load(f)
print("PCA loaded")

def encode(data, evecs):
    return np.matmul(data,evecs.T)

def decode(data,evecs):
    return np.matmul(data,evecs)

# Loading dataframes (patient info, scanner, site)
df_train = pd.read_csv(dataframe_path + '/new_train_df.csv',low_memory=False)
df_val = pd.read_csv(dataframe_path + '/new_val_df.csv',low_memory=False)
df_test = pd.read_csv(dataframe_path + '/new_test_df.csv',low_memory=False)


encoded_train = encode(data_train,evecs_train[:ncomps])
encoded_val = encode(df_val,evecs_val[:ncomps])
encoded_test = encode(df_test,evecs_test[:ncomps])

dataset_train = CustomDataset(encoded_train.astype(np.float32), device)
dataset_val = CustomDataset(encoded_val.astype(np.float32), device)
dataset_test = CustomDataset(encoded_test.astype(np.float32), device)

train_loader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(dataset_val, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)

model = SFCNModel()
opt = torch.optim.Adam(model.parameters(), lr)

# define loss
best_metric_epoch = -1