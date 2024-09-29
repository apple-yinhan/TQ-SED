import config
import os
from my_dataset import LASS_SED
import torch.optim as optim
import torch
from tqdm import tqdm
import sys
import h5py
import numpy as np
from torch.utils.data import DataLoader
import random

print('============== exp{}:{} ================'.format(config.exp_id, config.folder_idx))
print('Backbone:', config.backbone)
device = config.device
    
out_folder = './experiments/exp{}/{}'.format(config.exp_id, config.folder_idx)
os.makedirs(out_folder, exist_ok=True)

# LOAD DATASET
print ('===== Loading dataset =====')
train_audio_path = f'{config.preprocessed_data_folder}/{config.folder_idx}/train/mel'
train_sep_path = f'{config.preprocessed_data_folder}/{config.folder_idx}/train/sep_mel'
train_label_path = f'{config.preprocessed_data_folder}/{config.folder_idx}/train/soft'
train_dataset = LASS_SED(train_audio_path, train_label_path, train_sep_path)

val_audio_path = f'{config.preprocessed_data_folder}/{config.folder_idx}/val/mel'
val_sep_path = f'{config.preprocessed_data_folder}/{config.folder_idx}/val/sep_mel'
val_label_path = f'{config.preprocessed_data_folder}/{config.folder_idx}/val/soft'
val_dataset = LASS_SED(val_audio_path, val_label_path, val_sep_path)

train_loader = DataLoader(dataset=train_dataset, batch_size=config.batchsize, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=config.batchsize, shuffle=False)

# LOAD MODEL
print(config.backbone)
if config.lass_sed_type == 'Method_A':
    from models.tq_sed import CRNN_LASS_A
    model = CRNN_LASS_A(classes_num=config.class_nums, cnn_filters=config.cnn_filters, rnn_hid=config.rnn_hid, _dropout_rate=config._dropout_rate)

print(f'Moving model to {device}')
model = model.to(device)
# compute number of parameters
model_params = sum([np.prod(p.size()) for p in model.parameters()])
print ('\nTotal paramters: ' + str(model_params))

# finetune
if config.finetune:
    print('FINETUNE ...')
    freeze_list = ['linear2']#,'conformer1','conformer2'
    for param in model.named_parameters():
        for layers in freeze_list:
            if layers not in param[0]:
                param[1].requires_grad = False
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=config.lr)
else:
    optimizer = optim.Adam(params=model.parameters(), lr=config.lr)


# set up optimizer
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
 		patience=10)
mse_loss_fun = torch.nn.MSELoss()
bce_loss_fun = torch.nn.BCELoss()

model_path = out_folder + '/{}_{}.pth'.format(config.backbone, config.lass_sed_type)
dict_path = out_folder + '/{}_{}.h5'.format(config.backbone, config.lass_sed_type)
best_loss = np.inf

if config.load_model:
    print("Continuing training full model from checkpoint " + model_path)
    model.load_state_dict(torch.load(model_path), strict=False)
    hf = h5py.File(dict_path, mode='r')
    loss_hist_ = np.array(hf.get('val_loss_hist'))
    best_loss = np.min(loss_hist_)
    hf.close()

# TRAIN MODEL
print('TRAINING START')
worse_epoches = 0
total_epoch = 0
epoch = 1
print('Best_loss:', best_loss)
train_loss_hist = []
val_loss_hist = []
lr_hist = []
#while total_epoch < 60:
while worse_epoches < config.patience:
    # # training
    print("Training epoch " + str(epoch))
    model.train()
    train_loss = 0.
    total_iteration = len(train_loader)
    example_num = 0
    with tqdm(total=total_iteration) as pbar:
        for data in train_loader:
            soft = data[1]          # [Batch, seq_len, 17]
            sep = data[3]           # [Batch, n_class, seq_len, n_mel]

            soft = soft.float().to(device)
            sep = sep.float().to(device)

            soft = soft[:, :, :11]  # [Batch, seq_len, 11]

            outputs = model(sep)
            loss = mse_loss_fun(outputs, soft)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            # Update
            optimizer.step()
            
            train_loss += (1. / float(example_num + 1)) * (loss - train_loss)
            pbar.set_description("train_loss: {0:.6f}".format(train_loss))
            
            pbar.update(1)
            example_num += 1
    
    train_loss_hist.append(train_loss.detach().cpu().numpy())
    lr_hist.append(optimizer.param_groups[0]['lr'])
    # # validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.
        total_iteration = len(val_loader)
        example_num = 0
        with tqdm(total=total_iteration) as pbar:
            for data in val_loader:

                soft = data[1]          # [Batch, seq_len, 17]
                sep = data[3]           # [Batch, n_class, seq_len, n_mel]

                soft = soft.float().to(device)
                sep = sep.float().to(device)

                soft = soft[:, :, :11]  # [Batch, seq_len, 11]
                 
                outputs = model(sep)
                loss = mse_loss_fun(outputs, soft)
            
                val_loss += (1. / float(example_num + 1)) * (loss - val_loss)
                pbar.set_description("val_loss: {0:.6f}".format(val_loss))
                
                pbar.update(1)
                example_num += 1
    
    val_loss = val_loss.detach().cpu().numpy()
    val_loss_hist.append(val_loss)
    # print('val_loss : ', val_loss)
    # # compare and save
    if val_loss < best_loss:
        print("MODEL IMPROVED ON VALIDATION SET!")
        worse_epoches = 0
        best_loss = val_loss
        torch.save(model.state_dict(), model_path)
        # Save Results
        results = {'train_loss_hist': train_loss_hist,
                   'val_loss_hist': val_loss_hist,
                   'lr_hist': lr_hist,
                   }
        hf = h5py.File(dict_path, mode='w')
        for i in results:
            hf.create_dataset(i, data=results[i])
        hf.close()
    else:
        worse_epoches += 1
    
    scheduler.step(val_loss)
    epoch += 1
    total_epoch += 1
    
