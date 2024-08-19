import torch
import torch.nn.functional as F
import torchnet as tnt
import os, json
import pickle as pkl
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score



# ====================== TRAIN AND EVAL ITERATOR

def train_epoch(model, optimizer, criterion, data_loader, device, display_step):
    nrmse_meter = tnt.meter.MSEMeter(root=True)
    loss_meter = tnt.meter.AverageValueMeter()
    y_true = []
    y_pred = []

    for i, (x, y) in enumerate(data_loader):
        
        y_true.extend(list(map(float, y)))
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        nrmse_meter.add(out, y)
        loss_meter.add(loss.item())

        pred = out.detach().cpu().numpy()
        y_pred.extend(list(pred))


        if (i + 1) % display_step == 0:
            print('Step [{}/{}], Loss: {:.4f}, NRMSE : {:.2f}'.format(
                i + 1, len(data_loader), loss_meter.value()[0], 
                nrmse_meter.value()/np.mean(y_pred)))

    epoch_metrics = {'train_loss': loss_meter.value()[0],
                     'train_nrmse': nrmse_meter.value()/np.mean((y_pred)),
                     'train_R2': r2_score(np.array(y_true), np.array(y_pred)),
                     'train_r': pearsonr(np.array(y_true), np.array(y_pred))[0]}

    return epoch_metrics



def evaluation(model, criterion, loader, device, mode='val'):
    y_true = []
    y_pred = []

    nrmse_meter = tnt.meter.MSEMeter(root=True)
    loss_meter = tnt.meter.AverageValueMeter()

    for (x, y) in loader:

        y_true.extend(list(map(float, y)))
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            prediction = model(x)
            loss = criterion(prediction, y)

        nrmse_meter.add(prediction, y)
        loss_meter.add(loss.item())

        pred = prediction.cpu().numpy()
        y_pred.extend(list(pred))

    metrics = {'{}_nrmse'.format(mode): nrmse_meter.value()/np.mean(y_pred),
               '{}_loss'.format(mode): loss_meter.value()[0],
               '{}_R2'.format(mode): r2_score(np.array(y_true), np.array(y_pred)), 
               '{}_r'.format(mode): pearsonr(np.array(y_true), np.array(y_pred))[0]
               }

    if mode == 'val':
        return metrics
    elif mode == 'test':
        return metrics, y_true, y_pred 



# ================ HELPER FUNCTIONS I/O
    
# create output directory
def prepare_output(dir):
    os.makedirs(dir, exist_ok=True)


# store training and validation performance
def checkpoint(log, dir):
    with open(os.path.join(dir, 'trainlog.json'), 'w') as outfile:
        json.dump(log, outfile, indent=4)



# stores predicted and true values with their years
def save_results(metrics, dir, y_true, y_pred, years):

    with open(os.path.join(dir,'test_metrics.json'), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)
        
    # save y_true, y_pred and years
    pkl.dump(y_true, open(os.path.join(dir, 'y_true_test_data.pkl'), 'wb'))
    pkl.dump(y_pred, open(os.path.join(dir,  'y_pred_test_data.pkl'), 'wb'))
    pkl.dump(years, open(os.path.join(dir,  'years_test_data.pkl'), 'wb'))

    

# =====================