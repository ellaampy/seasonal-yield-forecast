import json, os, glob
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import optuna
import torch
import pickle as pkl
from torch.nn import MSELoss
from models.vanillann import MLP
from models.utils import *
from models.lstm import LSTM
from data_preparation.dataset import YieldDataset



config = {
    'results_path': '/app/dev/Seasonal_Climate/results/tuning',
    'predictor_path': '/app/dev/Seasonal_Climate/onedrive/cy_bench_16daybins_wheat_US_v2.csv',
    'yield_path': '/app/dev/Seasonal_Climate/cybench/cybench-data/wheat/US/yield_wheat_US.csv',
    'feature_selector': None,  
    'max_timesteps': 23,
    'temporal_truncation': [3, 24],  
    'proportion': 100,
    'state_selector': ['US-08', 'US-20', 'US-31', 'US-40', 'US-46', 'US-48'],  # ['BR41' 'BR42' 'BR43'] 
    'aez_selector': None,
    'train_years': list(range(2004, 2018)),
    'val_years': [2018, 2019, 2020],
    'test_years': [2021, 2022, 2023],
    'device': 'cuda',
    'display_step': 20,
    'input_dim': 19,
    'num_workers': 4,
    'seed': 3407,
    'optimizer_switch': 'ADAM',
    'epochs':50,
    'num_trials':20,
    'patience':10
}



def main(config, params, trial=None):

    # set seed for reproducability
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    device = torch.device(config['device'])


    os.makedirs(config['results_path'], exist_ok=True)

    # Initialize YieldDataset with various parameters
    train_dataset = YieldDataset(
        predictor_path= config['predictor_path'],
        yield_path= config['yield_path'],
        norm = None,
        years= config['train_years'],
        feature_selector= config['feature_selector'],
        max_timesteps= config['max_timesteps'],
        temporal_truncation= config['temporal_truncation'],
        proportion= config['proportion'],
        state_selector= config['state_selector'],
        aez_selector= config['aez_selector']
    )

    val_dataset = YieldDataset(
         predictor_path= config['predictor_path'],
        yield_path= config['yield_path'],
        norm = train_dataset.norm_values,
        years= config['val_years'],
        feature_selector= config['feature_selector'],
        max_timesteps= config['max_timesteps'],
        temporal_truncation= config['temporal_truncation'],
        proportion= config['proportion'],
        state_selector= config['state_selector'],
        aez_selector= config['aez_selector']
    )


    # create data loader from dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=config['num_workers'],  \
                                               batch_size=params['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=config['num_workers'], \
                                             batch_size=params['batch_size'], shuffle=True )
 

    print('============== STARTING TRIAL {}================'.format(trial.number))
    # print(trial.params)
    print('Train {}, Val {}'.format(len(train_loader), len(val_loader)))


    model = params['model']
    print(model.modelname)
    model = model.to(device)


    # selector for optimizer
    optimizer_switch = {
        
        'SDG':   torch.optim.SGD(model.parameters(), lr = params['lr']),
        'ADAM':  torch.optim.Adam(model.parameters(), lr= params['lr']),  
        'ADAMW': torch.optim.AdamW(model.parameters(), lr= params['lr'])
    }
    
    optimizer = optimizer_switch[config['optimizer_switch']]

    # evaluated on mean squared error. alternatively L1Loss()
    criterion = MSELoss()


    # holder for logging training performance during epochs
    trainlog = {}
    best_loss = np.inf
    no_improvement_count = 0
    

    # log training and val loss 
    t_loss = []
    t_nrmse = []
    t_r2 = []
    v_loss = []
    v_nrmse = []
    v_r2 = []

    try:
        for epoch in range(1, config['epochs'] + 1):

            print('EPOCH {}/{}'.format(epoch, config['epochs']))

            model.train()
            train_metrics = train_epoch(model, optimizer, criterion, train_loader, 
                                        device=config['device'], display_step=config['display_step'])

            print('Validation . . . ')
            model.eval()
            val_metrics = evaluation(model, criterion, val_loader, device=config['device'], mode='val')
            print('Loss {:.4f},  NRMSE {:.4f}, R2 {:.4f}'.format(val_metrics['val_loss'], val_metrics['val_nrmse'], val_metrics['val_R2']))


            # log training and validation loss
            t_loss.append(np.round(train_metrics['train_loss'],3))
            t_nrmse.append(np.round(train_metrics['train_nrmse'],3))
            t_r2.append(np.round(train_metrics['train_R2'],3))
            v_loss.append(np.round(val_metrics['val_loss'],3))
            v_nrmse.append(np.round(val_metrics['val_nrmse'],3))
            v_r2.append(np.round(val_metrics['val_R2'],3))
                
            # Report the validation score to Optuna
            if trial is not None:
                trial.report(val_metrics['val_loss'], epoch)

                # Check for early stopping based on validation score
                if best_loss > val_metrics['val_loss']:
                    best_loss = val_metrics['val_loss']
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            # Prune the trial if the score is not improving
            if no_improvement_count >= config['patience']:
                raise optuna.exceptions.TrialPruned()
                                
    # except optuna.exceptions.TrialPruned:
    except Exception as e:
         print(f"Trial failed due to: {e}")
        #  return float('inf')

    finally:
        trial.set_user_attr('train_loss', t_loss)
        trial.set_user_attr('val_loss', v_loss)
        trial.set_user_attr('train_nrmse', t_nrmse)
        trial.set_user_attr('val_nrmse', v_nrmse)
        trial.set_user_attr('train_R2', t_r2)
        trial.set_user_attr('val_R2', v_r2)

    return val_metrics['val_loss']




## START TUNING

for m in ['LSTM']:
    
    def objective(trial):

        if m == 'MLP':
            params = {
                'hidden_dim1': trial.suggest_categorical('hidden_dim1', [256, 512]),
                'hidden_dim2': trial.suggest_categorical('hidden_dim2', [32, 64, 128]),
                'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
                'dropout': trial.suggest_float('dropout', 0, 1),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
                
            }
            params['model'] = MLP(input_dim=config['input_dim'], sequence_len =config['seq_len'], hidden_dim1=params['hidden_dim1'], \
                hidden_dim2 =params['hidden_dim2'], dropout = params['dropout'])
 
            
        elif m == 'LSTM':        
            params = {
                'hidden_dim': trial.suggest_categorical('hidden_dim', [256, 128, 64, 32]),
                'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
                'num_layers': trial.suggest_int('num_layers', 2,8),
                'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
                'dropout': trial.suggest_float('dropout', 0, 1),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
            }
            
            params['model'] = LSTM(input_dim=config['input_dim'], num_classes=1, hidden_dims=params['hidden_dim'], \
                num_layers=params['num_layers'], dropout=params['dropout'], \
                bidirectional=params['bidirectional'], use_layernorm=True)
                

        return main(config, params, trial)
    
    
    study = optuna.create_study(direction='minimize')

    ## suppress loggings
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study.optimize(objective, n_trials=config['num_trials'])  # You can adjust the number of trials

    ## save study
    with open(os.path.join(config['results_path'], "{}_study.pkl".format(m)), "wb") as f:
        pkl.dump(study, f)

    # Get the best hyperparameters and corresponding score
    best_params = study.best_params
    best_score = study.best_value
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Validation Loss: {best_score}")
    
    print('CURRENTLY TUNING ---->')

    print('\n')


