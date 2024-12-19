from torch import dropout, nn
from torch.nn import MSELoss
import torch.nn.functional as F
import torchnet as tnt
import pandas as pd
import torch
import os
import torch.optim as optim
from models.utils import *
from models.lstm import LSTM
from data_preparation.dataset import YieldDataset
from data_preparation.dataset_scm import YieldDataset_SCM
import hydra
from omegaconf import DictConfig, OmegaConf
import random
from torch.utils.tensorboard import SummaryWriter

@hydra.main(config_path="conf", config_name="config", version_base=None)

def main(cfg: DictConfig):
    output_path_probabilistic = os.path.join(cfg.dataset.results_path, 'scm 4 steps/ensembles/')
    os.chdir("C:/Users/Max Zachow/OneDrive - TUM/Dokumente/GitHub/seasonal-yield-forecast/")
    
    # print(OmegaConf.to_yaml(cfg))

    # create results dir and save config
    os.makedirs(cfg.dataset.results_path, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.dataset.results_path, 'config.yaml'))

    # set seed for reproducability
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    device = torch.device(cfg.training.device)
    
    
    for i in range(0, 25):
        train_dataset = YieldDataset_SCM(
        predictor_path= cfg.dataset.predictor_path,
        yield_path= cfg.dataset.yield_path,
        norm = None,
        years= cfg.dataset.train_years,
        feature_selector=cfg.dataset.feature_selector,
        temporal_truncation=cfg.dataset.temporal_truncation, 
        proportion=100,
        state_selector=cfg.dataset.state_selector,
        scm_folder=cfg.dataset.scm_folder,
        simulation_num=i, 
        init_month=cfg.dataset.init_month,
        scm_bin=cfg.dataset.scm_bin, 
        bias_adjusted=cfg.dataset.bias_adjustment,
        scm_truncation=None, 
        scm_features=cfg.dataset.scm_features,
        return_type='zero_filled')
        
        test_dataset = YieldDataset_SCM(
            predictor_path= cfg.dataset.predictor_path,
            yield_path= cfg.dataset.yield_path,
            norm = train_dataset.norm_values,
            years= cfg.dataset.test_years,
            feature_selector=cfg.dataset.feature_selector,
            temporal_truncation=cfg.dataset.temporal_truncation, 
            proportion=100,
            state_selector=cfg.dataset.state_selector,
            scm_folder=cfg.dataset.scm_folder,
            simulation_num=i, 
            init_month=cfg.dataset.init_month,
            scm_bin=cfg.dataset.scm_bin, 
            bias_adjusted=cfg.dataset.bias_adjustment,
            scm_truncation=None, 
            scm_features=cfg.dataset.scm_features,
            return_type='zero_filled')
    
        test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=cfg.training.num_workers, \
                                                batch_size=cfg.training.batch_size, shuffle=False )
     
        # get year for each sample in test_loader
        prediction_years = test_dataset.years 

     
        model =  LSTM(input_dim=cfg.model.input_dim, num_classes=cfg.model.num_classes, 
                  hidden_dims=cfg.model.hidden_dims,num_layers=cfg.model.num_layers, 
                  dropout=cfg.model.dropout, bidirectional=cfg.model.bidirectional, 
                  use_layernorm=cfg.model.use_layernorm)
        
        experiment_name =  "timesteps_{}_{}_{}_{}_{}_{}".format(cfg.dataset.temporal_truncation[0], 
                                                             cfg.dataset.temporal_truncation[1],
                                                             cfg.dataset.init_month,
                                                             cfg.dataset.bias_adjustment,
                                                             "_".join(cfg.dataset.scm_features),
                                                             i)
        print(experiment_name)
        model = model.to(cfg.training.device)
        #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        #print(train_dataset.truncated_data.shape)

        criterion = MSELoss()
## =================== ACTIVATE FOR EVALUATION ON TEST =============
        # load best model
        model.load_state_dict(torch.load(os.path.join(output_path_probabilistic, '{}_model.pth.tar'.format(cfg.dataset.init_month)))['state_dict'])

        # # evaluate on test data
        model.eval()
        test_metrics, y_true, y_pred = evaluation(model, criterion, test_loader, device=device, mode='test')
        best_df = pd.DataFrame({"y_true":y_true, "y_pred":y_pred, "adm_id":test_loader.dataset.ids, 
                                    "year":test_loader.dataset.df.harvest_year, "member": i})
        print('========== Test Metrics ===========')
        print('Ensemble member: ', i)
        print('Loss {:.4f},  NRMSE {:.4f}, R2 {:.4f}'.format(test_metrics['test_loss'], 
                                                            test_metrics['test_nrmse'], 
                                                            test_metrics['test_R2']))
        print('========== Test Metrics ===========')
        #save_results(test_metrics, output_path_probabilistic, y_true, y_pred, prediction_years)
        best_df.to_csv(os.path.join(output_path_probabilistic, 'test_y_true_y_pred_{}.csv'.format(experiment_name)))



if __name__ == "__main__":
        main()