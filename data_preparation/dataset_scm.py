import torch
import pandas as pd
import numpy as np
import json, os
from torch.utils.data import Dataset



class YieldDataset_SCM(Dataset):
    def __init__(self, predictor_path, yield_path, norm=None, years=None, 
                 feature_selector=None, max_timesteps=46, temporal_truncation=None, 
                 proportion=100, state_selector=None, aez_selector=None,
                 scm_folder=None, simulation_num=None, init_month='july',
                 scm_bin=8, bias_adjusted=True, scm_truncation=None, return_type='zero_filled'):
        """
        Initialize the dataset 

        Parameters:
            - predictor_path (str): Path to the predictors DataFrame.
            - yield_path (str): Path to the yield DataFrame.
            - norm (dict): Dictionary of mean and std per channel and state
            - year_range (list): A list of (start_year, end_year) to filter data by year.
            - feature_selector (list): List of features to select from the DataFrame.
            - max_timesteps (int): maximum number of sequences
            - temporal_truncation (list): start and end index of the time series
            - proportion (int): percentage of time steps to subset
            - state_selector (list): List of states to filter data by.
            - aez_selector (list): List of AEZs to filter data by.
            - norm (dict): Dictionary of mean and std of features and timesteps
                           grouped by state
            - scm_path(str): path to csv containing SCM data
            - scm_folder(str): which simulation data to use. if None, an average
                                   of all simulations is used
            - init_month(str): initialization month of scm forecast e.g. july
            - scm_bin(int): days of temporal aggregation e.g 8 or 16
            - zero_filled(str): specify how the input observed and augmented data
                                is merged.

        Returns:
            - tensor object of features, yield
        """

        self.predictors_df = pd.read_csv(predictor_path)
        self.yield_df = pd.read_csv(yield_path)
        self.norm = norm
        self.return_type = return_type
        self.temporal_truncation = temporal_truncation

        ## todo 
        ## 1. add bias adjustment. file name structure has changed
        ## 2. implement scm truncation


        # read scm data
        self.scm_df = pd.read_csv(os.path.join(scm_folder, 'ecmwf_wheat_US_{}_{}daybins.csv'.format(init_month, scm_bin)))
        scm_features = ['tavg', 'tmin', 'tmax', 'prec']
        self.scm_timesteps = sum(self.scm_df.columns.str.startswith(scm_features[0]))

        # filter by simulation number
        if simulation_num is not None:
            self.scm_df = self.scm_df[self.scm_df['number']==simulation_num]

        else:
            # average all simulations by adm_id and year
            mean_columns = [col for col in self.scm_df.columns if 
                            col.startswith(tuple(scm_features))]

            # take first values of these cols
            first_columns = ['init_date', 'init_time_step', 'number']

            # perform the groupby operation
            self.scm_df = self.scm_df.groupby(['adm_id', 'harvest_year']).agg(
                {**{col: 'mean' for col in mean_columns}, 
                **{col: 'first' for col in first_columns}}).reset_index()
            
        self.scm_df.drop(columns=['init_date', 'init_time_step', 'number'], inplace=True)


        # merge predictor/scm and yield
        self.df = pd.merge(self.yield_df, self.predictors_df, on=['adm_id', 'harvest_year'], how='inner')
        self.df = pd.merge(self.df, self.scm_df, on=['adm_id', 'harvest_year'], how='inner', suffixes=('', '_new'))

        # replace predictor using scm
        replace_cols = self.scm_df.columns
        replace_cols = [x for x in replace_cols if not x in ['adm_id',  'harvest_year']]
        for col in replace_cols:
            self.df[col] = self.df[f'{col}_new']

        # drop the '_new' columns after replacement
        self.df.drop(columns=[f'{col}_new' for col in replace_cols], inplace=True)   

        # apply filtering and processing
        self.df = self._apply_filters(self.df, years, state_selector, aez_selector)
   
        # get admin ids and years
        self.ids = self.df['adm_id'].to_numpy()
        self.state_ids = np.array([x[:5] for x in self.ids])
        self.years = self.df['harvest_year'].to_numpy()
        self.target = self.df['yield'].to_numpy()     

    #     # ====================== FEATURE SELECTION START ==============================
        
        combined_features = []
        seq_feature_prefixes = ["tavg", "prec", 'tmax', 'tmin', "fpar", 
                                 "ndvi", "ssm", "rsm", "cwb", "et0", "rad"]
        static_features = ["awc", "bulk_density"] +['drainage_class_'+str(i) for i in range(0,7)] + \
                          ['lat', 'lon', 'yield_-1', 'yield_-2', 'yield_-3']
        

        # use all features is no feature selection
        if feature_selector is None:
            feature_selector = seq_feature_prefixes + static_features

        for feature in feature_selector:
            if feature in seq_feature_prefixes:
                filtered_df = self.df[[col for col in self.df.columns if col.startswith(feature)]].to_numpy()
                ## todo ensure that filtered columns are in chronologcal order

            elif feature in static_features:
                filtered_df = self.df[[col for col in self.df.columns if col.startswith(feature)]]
                filtered_df = np.repeat(filtered_df.to_numpy(), max_timesteps, axis=1)
            combined_features.append(filtered_df)

        # get index for scm variables
        self.scm_idx = [feature_selector.index(elem) for elem in scm_features if elem in feature_selector]
        # ====================== FEATURE SELECTION END ==============================

        # reconstruct array as samples x time x channels.
        combined_features = np.stack(combined_features, axis=-1)

        # normalize
        self.norm_data, self.norm_values = self._apply_normalization(combined_features, 
                                                                    self.state_ids, 
                                                                    self.norm)
        # truncate time series
        self.truncated_data = self._truncate_temporal(self.norm_data,
                                                      self.temporal_truncation, 
                                                      proportion)

    def _apply_filters(self, df, years, state_selector, aez_selector):

        # year filtering
        if years is not None:
            df = df[df ['harvest_year'].isin(years)]

        # state filtering
        if state_selector is not None:
            df = df[df['adm_id'].apply(lambda x: any(x.startswith(prefix) \
                                                     for prefix in state_selector))]
        return df

    
    def _apply_normalization(self, data, group_array, norm):

        unique_groups = np.unique(group_array)
        normalized_data = np.copy(data)

        norm_values ={}
        for group in unique_groups:
            group_indices = np.where(group_array == group)
            group_data = data[group_indices]

            if norm is not None:
                normalized_group_data = (group_data - norm[group][0]) / norm[group][1]
                normalized_data[group_indices] = normalized_group_data

            else:
                mean = np.mean(group_data, axis=(0,1), keepdims=True)
                std = np.std(group_data, axis=(0,1), keepdims=True)
                std = std + 1e-8
                norm_values[group] = [mean, std]
                normalized_group_data = (group_data - mean) / std
                normalized_data[group_indices] = normalized_group_data
    
        if norm is not None:
            return normalized_data, None
        else:
            return normalized_data, norm_values


    def _truncate_temporal(self, data, temporal_truncation, percentage):
        
        assert len(data.shape) > 2

        if temporal_truncation is not None:
            data = data[:,temporal_truncation[0]:temporal_truncation[1], :]
        
        if percentage is not None and percentage <100 :
            percentage = int(data.shape[1]* (percentage/100))

        return data[:, :percentage, :]



    def __len__(self):
        return len(self.truncated_data)
    


    def __getitem__(self, idx):

        predictor_item = self.truncated_data[idx]
        yield_item = self.target[idx]

        # returns separate data for truncated observed and scm
        if self.return_type != 'zero_filled':
            scm_data = self.norm_data[idx][-self.scm_timesteps:, self.scm_idx]
            return (torch.tensor(predictor_item, dtype=torch.float32), 
                    torch.tensor(scm_data, dtype=torch.float32)), \
                torch.tensor(yield_item, dtype=torch.float32)


        # returns observed data filled with scm and zeros after truncate
        else:
            merged_data = self.norm_data[idx]
            for i in range(self.truncated_data.shape[2]):
                if i not in self.scm_idx:
                    merged_data[self.temporal_truncation[1]:, i] = 0

            return torch.tensor(merged_data, dtype=torch.float32), \
                torch.tensor(yield_item, dtype=torch.float32)
        





# ### =================== TESTING BLOCKS - REMOVE DELETE AFTER USE================

x_df_path = '/app/dev/Seasonal_Climate/onedrive/cy_bench_8daybins_wheat_US_v2.csv'
y_df_path = "/app/dev/Seasonal_Climate/cybench/cybench-data/wheat/US/yield_wheat_US.csv"
scm_folder = "/app/dev/Seasonal_Climate/onedrive"


# # Initialize YieldDataset with various parameters
# dataset = YieldDataset_SCM(
#     predictor_path=x_df_path,
#     yield_path=y_df_path,
#     norm = None,
#     years=list(range(2004, 2018)),
#     feature_selector=['awc', 'tavg', 'tmin'],
#     max_timesteps = 46,
#     temporal_truncation=[3, 24], #[0,10]
#     proportion=100,
#     state_selector=['US-08'],
#     aez_selector=None,
#     scm_folder=scm_folder, 
#     simulation_num=None, 
#     init_month='july',
#     scm_bin=8, 
#     bias_adjusted=True,
#     scm_truncation=None, 
#     return_type='other')  #'zero_filled'



# # Print dataset length
# print(f"Dataset length: {len(dataset)}")

# # Retrieve and print a sample
# for i in range(len(dataset)):
#     predictor, yield_data = dataset[i]
#     print(f"Sample {i}:")
#     print(len(predictor))
#     print(predictor[0].shape, predictor[1].shape)
#     print(f"  Predictor: {predictor.shape}")
#     print(f"  Yield: {yield_data.cpu().shape}")
#     break
