import torch
import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset



class YieldDataset(Dataset):
    def __init__(self, predictor_path, yield_path, norm=None, years=None, 
                 feature_selector=None, temporal_truncation=None, 
                 proportion=100, state_selector=None):
        """
        Initialize the dataset 

        Parameters:
            - predictor_path (str): Path to the predictors DataFrame.
            - yield_path (str): Path to the yield DataFrame.
            - norm (dict): Dictionary of mean and std per channel and state
            - year_range (list): A list of (start_year, end_year) to filter data by year.
            - feature_selector (list): List of features to select from the DataFrame.
            - temporal_truncation (list): start and end index of the time series
            - proportion (int): percentage of time steps to subset
            - state_selector (list): List of states to filter data by.
            - norm (dict): Dictionary of mean and std of features and timesteps
                           grouped by state

        Returns:
            - tensor object of features, yield
        """

        self.predictors_df = pd.read_csv(predictor_path)
        self.yield_df = pd.read_csv(yield_path)
        self.norm = norm

        # merge predictors and yield
        self.df = pd.merge(self.yield_df, self.predictors_df, on=['adm_id', 'harvest_year'], how='inner')

        # apply filtering and processing
        self.df = self._apply_filters(self.df, years, state_selector)
   
        # get admin ids and years
        self.ids = self.df['adm_id'].to_numpy()
        self.state_ids = np.array([x[:5] for x in self.ids])
        self.years = self.df['harvest_year'].to_numpy()
        self.target = self.df['yield'].to_numpy()     

        # ====================== FEATURE SELECTION START ==============================
        
        combined_features = []
        seq_feature_prefixes = ["tavg", "prec", 'tmax', 'tmin', "fpar", 
                                 "ndvi", "ssm", "rsm", "cwb", "et0", "rad"]
        static_features = ["awc", "bulk_density"] +['drainage_class_'+str(i) for i in range(0,7)] + \
                          ['lat', 'lon', 'yield_-1', 'yield_-2', 'yield_-3', 'yield_-4', 'yield_-5']
        
        # use all features is no feature selection
        if feature_selector is None:
            feature_selector = seq_feature_prefixes + static_features

        for feature in feature_selector:
            if feature in seq_feature_prefixes:
                filtered_df = self.df[[col for col in self.df.columns if col.startswith(feature)]].to_numpy()
                self.max_timesteps = filtered_df.shape[1]

            elif feature in static_features:
                filtered_df = self.df[[col for col in self.df.columns if col.startswith(feature)]]
                filtered_df = np.repeat(filtered_df.to_numpy(), self.max_timesteps, axis=1)

            combined_features.append(filtered_df)
        # ====================== FEATURE SELECTION END ==============================


        # reconstruct array as samples x time x channels.
        combined_features = np.stack(combined_features, axis=-1)

        # normalize
        norm_data, self.norm_values = self._apply_normalization(combined_features, self.state_ids, self.norm)

        # truncate time series
        self.truncated_data = self._truncate_temporal(norm_data, temporal_truncation, proportion)


    def _apply_filters(self, df, years, state_selector):

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
        
        return torch.tensor(predictor_item, dtype=torch.float32), \
            torch.tensor(yield_item, dtype=torch.float32)
        





# ### =================== TESTING BLOCKS - REMOVE DELETE AFTER USE================

# x_df_path = "/app/dev/Seasonal_Climate/onedrive/cy_bench_16daybins_wheat_US_v6.csv"
# y_df_path = "/app/dev/Seasonal_Climate/cybench/cybench-data/wheat/US/yield_wheat_US.csv"


# # Initialize YieldDataset with various parameters
# dataset = YieldDataset(
#     predictor_path=x_df_path,
#     yield_path=y_df_path,
#     norm = None,
#     years=list(range(2002, 2018)),
#     feature_selector=None,
#     temporal_truncation=[3, 24], #[0,10]
#     proportion=100,
#     state_selector=['US-08']

# # Print dataset length
# print(f"Dataset length: {len(dataset)}")

# # Retrieve and print a sample
# for i in range(len(dataset)):
#     predictor, yield_data = dataset[i]
#     print(f"Sample {i}:")
#     print(f"  Predictor: {predictor.shape}")
#     print(f"  Yield: {yield_data.cpu().shape}")
#     break
