import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset



class YieldDataset(Dataset):
    def __init__(self, predictor_path, yield_path, years=None, 
                 feature_selector=None, max_timesteps=26, temporal_truncation=None, 
                 proportion=100, state_selector=None, aez_selector=None):
        """
        Initialize the dataset 

        Parameters:
            - predictor_path (str): Path to the predictors DataFrame.
            - yield_path (str): Path to the yield DataFrame.
            - year_range (list): A list of (start_year, end_year) to filter data by year.
            - feature_selector (list): List of features to select from the DataFrame.
            - max_timesteps (int): maximum number of sequences
            - temporal_truncation (list): start and end index of the time series
            - proportion (int): percentage of time steps to subset
            - state_selector (list): List of states to filter data by.
            - aez_selector (list): List of AEZs to filter data by.

        Returns:
            - tensor object of features, yield
        """

        self.predictors_df = pd.read_csv(predictor_path)
        self.yield_df = pd.read_csv(yield_path)

        # merge predictors and yield
        self.df = pd.merge(self.yield_df, self.predictors_df, on=['adm_id', 'harvest_year'], how='inner')

        # apply filtering and processing
        self.df = self._apply_filters(self.df, years, state_selector, aez_selector)
   
        # get admin ids and years
        self.ids = self.df['adm_id'].to_numpy()
        self.years = self.df['harvest_year'].to_numpy()
        self.target = self.df['yield'].to_numpy()     

        # ====================== FEATURE SELECTION START ==============================
        
        self.combined_features = []
        seq_feature_prefixes = ["tavg", "prec", 'tmax', 'tmin', "fpar", \
                                "ndvi", "ssm", "rsm", "cwb", "et0", "rad"]
        static_features = ["harvest_year", "awc", "bulk_density"] +['drainage_class_'+str(i) for i in range(1,7)]

        
        for feature in feature_selector:
            if feature in seq_feature_prefixes:
                filtered_df = self.df[[col for col in self.df.columns if col.startswith(feature)]].to_numpy()
                ## todo ensure that filtered columns are in chronologcal order

            elif feature in static_features:
                filtered_df = self.df[[col for col in self.df.columns if col.startswith(feature)]]
                filtered_df = np.repeat(filtered_df.to_numpy(), max_timesteps, axis=1)

            self.combined_features.append(filtered_df)

        # ====================== FEATURE SELECTION END ==============================


        # reconstruct array as samples x time x channels.
        self.combined_features = np.stack(self.combined_features, axis=-1)
        print('combined features', self.combined_features.shape)

        # truncate time series
        self.combined_features = self._truncate_temporal(self.combined_features, \
                                                         temporal_truncation, proportion)
        print('combined features', self.combined_features.shape)

        print('data sizes', self.combined_features.shape, self.target.shape, self.ids.shape, self.years.shape)


    def _apply_filters(self, df, years, state_selector, aez_selector):

        # year filtering
        if years is not None:
            df = df[df ['harvest_year'].isin(years)]

        # state filtering
        if state_selector is not None:
            df = df[df['adm_id'].apply(lambda x: any(x.startswith(prefix) \
                                                     for prefix in state_selector))]

        return df


    def _truncate_temporal(self, data, temporal_truncation, percentage):
        
        assert len(data.shape) > 2

        if temporal_truncation is not None:
            data = data[:,temporal_truncation[0]:temporal_truncation[1], :]
        
        if percentage is not None and percentage <100 :
            percentage = int(data.shape[1]* (percentage/100))

        return data[:, :percentage, :]



    def __len__(self):
        return len(self.combined_features)
    

    def __getitem__(self, idx):

        predictor_item = self.combined_features[idx]
        yield_item = self.target[idx]
        
        return torch.tensor(predictor_item, dtype=torch.float32), \
            torch.tensor(yield_item, dtype=torch.float32)
        




# ### =================== TESTING BLOCKS - REMOVE DELETE AFTER USE================

# x_df_path ='/app/dev/Seasonal_Climate/onedrive/ndvi_soil_soil_moisture_meteo_fpar_maize_BR.csv'
# y_df_path = "/app/dev/Seasonal_Climate/cybench/cybench-data/maize/BR/yield_maize_BR.csv"


# # Initialize YieldDataset with various parameters
# dataset = YieldDataset(
#     predictor_path=x_df_path,
#     yield_path=y_df_path,
#     years=[2020],
#     feature_selector=['ssm', 'rsm'],
#     temporal_truncation=None, #[0,10]
#     proportion=100,
#     state_selector=['BR11'],
#     aez_selector=None
# )

# # Print dataset length
# print(f"Dataset length: {len(dataset)}")

# # Retrieve and print a sample
# for i in range(len(dataset)):
#     predictor, yield_data = dataset[i]
#     print(f"Sample {i}:")
#     print(f"  Predictor: {predictor}")
#     print(f"  Yield: {yield_data}")


