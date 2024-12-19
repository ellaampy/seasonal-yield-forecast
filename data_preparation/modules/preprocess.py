import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.stats import gamma, norm
from shapely.geometry import Point
from scipy.spatial.distance import cdist

static_features = ["awc", "bulk_density", "drainage_class_1", "drainage_class_2", "drainage_class_3", "drainage_class_4", "drainage_class_5", "drainage_class_6"]
temporal_prefixes = ["tavg", "prec", "tmin", "tmax", "ndvi", "fpar", "rad", "et0", "cwb", "ssm", "rsm"]

day_of_year_to_time_step_ecmwf = {
    1: 0, 9: 1, 17: 2, 25: 3, 33: 4, 41: 5, 49: 6, 57: 7, 65: 8, 73: 9, 81: 10, 89: 11, 90:11,
    97: 12, 98:12, 105: 13, 106:13, 113: 14, 114:14, 121: 15, 122:15, 129: 16, 130:16, 137: 17, 138:17, 145: 18, 146:18, 153: 19, 154:19, 161: 20, 162:20, 
    169: 21, 170:21,  177: 22, 178:22, 185: 23, 186:23, 193: 24, 194:24, 201: 25, 202:25, 209: 26, 210:26,
    217: 27, 218:27, 225: 28, 226:28, 233: 29, 234: 29, 
    241: 30, 249: 31, 257: 32, 265: 33, 273: 34, 281: 35, 289: 36, 297: 37,  305: 38, 
    313: 39, 321: 40, 329: 41, 337: 42, 345: 43, 353: 44, 358:45, 359: 45
}

day_of_year_to_time_step_era = {
    1: 0, 9: 1, 17: 2, 25: 3, 33: 4, 41: 5, 49: 6, 57: 7, 65: 8, 73: 9, 81: 10, 89: 11, 
    97: 12, 105: 13, 113: 14, 121: 15, 129: 16, 137: 17, 145: 18, 153: 19, 161: 20, 
    169: 21, 177: 22, 185: 23, 193: 24, 201: 25, 209: 26, 217: 27, 225: 28, 233: 29, 
    241: 30, 249: 31, 257: 32, 265: 33, 273: 34, 281: 35, 289: 36, 297: 37,  305: 38, 
    313: 39, 321: 40, 329: 41, 337: 42, 345: 43, 353: 44, 361: 45
}

       
fpar_doy_to_time_step = {
    1: 0,  11: 2,  21: 3,  32: 4,  42: 6,  52: 7,  60: 8,  61: 8,  70: 9,  71: 9,  
    80: 10,  81: 10,  91: 12, 92: 12, 101: 13, 102: 13, 111: 14, 112: 14, 121: 16, 122: 16, 
    131: 17, 132: 17, 141: 18, 142: 18, 152: 19, 153: 19, 162: 21, 163: 21, 172: 22, 
    173: 22, 182: 23, 183: 23, 192: 24, 193: 24, 202: 26, 203: 26, 213: 27, 214: 27, 
    223: 28, 224: 28, 233: 29, 234: 29, 244: 31, 245: 31, 254: 32, 255: 32, 264: 33, 
    265: 33, 274: 35, 275: 35, 284: 36, 285: 36, 294: 37, 295: 37, 305: 38, 306: 38, 
    315: 40, 316: 40, 325: 41, 326: 41, 335: 42, 336: 42, 345:43, 346: 43, 355: 45, 356: 45
}

# days refers to start days fo year of first and last 8-day bin of crop season
country_crop_to_crop_season = {
    "BR": {
        "wheat": {
            "days": (129, 329),
            "time_steps": (16, 41),
            "month": (5, 11),
            "test_years": [2021, 2022, 2023]
        },
        "maize": {
            "days": (1, 201),
            "time_steps": (0, 25),
            "month": (1, 7),
            "test_years": [2021, 2022, 2023] 
        },
        "shapefile_path": "../data/shapefiles/BR/bra_admbnda_adm2_ibge_2020.shp"
    },
    "US": {
        "wheat": {
            "days": (1, 209),
            "time_steps": (27, 26),
            "offset": 19,
            "month": (1, 7),
            "test_years": [2021, 2022, 2023]
        },
        "maize": {
            "days": (97, 297),
            "time_steps": (12, 37),
            "month": (4, 10),
            "test_years": [2021, 2022, 2023] 
        },
        "shapefile_path": "../data/shapefiles/US/tl_2023_us_county.shp"
    }
}


def get_study_metadata(country, crop):
    """
    Returns the file paths, crop season and test years for a given country and crop.

    Parameters:
        country (str): The country code (e.g., "BR" for Brazil, "US" for United States).
        crop (str): The crop type (e.g., "wheat", "maize").
    Returns:
        tuple: A tuple containing the shapefile path, the crop season start and end as days of year and months and test years.
    Raises:
        ValueError: If an invalid country or crop is provided.
    """    
    shapefile_path = country_crop_to_crop_season[country]["shapefile_path"]
    crop_season_in_days_of_year = country_crop_to_crop_season[country][crop]["days"]
    crop_season_in_months = country_crop_to_crop_season[country][crop]["month"]
    offset = country_crop_to_crop_season[country][crop]["offset"]
    test_years = country_crop_to_crop_season[country][crop]["test_years"]
    
    return (shapefile_path, crop_season_in_days_of_year, crop_season_in_months, offset, test_years)


def resample_era(era):
    """
    Resamples the ERA dataset by grouping it based on administrative ID and year, and then resampling the data
    to an 8-day frequency.

    Parameters:
        era (DataFrame): The ERA dataset to be resampled.
    Returns:
        DataFrame: The resampled ERA dataset with additional columns 'start_date_bin' and 'time_step'.
    """
    era_resampled = (era
                    .groupby(["adm_id", "year"]).resample("8D", on="date")[["tmin", "tmax", "prec", "tavg"]].mean().reset_index()
                    .rename(columns={"date":"start_date_bin"})
                    )
    era_resampled = era_resampled.assign(time_step=era_resampled["start_date_bin"].apply(lambda x: day_of_year_to_time_step_era[x.day_of_year]))
    
    return era_resampled

def transform_to_time_step(val, df, id_vars=['adm_id', 'year']):
    df_transformed = df.melt(id_vars=id_vars, value_vars=[col for col in df.columns if col.startswith('{}_'.format(val))], var_name='time_step', value_name=val)
    df_transformed['time_step'] = df_transformed['time_step'].str.split('_').str[1].astype(int)
    return df_transformed

def interpolate_fpar_timesteps(df, reference_quantity):
    """ 
    interpolate fpar values for all timesteps of a reference quantity.
    
    Params:
        df: pd.DataFrame, dataframe containing the fpar columns
        reference_quantity: str, reference quantity with complete  timesteps
    Returns:
        df: pd.DataFrame, dataframe with interpolated fpar values
    
    """
    min_time_step = min([int(c.split("_")[1]) for c in [l for l in df.columns if reference_quantity in l]])
    max_time_step = max([int(c.split("_")[1]) for c in [l for l in df.columns if reference_quantity in l]])
    fpar_columns = [c for c in df.columns if "fpar" in c]
    fpar_columns_all_timesteps = ["fpar_{}".format(n) for n in list(range(min_time_step, max_time_step + 1))]
    new_fpar_columns = list(set(fpar_columns_all_timesteps).difference(set(fpar_columns)))
    df[new_fpar_columns] = np.nan
    df[fpar_columns_all_timesteps] = df[fpar_columns_all_timesteps].interpolate(method='linear', axis=1, limit_direction='both')
    df = df.reindex(sorted(df.columns), axis=1) 

    return df

def temporal_aggregation_from8day_to_window(df, feature_prefix_list, window_size):
    """ 
    Aggregate steps of temporal features and return the resulting dataframe.
    
    Params:
        df: pd.DataFrame, dataframe containing the features and time steps as column names
        feature_prefix_list: list, list of prefixes of temporal features that should be aggregated
        window_size: int, number of steps to aggregate by 
    """ 
    
    df  = df.set_index(static_features + ["adm_id", "harvest_year"], append=False)
    li = []
    for feature in feature_prefix_list:
        res = df[get_temporal_feature_subset(df, [feature])].rolling(window=window_size, min_periods=1, axis=1).mean()
        res = res.iloc[:, 1::window_size] 
        li.append(res) 
   
    df = pd.concat(li, axis=1).reset_index()
    return df 

def get_temporal_feature_subset(df, feature_prefix_list):
    """ 
    Filter temporal features by prefix and return the resulting list of features names.
    
    Params:
        df: pd.DataFrame, dataframe containing the features as column names
        feature_prefix_list: list, list of prefixes to filter by
    Returns:
        filtered_temporal_features: list, list of filtered temporal feature names
    """
    all_columns = df.columns
    filtered_temporal_features  = [f for f in all_columns if any([f.startswith(prefix) for prefix in feature_prefix_list])]
    return filtered_temporal_features 

def order_temporal_features(df, temporal_prefixes):
    candidate_columns = [c for c in df.columns if c.split("_")[0] in temporal_prefixes]
    formated_candidate_columns = format_column_names(candidate_columns)
    df = df.rename(columns=dict(zip(candidate_columns, formated_candidate_columns)))
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.set_index(["harvest_year"], append=True).reset_index(level=[1])
    return df

def format_column_names(column_names):
    formatted_columns = [] 
    for name in column_names:
        prefix, number = name.split('_')
        formatted_name = f"{prefix}_{int(number):02d}"
        formatted_columns.append(formatted_name) 
    return formatted_columns


def pivot_ecmwf(ecmwf, adjust_time_step=True):
    """
    Pivots the ECMWF dataframe.
    
    Parameters:
        ecmwf (DataFrame): ECMWF dataframe
    Returns:
        ecmwf_pivot (DataFrame): Pivoted ECMWF dataframe
    """
    if adjust_time_step:
        offset = 19
    else:
        offset = 0
    ecmwf = ecmwf.assign(time_step = (ecmwf["time_step"] + offset).replace(64, 18),
                         init_time_step = ecmwf["time_step"].min() + offset, 
                         year = ecmwf["init_date"].dt.year)
    ecmwf = ecmwf.pivot(index=["adm_id", "number", "init_date", "init_time_step", "year"], 
                        columns="time_step", values=["tavg", "tmin", "tmax", "prec"])
    ecmwf.columns = ["_".join([str(col) for col in c]).strip() for c in ecmwf.columns]
    ecmwf = ecmwf.reset_index()
    
    return ecmwf  

def pivot_era(era):
    """
    Pivots the ERA dataframe.
    
    Parameters:
        era (DataFrame): ERA dataframe
    Returns:
        era_pivot (DataFrame): Pivoted ERA dataframe
    """
    era_pivot = era.pivot(index=["adm_id", "harvest_year"], columns="time_step", values=["tavg", "tmin", "tmax", "prec"]).reset_index()
    era_pivot.columns = ["_".join([str(col) for col in c]).strip("_") for c in era_pivot.columns]
    era_pivot = era_pivot.reset_index(drop=True)
    
    return era_pivot


def pivot_predictors(predictors):
    """
    Pivots the predictors dataframe by time step.

    Parameters:
        predictors (DataFrame): The input dataframe containing the predictors.
    Returns:
        DataFrame: The pivoted dataframe with predictors.
    """
    value_columns = predictors.columns.difference(["adm_id", "harvest_year", "calendar_year", "date", "time_step"]).tolist()
    predictors_pivot = predictors.dropna(subset="time_step").pivot(index=["adm_id", "harvest_year", "calendar_year"], columns="time_step", values=value_columns)
    predictors_pivot.columns = ["_".join([str(col) for col in c]).strip() for c in predictors_pivot.columns]
    predictors_pivot = predictors_pivot.reset_index()
    
    return predictors_pivot


def temporal_aggregation_ecmwf(df, feature_prefix_list, static_features, window_size):
    """ 
    Aggregate steps of temporal features and return the resulting dataframe.
    
    Parameters:
        df: pd.DataFrame, dataframe containing the features and time steps as column names
        feature_prefix_list: list, list of prefixes of temporal features that should be aggregated
        window_size: int, number of steps to aggregate by 
    Returns:
        df: pd.DataFrame, dataframe with aggregated temporal features
    """ 
    df  = df.set_index(static_features, append=False)
    li = []
    for feature in feature_prefix_list:
        res = df[get_temporal_feature_subset(df, [feature])].rolling(window=window_size, min_periods=1, axis=1).mean()
        res = res.iloc[:, 1::2] 
        li.append(res) 
   
    df = pd.concat(li, axis=1).reset_index()
    return df 


def get_temporal_feature_subset(df, feature_prefix_list):
    """ 
    Filter temporal features by prefix and return the resulting list of features names.
    
    Parameters:
        df: pd.DataFrame, dataframe containing the features as column names
        feature_prefix_list: list, list of prefixes to filter by
    Returns:
        filtered_temporal_features: list, list of filtered temporal feature names
    """
    all_columns = df.columns
    filtered_temporal_features  = [f for f in all_columns if any([f.startswith(prefix) for prefix in feature_prefix_list])]
    return filtered_temporal_features


def filter_predictors_by_adm_ids(predictor_list, adm_ids):
    """
    Filters the predictors by the provided adm_ids.

    Parameters:
        predictor_list (list of DataFrames): The predictor DataFrames.
        adm_ids (list of str): The list of adm_ids to filter.
    Returns:
        list of pd.DataFrames: The list of filtered predictor DataFrames.
    """
    return [predictors.loc[predictors["adm_id"].isin(adm_ids)].reset_index(drop=True) for predictors in predictor_list]

    
def assign_time_steps(predictors, crop, country, crop_season_in_months):
    """
    Assigns time step columns to the predictors DataFrame.

    Parameters:
        predictors (DataFrame): The predictors DataFrame.
    Returns:
        DataFrame: The predictors DataFrame with time step columns.
    """  
    #unique_fpar_ts = list(set(predictors["date"].dt.day_of_year.unique().tolist()))
    #unique_fpar_ts.sort()
    #print([c for c in unique_fpar_ts if c not in day_of_year_to_time_step.keys()])
    if set(predictors["date"].dt.day_of_year.unique().tolist()).issubset(fpar_doy_to_time_step.keys()):
        predictors = predictors.assign(time_step=predictors["date"].dt.day_of_year.map(fpar_doy_to_time_step).astype('Int64'))
    
    else:
        #doy_list = [c for c in predictors["date"].dt.day_of_year.unique().tolist() if c not in day_of_year_to_time_step.keys()]
        #doy_list.sort()
        #print(doy_list)
        predictors = predictors.assign(time_step=predictors["date"].dt.day_of_year.map(day_of_year_to_time_step_era).astype('Int64'))
    
    if (crop == "wheat") & (country == "US") & (crop_season_in_months[0] >= crop_season_in_months[1]):
        predictors.loc[predictors["date"].dt.month >= 8, "time_step"] = predictors.loc[predictors["date"].dt.month >= 8, "time_step"] - 27
        predictors.loc[predictors["date"].dt.month <= 7, "time_step"] = predictors.loc[predictors["date"].dt.month <= 7, "time_step"] + 19
    
    return predictors


def assign_time_columns(predictors):
    """
    Assigns time columns to the predictors DataFrame.

    Parameters:
        predictors (DataFrame): The predictors DataFrame.
    Returns:
        DataFrame: The predictors DataFrame with date, day of year and year columns.
    """
    predictors = predictors.assign(date=pd.to_datetime(predictors["date"], format="%Y%m%d"), 
                                   calendar_year=pd.to_datetime(predictors["date"], format="%Y%m%d").dt.year)
    
    return predictors


def filter_predictors_by_crop_season(predictors, crop_season_in_days_of_year):
    """
    Filters the predictors by the crop season start and end.

    Parameters:
        predictors (DataFrame): The predictors DataFrame.
        crop_season_start (int): The day of year of the first bin of the crop season start month.
        crop_season_end (int): The day of year of the last bin of the crop season end month.
    Returns:
        DataFrame: The filtered predictors DataFrame.
    """
    crop_season_start = crop_season_in_days_of_year[0]
    crop_season_end = crop_season_in_days_of_year[1]
    
    predictors = predictors.assign(doy=predictors["date"].dt.day_of_year)
    predictors = predictors.loc[(predictors["doy"].between(crop_season_start, crop_season_end)) & (predictors["calendar_year"].between(2004, 2023))].reset_index(drop=True)
    
    return predictors


def temporal_aggregation_8day(predictors):
    """
    Resamples the predictors to 8-day bins.

    Parameters:
        predictors (DataFrame): The predictors DataFrame.
    Returns:
        DataFrame: The resampled predictors DataFrame.
    """
    # test if predictors are already in 8- (ndvi) or 10- (fpar) day bins
    if 365 > predictors["date"].dt.day_of_year.nunique():
        return predictors
    value_columns = predictors.columns.difference(["adm_id", "date", "calendar_year"]).tolist()
    predictors = predictors.groupby(["adm_id", "calendar_year"]).resample("8D", on="date")[value_columns].mean().reset_index()
    
    return predictors

def assign_harvest_year(predictors, country, crop, crop_season_in_months):
    """
    Assigns the harvest year to the predictors DataFrame.
    
    Parameters:
        predictors (DataFrame): The predictors DataFrame.
        country (str): The country code (e.g., "BR" for Brazil, "US" for United States).
        crop (str): The crop type (e.g., "wheat", "maize").
        crop_season_in_months (tuple): The start and end of the crop season as months.
    Returns:
        predictors (DataFrame): The predictors DataFrame with the harvest year column.
    """
    predictors = predictors.assign(harvest_year=predictors["calendar_year"])
    if (crop == "wheat") & (country == "US") & (crop_season_in_months[0] >= crop_season_in_months[1]):
        predictors.loc[predictors["date"].dt.month >= crop_season_in_months[0], "harvest_year"] = predictors.loc[predictors["date"].dt.month >= crop_season_in_months[0], "calendar_year"] + 1
        
    return predictors
    
def combine_rows(group):
    return group.iloc[0].combine_first(group.iloc[1])

def preprocess_temporal_data(data_list, crop_season_in_months, crop_season_in_days_of_year, crop, country):
    """
    Preprocesses the temporal predictor datasets.
    
    Parameters:
        data_list (list): A list of pandas DataFrames containing the temporal predictor datasets.
        crop_season_in_months (tuple): Crop season in months
        crop_season_in_days_of_year (tuple): Crop season in days of year
        crop: The crop type (e.g., "wheat", "maize").
        country: The country code (e.g., "BR" for Brazil, "US" for United States).
    Returns:
        list: A list of preprocessed pandas DataFrames.
    """
    processed_data = []
    for df in data_list:
        df = df.drop(columns=["crop_name"])
        df = assign_time_columns(df)
        if "rsm" in df.columns:
            df = df.loc[df["date"] >= pd.to_datetime("20030202" , format="%Y%m%d")].reset_index(drop=True)
        df = temporal_aggregation_8day(df)
        df = assign_harvest_year(df, country, crop, crop_season_in_months)
        df = assign_time_steps(df, crop, country, crop_season_in_months)
        df = pivot_predictors(df)
        if (crop == "wheat") & (country == "US") & (crop_season_in_months[0] >= crop_season_in_months[1]):
            df = df.loc[df["harvest_year"].between(2004, 2023)].groupby(['adm_id', 'harvest_year']).apply(combine_rows).drop(["adm_id", "harvest_year", "calendar_year"], axis=1).interpolate(axis=1, method='linear', limit_direction='both').reset_index()
        processed_data.append(df)
        
    return processed_data


def format_ecmwf_columns(ecmwf):
    """
    Formats the columns of the ECMWF dataframe.
    
    Parameters:
        ecmwf (pd.DataFrame): ECMWF dataframe
    Returns:
        ecmwf (pd.DataFrame): ECMWF dataframe with formatted columns
    """
    ecmwf = ecmwf.rename(columns={"t2m":"tavg", "tp":"prec", "mx2t24":"tmax", "mn2t24":"tmin", "time":"init_date"}).drop(columns=["surface", "step"])
    ecmwf = ecmwf.assign(valid_time=pd.to_datetime(ecmwf["valid_time"]), init_date=pd.to_datetime(ecmwf["init_date"]), 
                        doy=pd.to_datetime(ecmwf["valid_time"]).dt.day_of_year, year=pd.to_datetime(ecmwf["init_date"]).dt.year,
                        location=ecmwf["latitude"].astype(int).astype(str) + ", " + ecmwf["longitude"].astype(int).astype(str),
                        tavg=ecmwf["tavg"] - 273.15, tmax=ecmwf["tmax"] - 273.15, tmin=ecmwf["tmin"] - 273.15,
                        prec=ecmwf.groupby(["init_date", "latitude", "longitude"])["prec"].transform(lambda x: x.diff().fillna(x).clip(lower=0) * 1000))
    
    return ecmwf


def resample_ecmwf(ecmwf, crop_season_in_doy):
    """
    Resamples the ECMWF data to 8-day bins.
    
    Parameters:
        ecmwf (DataFrame): The ECMWF data to be resampled.
        crop_season_in_doy (tuple): The start and end of the crop season as day of year.    
    Returns:
        ecmwf_resampled (DataFrame): The resampled ECMWF data.
    """
    start_doy, end_doy = calculate_start_and_end_doy(ecmwf, crop_season_in_doy)
    doy_filter = ecmwf["doy"].between(start_doy, end_doy+7)
    if ecmwf["init_date"].dt.month.unique()[0] == 1:
        start_doy = 357
        doy_filter = (ecmwf["doy"] >= start_doy) | (ecmwf["doy"] < end_doy+7) 
    li = []
    for year in ecmwf["year"].unique():
        ecmwf_year = ecmwf[(ecmwf["year"] == year) 
                           & (doy_filter)].reset_index(drop=True)
        ecmwf_year_resampled = (ecmwf_year
                                .groupby(["init_date", "year", "location", "number"])
                                .resample("8D", on="valid_time")[["tavg", "tmax", "tmin", "prec"]].mean().reset_index()
                                .rename(columns={"valid_time":"start_date_bin"})
                                )
        li.append(ecmwf_year_resampled)
    ecmwf_resampled = pd.concat(li, ignore_index=True)
    
    return ecmwf_resampled


def create_end_of_season(era_resampled, crop_season_in_months):
    """
    Creates the end of season dataframe using all ERA data.
    """
    era_end_of_season = pivot_era(era_resampled)
    end_of_season_date = era_end_of_season["harvest_year"].astype(str) + "-" + str(crop_season_in_months[1]+1) + "-01"
    era_end_of_season = era_end_of_season.assign(init_date=pd.to_datetime(end_of_season_date))
    
    return era_end_of_season


def calculate_start_and_end_doy(ecmwf, crop_season_in_days_of_year):
    """
    Calculates the start and end day of year for the ECMWF data.
    
    Parameters:
        ecmwf (pd.DataFrame): ECMWF dataframe
        crop_season_in_days_of_year (tuple): Crop season in days of year
    Returns:
        start_doy (int): Bin start as day of year
        end_doy (int): Bin end as day of year
    """
    first_doy_available_in_all_years = ecmwf.loc[(ecmwf["year"] == ecmwf["valid_time"].dt.year)].groupby("year")["doy"].min().min()
    start_doy = np.intersect1d(list(range(first_doy_available_in_all_years, ecmwf.doy.unique().max())), list(range(crop_season_in_days_of_year[0], crop_season_in_days_of_year[1]+1, 8))).min()
    end_doy = crop_season_in_days_of_year[1]
    
    return start_doy, end_doy


def create_adm_unit_level_forecast_dataframe(list_of_index_values, index_names, adm_units_shapefile, ecmwf):
    """
    Creates a DataFrame that can store forecasts on adm_id level for each time step for a corresponding ECWMF grid cell.
    
    Parameters:
        list_of_index_values (list of lists/arrays): The lists of values for the Cartesian product.
        index_names (list of str): The names of the indices.
        adm_units_shapefile (geoDataFrame): A geodataframe containing the polygons of the administrative units.
    Returns:
        DataFrame: A DataFrame that can store forecasts on adm_id level for each time step for a corresponding ECWMF grid cell.
    """
    multi_index = pd.MultiIndex.from_product(list_of_index_values, names=index_names)
    df = pd.DataFrame(index=multi_index).reset_index()
    if df["init_date"].dt.month.unique()[0] == 1:
        df = df.loc[df["start_date_bin"].between(df["init_date"] - pd.Timedelta(days=8), 
                                                               df["init_date"]+pd.Timedelta(days=216))].reset_index(drop=True)
    else: 
        df = df.loc[(df["init_date"].dt.year == df["start_date_bin"].dt.year)].reset_index(drop=True)
    
    df = df.merge(adm_units_shapefile[["adm_id", "geometry"]], on="adm_id", how="left").set_index(index_names)
    df = gpd.GeoDataFrame(df[["geometry"]], geometry="geometry").reset_index()

    adm_units_shapes_to_ecmwf_grids = assign_adm_units_to_ecmwf_grid_cells(ecmwf, adm_units_shapefile)
    df = df.merge(adm_units_shapes_to_ecmwf_grids.drop(["index_right", "geometry"], axis=1), on="adm_id", how="left")
    
    return df


def get_era_for_n_nearest_neighbors(df, adm_id, n, distance_df):
    """
    Return ERA data for n nearest neighbors for a given adm_id.
    
    Parameters:
        df (DataFrame): The ERA data.
        adm_id (str): The administrative unit ID.
        n (int): The number of nearest neighbors.
        distance_df (DataFrame): A DataFrame containing the pairwise distances between the centroids of the administrative units.
    Returns:
        DataFrame: The ERA data for n nearest neighbors.
    """
    nearest_neighbours = distance_df.loc[adm_id, :].sort_values().index.tolist()[:n]
    df = df.loc[(df["year"].between(2004, 2017)) 
                & (df["adm_id"].isin(nearest_neighbours))].reset_index(drop=True)
    
    return df


def calculate_pairwise_distances(country_gpd_crop):
    """
    Calculate pairwise distances between the centroids of the administrative units.
    
    Parameters:
        country_gpd_crop (geoDataFrame): A geodataframe containing the polygons of the administrative units.
    Returns:
        distance_df (DataFrame): A DataFrame containing the pairwise distances between the centroids of the administrative units.
    """
    country_gpd_crop['centroid'] = country_gpd_crop.geometry.to_crs(epsg=32723).centroid
    centroids = country_gpd_crop['centroid'].apply(lambda geom: (geom.x, geom.y)).tolist()
    distance_matrix = cdist(centroids, centroids, metric='euclidean')
    distance_df = pd.DataFrame(distance_matrix, index=country_gpd_crop['adm_id'], columns=country_gpd_crop['adm_id'])
    
    return distance_df

def assign_adm_units_to_ecmwf_grid_cells(ecmwf, adm_units_shapefile):
    """ 
    Assign adm_ids to ECMWF grid cells based on the grid cell coordinates and county polygons.   

    Parameters:
        ecmwf (DataFrame): The ECMWF data.
        adm_units_shapefile (geoDataFrame): A geodataframe containing the polygons of the administrative units.
    Returns:
        counties_with_ecmwf_data (DataFrame): The counties with ECMWF data.
    """
    # reproject county polygons from geographic to planar coordinates
    shapefile_planar = adm_units_shapefile[["adm_id", "geometry"]].to_crs(epsg=32723)
    
    ecmwf_unique_lat_lon = ecmwf["location"].str.split(", ", expand=True).rename(columns={0: "latitude", 1: "longitude"}).drop_duplicates().reset_index(drop=True)
    ecmwf_unique_lat_lon_geo_df = gpd.GeoDataFrame(ecmwf_unique_lat_lon, geometry=gpd.points_from_xy(ecmwf_unique_lat_lon["longitude"], ecmwf_unique_lat_lon["latitude"]), crs="EPSG:4326").to_crs(epsg=32723)
    county_shapes_to_ecmwf_grids = gpd.sjoin_nearest(shapefile_planar, ecmwf_unique_lat_lon_geo_df, how="left")
    county_shapes_to_ecmwf_grids["location"] = county_shapes_to_ecmwf_grids["latitude"].astype(str) + ", " + county_shapes_to_ecmwf_grids["longitude"].astype(str)
   
    return county_shapes_to_ecmwf_grids
    

def merge_ecmwf_to_adm_units(ecmwf, adm_level_forecasts):
    """ 
    Merge the ECMWF data to the administrative unit level.   

    Parameters:
        ecmwf (DataFrame): The ECMWF data.
        adm_level_forecasts (DataFrame): The administrative unit level.
        init_month (int): The month when the ECWMF forecast was initialized.  
    Returns:
        adm_level_forecasts (DataFrame): The administrative unit level with ECMWF data.
    """
    ecmwf_assigned_to_adm_units = (
        adm_level_forecasts
        .merge(ecmwf, on=["init_date", "number", "start_date_bin", "location"], how="left")
        .groupby(["init_date", "number", "start_date_bin", "adm_id"], as_index=False)[["tavg", "tmax", "tmin", "prec"]].mean()
        .assign(time_step=lambda df: df["start_date_bin"].apply(lambda x: day_of_year_to_time_step_ecmwf[x.day_of_year]))
    )
        
    return ecmwf_assigned_to_adm_units


def assign_ecmwf_forecasts_to_adm_units(ecmwf, adm_units_shapefile):
    """ 
    Assign ECMWF grid cell forecasts to adm units based on the grid cell coordinates and admin units polygons.   

    Parameters:
        ecmwf (DataFrame): The ECMWF data.
        adm_units_shapefile (geoDataFrame): A geodataframe containing the polygons of the administrative units.
        init_month (int): The month when the ECWMF forecast was initialized.
    Returns:
        counties_with_ecmwf_data (DataFrame): The counties with ECMWF data.
        first_time_step (int): The first time step of the forecast.
    """
    adm_level_forecasts = create_adm_unit_level_forecast_dataframe(
        [ecmwf["init_date"].unique(), ecmwf["number"].unique(), ecmwf["start_date_bin"].unique(), adm_units_shapefile["adm_id"].unique()], 
        ["init_date", "number", "start_date_bin", "adm_id"], adm_units_shapefile, ecmwf)

    ecmwf_assigned_to_adm_units = merge_ecmwf_to_adm_units(ecmwf, adm_level_forecasts)
    
    return ecmwf_assigned_to_adm_units


def normal_correction(obs_data, mod_data, sce_data, cdf_threshold=0.9999999):
    """
    Apply quantile mapping to correct the SCM data to match the observed data distribution.
    
    Parameters:
        obs_data: np.array or pd.Series, observed reference data
        mod_data: np.array or pd.Series, model reference data
        sce_data: np.array or pd.Series, SCM data to be adjusted
        cdf_threshold: float, threshold for CDF values
    Returns:
        sce_adjusted: np.array, adjusted SCM data
    """
    obs_norm, mod_norm = [norm.fit(x) for x in [obs_data, mod_data]]
    obs_cdf = norm.cdf(np.sort(obs_data), *obs_norm)
    mod_cdf = norm.cdf(np.sort(mod_data), *mod_norm)
    obs_cdf = np.maximum(np.minimum(obs_cdf, cdf_threshold), 1 - cdf_threshold)
    mod_cdf = np.maximum(np.minimum(mod_cdf, cdf_threshold), 1 - cdf_threshold)
    mod_cdf_intpol = np.interp(sce_data, np.sort(mod_data), mod_cdf)
    sce_adjusted = norm.ppf(mod_cdf_intpol, *obs_norm)
    
    return sce_adjusted
