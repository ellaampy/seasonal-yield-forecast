import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.stats import gamma, norm

day_of_year_to_time_step = {
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
            "test_years": [2006, 2015, 2017]
        },
        "maize": {
            "days": (1, 201),
            "time_steps": (0, 25),
            "month": (1, 7),
            "test_years": [2006, 2015, 2017] # TODO: define test years
        },
        "shapefile_path": "../data/shapefiles/BR/bra_admbnda_adm2_ibge_2020.shp"
    },
    "US": {
        "wheat": {
            "days": (9, 233),
            "time_steps": (4, 29),
            "month": (9, 7),
            "test_years": [2015, 2018, 2022]
        },
        "maize": {
            "days": (97, 297),
            "time_steps": (12, 37),
            "month": (4, 10),
            "test_years": [2006, 2015, 2017] # TODO: define test years
        },
        "shapefile_path": "../data/shapefiles/US/tl_2023_us_county/tl_2023_us_county.shp"
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
    test_years = country_crop_to_crop_season[country][crop]["test_years"]

    return (shapefile_path, crop_season_in_days_of_year, crop_season_in_months, test_years)


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
                    .groupby(["adm_id", "harvest_year"]).resample("8D", on="date")[["tmin", "tmax", "prec", "tavg"]].mean().reset_index()
                    .rename(columns={"date":"start_date_bin"})
                    )
    era_resampled = era_resampled.assign(time_step=era_resampled["start_date_bin"].apply(lambda x: day_of_year_to_time_step[x.day_of_year]))
    
    return era_resampled


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
        predictors = predictors.assign(time_step=predictors["date"].dt.day_of_year.map(day_of_year_to_time_step).astype('Int64'))
    
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
    
    return predictors.loc[(predictors["doy"].between(crop_season_start, crop_season_end)) & (predictors["harvest_year"].between(2003, 2022))].reset_index(drop=True)


def temporal_aggregation(predictors):
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
        df = temporal_aggregation(df)
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
                        doy=pd.to_datetime(ecmwf["valid_time"]).dt.day_of_year, harvest_year=pd.to_datetime(ecmwf["init_date"]).dt.year,
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
    li = []
    for year in ecmwf["harvest_year"].unique():
        ecmwf_year = ecmwf[(ecmwf["harvest_year"] == year) & (ecmwf["valid_time"].dt.year == year)
                           & (ecmwf["doy"].between(start_doy, end_doy+7))].reset_index(drop=True)
        ecmwf_year_resampled = (ecmwf_year
                                .groupby(["init_date", "harvest_year", "location"])
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
    first_doy_available_in_all_years = ecmwf.loc[(ecmwf["harvest_year"] == ecmwf["valid_time"].dt.year)].groupby("harvest_year")["doy"].min().max()
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
    df = df.loc[(df["init_date"].dt.year == df["start_date_bin"].dt.year)].reset_index(drop=True)
    df = df.merge(adm_units_shapefile[["adm_id", "geometry"]], on="adm_id", how="left").set_index(index_names)
    df = gpd.GeoDataFrame(df[["geometry"]], geometry="geometry").reset_index()

    adm_units_shapes_to_ecmwf_grids = assign_adm_units_to_ecmwf_grid_cells(ecmwf, adm_units_shapefile)
    df = df.merge(adm_units_shapes_to_ecmwf_grids.drop(["index_right", "geometry"], axis=1), on="adm_id", how="left")
    
    return df


def assign_adm_units_to_ecmwf_grid_cells(ecmwf, adm_units_shapefile):
    """ 
    Assigns adm_ids to ECMWF grid cells based on the grid cell coordinates and county polygons.   

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
    Merges the ECMWF data to the administrative unit level.   

    Parameters:
    ecmwf (DataFrame): The ECMWF data.
    adm_level_forecasts (DataFrame): The administrative unit level.
    init_month (int): The month when the ECWMF forecast was initialized.
        
    Returns:
    adm_level_forecasts (DataFrame): The administrative unit level with ECMWF data.
    """
    ecmwf_assigned_to_adm_units = (
        adm_level_forecasts
        .merge(ecmwf, on=["init_date", "start_date_bin", "location"], how="left")
        .groupby(["init_date", "start_date_bin", "adm_id"], as_index=False)[["tavg", "tmax", "tmin", "prec"]].mean()
        .assign(time_step=lambda df: df["start_date_bin"].apply(lambda x: day_of_year_to_time_step[x.day_of_year]))
    )
        
    return ecmwf_assigned_to_adm_units


def assign_ecmwf_forecasts_to_adm_units(ecmwf, adm_units_shapefile):
    """ 
    Assigns ECMWF grid cell forecasts to adm units based on the grid cell coordinates and admin units polygons.   

    Parameters:
    ecmwf (DataFrame): The ECMWF data.
    adm_units_shapefile (geoDataFrame): A geodataframe containing the polygons of the administrative units.
    init_month (int): The month when the ECWMF forecast was initialized.
        
    Returns:
    counties_with_ecmwf_data (DataFrame): The counties with ECMWF data.
    first_time_step (int): The first time step of the forecast.
    """
    adm_level_forecasts = create_adm_unit_level_forecast_dataframe(
        [ecmwf["init_date"].unique(), ecmwf["start_date_bin"].unique(), adm_units_shapefile["adm_id"].unique()], 
        ["init_date", "start_date_bin", "adm_id"], adm_units_shapefile, ecmwf)

    ecmwf_assigned_to_adm_units = merge_ecmwf_to_adm_units(ecmwf, adm_level_forecasts)
    first_time_step = ecmwf_assigned_to_adm_units["time_step"].min()
    
    ecmwf_assigned_to_adm_units_pivot = ecmwf_assigned_to_adm_units.pivot(index=["adm_id", "init_date"], columns="time_step", values=["tavg", "tmin", "tmax", "prec"])
    ecmwf_assigned_to_adm_units_pivot.columns = ["_".join([str(col) for col in c]).strip() for c in ecmwf_assigned_to_adm_units_pivot.columns]
    ecmwf_assigned_to_adm_units_pivot = ecmwf_assigned_to_adm_units_pivot.reset_index()
    ecmwf_assigned_to_adm_units_pivot = ecmwf_assigned_to_adm_units_pivot.assign(harvest_year=ecmwf_assigned_to_adm_units_pivot["init_date"].dt.year)
    
    return ecmwf_assigned_to_adm_units_pivot, first_time_step


def normal_correction(obs_data, mod_data, sce_data, cdf_threshold=0.9999999):
    obs_len, mod_len, sce_len = [len(x) for x in [obs_data, mod_data, sce_data]]
    obs_mean, mod_mean, sce_mean = [x.mean() for x in [obs_data, mod_data, sce_data]]
    
    obs_norm, mod_norm, sce_norm = [
        norm.fit(x) for x in [obs_data, mod_data, sce_data]
    ]
    
    sce_norm = list(sce_norm)
    sce_norm[1] += 1e-5
    sce_norm = tuple(sce_norm)
    
    #print(*sce_norm)
    
    obs_cdf = norm.cdf(np.sort(obs_data), *obs_norm)
    mod_cdf = norm.cdf(np.sort(mod_data), *mod_norm)
    sce_cdf = norm.cdf(np.sort(sce_data), *sce_norm)

    obs_cdf = np.maximum(np.minimum(obs_cdf, cdf_threshold), 1 - cdf_threshold)
    mod_cdf = np.maximum(np.minimum(mod_cdf, cdf_threshold), 1 - cdf_threshold)
    sce_cdf = np.maximum(np.minimum(sce_cdf, cdf_threshold), 1 - cdf_threshold)

    sce_argsort = np.argsort(sce_data)

    obs_cdf_intpol = np.interp(
        np.linspace(1, obs_len, sce_len), np.linspace(1, obs_len, obs_len), obs_cdf
    )
    mod_cdf_intpol = np.interp(
        np.linspace(1, mod_len, sce_len), np.linspace(1, mod_len, mod_len), mod_cdf
    )
    obs_cdf_shift, mod_cdf_shift, sce_cdf_shift = [
        (x - 0.5) for x in [obs_cdf_intpol, mod_cdf_intpol, sce_cdf]
    ]

    obs_inverse, mod_inverse, sce_inverse = [
        1.0 / (0.5 - np.abs(x)) for x in [obs_cdf_shift, mod_cdf_shift, sce_cdf_shift]
    ]

    adapted_cdf = np.sign(obs_cdf_shift) * (
        1.0 - 1.0 / (obs_inverse * sce_inverse / mod_inverse)
    )
    adapted_cdf[adapted_cdf < 0] += 1.0
    adapted_cdf = np.maximum(np.minimum(adapted_cdf, cdf_threshold), 1 - cdf_threshold)

    xvals = norm.ppf(np.sort(adapted_cdf), *obs_norm) + obs_norm[-1] / mod_norm[-1] * (
        norm.ppf(sce_cdf, *sce_norm) - norm.ppf(sce_cdf, *mod_norm)
    )

    xvals -= xvals.mean()
    xvals += obs_mean + (sce_mean - mod_mean)

    correction = np.zeros(sce_len)
    correction[sce_argsort] = xvals

    return correction