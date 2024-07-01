import pandas as pd
import numpy as np
import geopandas as gpd

day_of_year_to_time_step = {
    1: 0, 9: 1, 17: 2, 25: 3, 33: 4, 41: 5, 49: 6, 57: 7, 65: 8, 73: 9, 81: 10, 89: 11, 
    97: 12, 105: 13, 113: 14, 121: 15, 129: 16, 137: 17, 145: 18, 153: 19, 161: 20, 
    169: 21, 177: 22, 185: 23, 193: 24, 201: 25, 209: 26, 217: 27, 225: 28, 233: 29, 
    241: 30, 249: 31, 257: 32, 265: 33, 273: 34, 281: 35, 289: 36, 297: 37, 305: 38, 
    313: 39, 321: 40, 329: 41, 337: 42, 345: 43, 353: 44, 361: 45
}

month_to_first_bin = {
    1: 1, 2: 33, 3: 65, 4: 97, 5: 121, 6: 153, 7: 185, 8: 217, 9: 249, 10: 281, 
    11: 305, 12: 337
}

month_to_last_bin = {
    1: 25, 2: 57, 3: 89, 4: 113, 5: 145, 6: 177, 7: 209, 8: 241, 9: 273, 10: 297, 
    11: 329, 12: 361
}


def set_crop_season(country, crop):
    """
    Sets the crop season for a given country and crop.

    Parameters:
    country (str): The country code (e.g., "BR" for Brazil, "US" for United States).
    crop (str): The crop type (e.g., "wheat", "maize").

    Returns:
    tuple: A tuple containing the crop season start and crop season end.

    Raises:
    ValueError: If an invalid country or crop is provided.
    """
    match country:
        case "BR":
            match crop:
                case "wheat":
                    crop_season_start_day = 121
                    crop_season_end_day = 329
                case "maize":
                    crop_season_start_day = 1
                    crop_season_end_day = 209
                case _:
                    raise ValueError("Invalid crop, must be one of ['maize', 'wheat']")
        case "US":
            match crop:
                case "wheat":
                    crop_season_start_day = 33
                    crop_season_end_day = 241
                case "maize":
                    crop_season_start_day = 97
                    crop_season_end_day = 297
                case _:
                    raise ValueError("Invalid crop, must be one of ['maize', 'wheat']")
        case _:
            raise ValueError("Invalid country, must be one of ['US', 'BR']")

    print(f"Running for {country} and {crop} with crop season from day of years {crop_season_start_day}-{crop_season_end_day}")
    return (crop_season_start_day, crop_season_end_day)

def pivot_predictors(predictors):
    
    value_columns = predictors.columns.difference(["adm_id", "harvest_year", "date", "time_step"]).tolist()
    
    predictors_pivot = predictors.dropna(subset="time_step").pivot(index=["adm_id", "harvest_year"], columns="time_step", values=value_columns).interpolate(axis=1, method='linear', limit_direction='both')
    predictors_pivot.columns = ["_".join([str(col) for col in c]).strip() for c in predictors_pivot.columns]
    predictors_pivot = predictors_pivot.reset_index()
    
    return predictors_pivot

def filter_predictors_by_adm_ids(predictor_list, adm_ids):
    """
    Filters the predictors by the provided adm_ids.

    Parameters:
    predictor_list (list of pd.DataFrame): The predictor DataFrames.
    adm_ids (list): The list of adm_ids to filter.

    Returns:
    list of pd.DataFrames: The list of filtered predictor DataFrames.
    """
    return [predictors.loc[predictors["adm_id"].isin(adm_ids)].reset_index(drop=True) for predictors in predictor_list]


def assign_time_steps(predictors):
    """
    Assigns date columns to the predictors DataFrame.

    Parameters:
    predictors (pd.DataFrame): The predictors DataFrame.

    Returns:
    pd.DataFrame: The predictors DataFrame with date columns.
    """
    
    return predictors.assign(time_step=predictors["date"].dt.day_of_year.map(day_of_year_to_time_step))


def assign_date_and_year_columns(predictors):
    """
    Assigns date columns to the predictors DataFrame.

    Parameters:
    predictors (pd.DataFrame): The predictors DataFrame.

    Returns:
    pd.DataFrame: The predictors DataFrame with date columns.
    """
    predictors = predictors.assign(date=pd.to_datetime(predictors["date"], format="%Y%m%d"), harvest_year=pd.to_datetime(predictors["date"], format="%Y%m%d").dt.year)
    
    return predictors


def filter_predictors_by_crop_season(predictors, crop_season_start, crop_season_end):
    """
    Filters the predictors by the crop season start and end.

    Parameters:
    predictors (pd.DataFrame): The predictors DataFrame.
    crop_season_start (int): The day of year of the first bin of the crop season start month.
    crop_season_end (int): The day of year of the last bin of the crop season end month.

    Returns:
    pd.DataFrame: The filtered predictors DataFrame.
    """
    return predictors.loc[(predictors["date"].dt.day_of_year.between(crop_season_start, crop_season_end)) & (predictors["harvest_year"].between(2003, 2023))].reset_index(drop=True)

def resample_to_8_day_bins(predictors, start_date, end_date):
    """
    Resamples the predictors to 8-day bins.

    Parameters:
    predictors (pd.DataFrame): The predictors DataFrame.

    Returns:
    pd.DataFrame: The resampled predictors DataFrame.
    """
    # test if predictors are already in 8-day bins
    if (end_date - start_date + 1) > predictors["date"].dt.day_of_year.nunique():
        return predictors
    
    value_columns = predictors.columns.difference(["adm_id", "harvest_year", "date", "day_of_year"]).tolist()
    
    return predictors.groupby(["adm_id", "harvest_year"]).resample("8D", on="date")[value_columns].mean().reset_index()

def preprocess_temporal_data(data_list, start_date, end_date):
    """
    Preprocesses the temporal data by assigning date columns and filtering by the start and end dates.
    """
    processed_data = []
    for df in data_list:
        df = df.drop(columns=["crop_name"])
        df = assign_date_and_year_columns(df)
        df = filter_predictors_by_crop_season(df, start_date, end_date)
        df = resample_to_8_day_bins(df, start_date, end_date)
        df = assign_time_steps(df)
        df = pivot_predictors(df)
        processed_data.append(df)
    return processed_data


def resample_ecmwf(ecmwf, start_dates):
    """
    Resample the ECMWF data to 8-day periods to match the MODIS data.
    
    Parameters
    ----------
    ecmwf : pandas.DataFrame
        The ECMWF data to be resampled.
    start_dates : pandas.DataFrame
        A dataframe containing the dates of the 8-day bins for each year.
        
    Returns
    -------
    ecmwf_resampled : pandas.DataFrame
        The resampled ECMWF data.
    """
    
    li = []
    for year in ecmwf["time"].dt.year.unique():
        # Filter by year. MODIS bins for each year are 8-day periods but the last bin may not be complete and stops at the end of the year
        ecmwf_year = ecmwf[(ecmwf["time"].dt.year == year) & (ecmwf["valid_time"].dt.year == year)].reset_index(drop=True)
        # For faster resampling, we will use the location as a string
        ecmwf_year["location"] = ecmwf_year["latitude"].astype(int).astype(str) + ", " + ecmwf_year["longitude"].astype(int).astype(str)
        # start date of first bin is the first purely forecasted bin
        start_date = pd.to_datetime(np.intersect1d(start_dates[str(year)].values, ecmwf_year["valid_time"].drop_duplicates()).min())
        # start date of last bin is the max
        end_date = start_dates[str(year)].max()

        days_to_end_of_forecast = (end_date - start_date).days + 1
        dates_to_end_of_forecast = pd.date_range(start_date, periods=days_to_end_of_forecast, freq='D')

        ecmwf_year_resampled = (
                                pd.DataFrame(dates_to_end_of_forecast, index=range(days_to_end_of_forecast), columns=["start_date_bin"])
                                    .merge(ecmwf_year, left_on="start_date_bin", right_on="valid_time", how="left")
                                    .groupby(["time", "location"]).resample("8D", on="start_date_bin")[["tavg", "tmax", "tmin", "prec"]].mean()
                                    .reset_index()
                                )

        li.append(ecmwf_year_resampled)
    ecmwf_resampled = pd.concat(li, ignore_index=True)
    
    return ecmwf_resampled


def assign_ecmwf_forecasts_to_counties(ecmwf, continental_counties, start_dates, adm_level_column_name):
    """ Assigns ECMWF grid cell forecasts to US counties based on the grid cell coordinates and county polygons.   

    Parameters
    ----------
    ecmwf : pd.DataFrame
        The ECMWF data.
    continental_counties : geopandas.geoDataFrame
        A geodataframe containing the counties.
    start_dates : pd.DataFrame
        A dataframe containing the dates of the 8-day bins for each year.
    adm_level_column_name: string
        The name of the column containing the unique identifier for each county.
        
    Returns
    -------
    counties_with_ecmwf_data : pd.DataFrame
        The counties with ECMWF data.
    """
    # Create a dictionary to map the day of the year to the time step
    day_of_year_to_time_step = start_dates.iloc[:, 0].dt.day_of_year.reset_index().set_index(start_dates.columns[0])["index"].to_dict()
    
    init_month_as_number = ecmwf["time"].dt.month.unique()[0]
    
    # create final dataframe with all possible combinations of init_date, modis_bin, and adm_level_column_name
    counties_with_ecmwf_data = pd.DataFrame(index=pd.MultiIndex.from_product([
        ecmwf["time"].unique(), 
        ecmwf["start_date_bin"].unique(), 
        continental_counties[adm_level_column_name].unique()], names=["init_date", "start_date_bin", adm_level_column_name])).reset_index()
    counties_with_ecmwf_data = counties_with_ecmwf_data.loc[(counties_with_ecmwf_data["init_date"].dt.year == counties_with_ecmwf_data["start_date_bin"].dt.year)].reset_index(drop=True)

    # merge US county polygons to final dataframe and convert to geodataframe
    counties_with_ecmwf_data = counties_with_ecmwf_data.merge(continental_counties[[adm_level_column_name, "geometry"]], left_on=adm_level_column_name, right_on=adm_level_column_name, how="left").set_index(["init_date", "start_date_bin", adm_level_column_name])
    counties_with_ecmwf_data = gpd.GeoDataFrame(counties_with_ecmwf_data[["geometry"]], geometry="geometry").reset_index()

    # reproject county polygons from geographic to planar coordinates
    # https://gis.stackexchange.com/questions/466703/warning-message-when-doing-spatial-join-nearest-neighbor-on-geopandas
    us_continental_counties_planar = continental_counties[[adm_level_column_name, "geometry"]].to_crs(epsg=32723)
    # extract latitude and longitude from location column in ECWMF dataframe
    ecmwf_unique_lat_lon = ecmwf["location"].str.split(", ", expand=True).rename(columns={0: "latitude", 1: "longitude"}).drop_duplicates().reset_index(drop=True)
    # convert ECMWF lat-lon pairs to geodataframe
    ecmwf_unique_lat_lon_geo_df = gpd.GeoDataFrame(ecmwf_unique_lat_lon, geometry=gpd.points_from_xy(ecmwf_unique_lat_lon["longitude"], ecmwf_unique_lat_lon["latitude"]), crs="EPSG:4326").to_crs(epsg=32723)
    # spatial join of county polygons to ECMWF lat-lon pairs
    county_shapes_to_ecmwf_grids = gpd.sjoin_nearest(us_continental_counties_planar, ecmwf_unique_lat_lon_geo_df, how="left")

    # merge ECMWF grid cell coordinates to each county in final dataframe
    counties_with_ecmwf_data = counties_with_ecmwf_data.merge(county_shapes_to_ecmwf_grids.drop(["index_right", "geometry"], axis=1), on=adm_level_column_name, how="left")

    # create location column for each county in final dataframe
    counties_with_ecmwf_data["location"] = counties_with_ecmwf_data["latitude"].astype(str) + ", " + counties_with_ecmwf_data["longitude"].astype(str)
    # merge ECMWF data to final dataframe
    counties_with_ecmwf_data = counties_with_ecmwf_data.merge(ecmwf.rename({"time":"init_date"}, axis=1), on=["init_date", "start_date_bin", "location"], how="left")
    # groupby county and calculate mean of all ECMWF variables (in case several ECMWF grids are assigned to one county)
    counties_with_ecmwf_data = counties_with_ecmwf_data.groupby(["init_date", "start_date_bin", adm_level_column_name])[["tavg", "tmax", "tmin", "prec"]].mean().reset_index()

    # assign time step to each bin
    counties_with_ecmwf_data = counties_with_ecmwf_data.assign(time_step=counties_with_ecmwf_data["start_date_bin"].apply(lambda x: day_of_year_to_time_step[x.day_of_year]))
    
    if init_month_as_number == 1:
        counties_with_ecmwf_data["time_step"] = counties_with_ecmwf_data["time_step"] - 1
    
    # convert to wide format
    counties_with_ecmwf_time_step_pivot = counties_with_ecmwf_data.pivot(index=["adm_id", "init_date"], columns="time_step", values=["tavg", "tmin", "tmax", "prec"])
    counties_with_ecmwf_time_step_pivot.columns = ["_".join([str(col) for col in c]).strip() for c in counties_with_ecmwf_time_step_pivot.columns]
    counties_with_ecmwf_time_step_pivot = counties_with_ecmwf_time_step_pivot.reset_index()
    
    return counties_with_ecmwf_time_step_pivot