import pandas as pd
import numpy as np
import geopandas as gpd


def resample_ecmwf(ecmwf, modis_dates):
    """
    Resample the ECMWF data to 8-day periods to match the MODIS data.
    
    Parameters
    ----------
    ecmwf : pandas.DataFrame
        The ECMWF data to be resampled.
    modis_dates : pandas.DataFrame
        A dataframe containing the dates of the MODIS bins for each year.
        
    Returns
    -------
    ecmwf_resampled : pandas.DataFrame
        The resampled ECMWF data.
    """
    
    li = []
    for year in list(range(2003, 2022)):
        # Filter by year. MODIS bins for each year are 8-day periods but the last bin may not be complete and stops at the end of the year
        ecmwf_year = ecmwf[(ecmwf["time"].dt.year == year) & (ecmwf["valid_time"].dt.year == year)].reset_index(drop=True)
        # For faster resampling, we will use the location as a string
        ecmwf_year["location"] = ecmwf_year["latitude"].astype(int).astype(str) + ", " + ecmwf_year["longitude"].astype(int).astype(str)
        # start date of first bin is the first purely forecasted bin
        start_date = pd.to_datetime(np.intersect1d(modis_dates[str(year)].values, ecmwf_year["valid_time"].drop_duplicates().sort_values()).min())
        # start date of last bin is the max
        end_date = modis_dates[str(year)].max()

        days_to_end_of_forecast = (end_date - start_date).days + 1
        dates_to_end_of_forecast = pd.date_range(start_date, periods=days_to_end_of_forecast, freq='D')

        ecmwf_year_resampled = (
                                pd.DataFrame(dates_to_end_of_forecast, index=range(days_to_end_of_forecast), columns=["start_date_bin"])
                                    .merge(ecmwf_year, left_on="start_date_bin", right_on="valid_time", how="left")
                                    .groupby(["time", "location"]).resample("8D", on="start_date_bin")[["t2m", "mx2t24", "mn2t24", "tp"]].mean()
                                    .reset_index()
                                )

        li.append(ecmwf_year_resampled)
    ecmwf_resampled = pd.concat(li, ignore_index=True)
    
    return ecmwf_resampled


def assign_ecmwf_forecasts_to_counties(ecmwf, us_continental_counties, adm_level_column_name):
    """ Assigns ECMWF grid cell forecasts to US counties based on the grid cell coordinates and county polygons.   

    Parameters
    ----------
    ecmwf : pd.DataFrame
        The ECMWF data.
    us_continental_counties : geopandas.geoDataFrame
        A geodataframe containing the US counties.
    adm_level_column_name: string
        The name of the column containing the unique identifier for each county.
        
    Returns
    -------
    counties_with_ecmwf_data : pd.DataFrame
        The counties with ECMWF data.
    """
    # create final dataframe with all possible combinations of init_date, modis_bin, and adm_level_column_name
    counties_with_ecmwf_data = pd.DataFrame(index=pd.MultiIndex.from_product([
        ecmwf["time"].unique(), 
        ecmwf["start_date_bin"].unique(), 
        us_continental_counties[adm_level_column_name].unique()], names=["init_date", "modis_bin", adm_level_column_name])).reset_index()
    counties_with_ecmwf_data = counties_with_ecmwf_data.loc[(counties_with_ecmwf_data["init_date"].dt.year == counties_with_ecmwf_data["modis_bin"].dt.year)].reset_index(drop=True)

    # merge US county polygons to final dataframe and convert to geodataframe
    counties_with_ecmwf_data = counties_with_ecmwf_data.merge(us_continental_counties[[adm_level_column_name, "geometry"]], left_on=adm_level_column_name, right_on=adm_level_column_name, how="left").set_index(["init_date", "modis_bin", adm_level_column_name])
    counties_with_ecmwf_data = gpd.GeoDataFrame(counties_with_ecmwf_data[["geometry"]], geometry="geometry").reset_index()

    # reproject county polygons from geographic to planar coordinates
    # https://gis.stackexchange.com/questions/466703/warning-message-when-doing-spatial-join-nearest-neighbor-on-geopandas
    us_continental_counties_planar = us_continental_counties[[adm_level_column_name, "geometry"]].to_crs(epsg=32723)
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
    counties_with_ecmwf_data = counties_with_ecmwf_data.merge(ecmwf.rename({"time":"init_date", "start_date_bin":"modis_bin"}, axis=1), on=["init_date", "modis_bin", "location"], how="left")
    # groupby county and calculate mean of all ECMWF variables (in case several ECMWF grids are assigned to one county)
    counties_with_ecmwf_data = counties_with_ecmwf_data.groupby(["init_date", "modis_bin", adm_level_column_name])[["t2m", "mx2t24", "mn2t24", "tp"]].mean().reset_index()

    return counties_with_ecmwf_data