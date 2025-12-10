import pandas
import geopandas
import numpy as np

def get_landuse_and_zones(parameter):
    #### Loading Zone Data ####
    zones = geopandas.read_file("zones.geojson")
       
    ####to create a column with centroids ####
    centroidFunction = lambda row: (row['geometry'].centroid.y, row['geometry'].centroid.x)
    zones['centroid'] = zones.apply(centroidFunction, axis=1)
    
    #### Loading Landuse Data ####
    landuse = pandas.read_csv("landuse.csv", sep=";")

    ##parameter
    constant, beta_inc, beta_dummy = parameter
    ##Calculate Utility of owning a car
    V_car = constant + beta_inc * landuse['inc'] + beta_dummy * landuse['citycenter']
    ##Binary Logit
    landuse['car_ownership'] = np.exp(V_car) / (1 + np.exp(V_car))
    
    # Add zone information to landuse data
    zones.set_index('area', inplace=True)
    landuse = landuse.join(zones, on='area')
    
    return landuse, zones   
