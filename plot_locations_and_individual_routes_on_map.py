# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:42:58 2017

@author: tiwarir
"""

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import os
from sklearn.cluster import KMeans
import numpy as np

# change to required directory
os.chdir("C:/Users/tiwarir/Documents/miscellaneous/yoji")


def plot_location(lat, lon, margin = 0.5, fixed_area = True):
    # determine range to print based on min, max lat and lon of the data
    
    if (fixed_area):
        margin = 0.1
        lat_min = 11.4523739 - margin
        lat_max = 11.693843899999999 + margin
        lon_min = 104.7079825 - margin
        lon_max = 105.008719 + margin
    else:
        lat_min = min(lat) - margin
        lat_max = max(lat) + margin
        lon_min = min(lon) - margin
        lon_max = max(lon) + margin
    
    
    # create map using BASEMAP
    m = Basemap(llcrnrlon=lon_min,
            llcrnrlat=lat_min,
            urcrnrlon=lon_max,
            urcrnrlat=lat_max,
            lat_0=(lat_max - lat_min)/2,
            lon_0=(lon_max-lon_min)/2,
            projection='merc',
            resolution = 'h',
            area_thresh=10000.,
            )
    
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.drawmapboundary(fill_color='#46bcec')
    m.fillcontinents(color = 'white',lake_color='#46bcec')
    
    # convert lat and lon to map projection coordinates
    lons, lats = m(lon, lat)
    lons_0, lats_0 = m(104.933636, 11.552931 )
    # plot points as red dots
#    m.scatter(lons, lats, s = 0.4, marker = 'o', color='r', zorder=2)
    m.scatter(lons_0, lats_0, s = 20, marker = 'o', color='b', zorder=2)
    m.plot(lons, lats, color = 'r')

    plt.show()
    
    
       
# read in data to use for plotted points
location_data = pd.read_csv('locations.csv', header = None)
lat = location_data[0].values
lon = location_data[1].values
plot_location(lat, lon, margin = 2)

# clearly there are three clusters in  the data so I separate them first 
# using k-means clustering

lat_long = location_data.values
kmeans = KMeans(n_clusters = 3, n_init = 10).fit(lat_long)
pd.value_counts(kmeans.labels_)

ind_0 = kmeans.labels_ == 0
ind_1 = kmeans.labels_ == 1
ind_2 = kmeans.labels_ == 2

kmeans.labels_.tolist().index(1)
kmeans.labels_.tolist().index(2)

# points belonging to cluster 0
location_data_0 = location_data.iloc[ind_0, :]
lat = location_data_0[0].values
lon = location_data_0[1].values
plot_location(lat, lon, margin = 0.02)

# points belonging to cluster 1
location_data_1 = location_data.iloc[ind_1, :]
lat = location_data_1[0].values
lon = location_data_1[1].values
plot_location(lat, lon, margin = 20)
location_data_1


# points belonging to cluster 2
location_data_2 = location_data.iloc[ind_2, :]
lat = location_data_2[0].values
lon = location_data_2[1].values
plot_location(lat, lon, margin = 20)
location_data_2

# plotting individual route for delivery boys corresponding to optimized
# route

def get_individual_route(route, breaks, i = 1):
    n_city = len(route) 
    breaks = np.insert(breaks, 0, 0 )               # add the starting point
    breaks = np.append(breaks, n_city)              # add the ending point
    breaks = breaks.astype(int)
    i_route = route[breaks[i-1] : breaks[i]]    
    return i_route

i_route = {}
for i in range(1,n_deliveryboy + 1):
    i_route[i] = get_individual_route(optimal_route, optimal_break, i = i)
       

# plotting route for a given delivery boy
central_location = [11.552931,104.933636]
i = 3
lat = location_data.iloc[i_route[i],:][0].values
lon = location_data.iloc[i_route[i],:][1].values                    
lat = np.insert(lat, 0, central_location[0])   # add central city at the begining
lat = np.append(lat, central_location[0])
lon = np.insert(lon, 0, central_location[1])   # add central city at the end
lon = np.append(lon, central_location[1])
plot_location(lat, lon, margin = 0.001, fixed_area = False) # To zoom in choose fixed_area to be False
