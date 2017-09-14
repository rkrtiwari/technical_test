# -*- coding: utf-8 -*-
"""
Created on Sat Sep 09 15:23:30 2017

@author: tiwarir
"""

import os
import pandas as pd
import numpy as np
import mts_functions as mtsf
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

reload(mtsf)

# set the working directory
os.getcwd()
os.chdir("C:/Users/tiwarir/Documents/miscellaneous/yoji")

location = pd.read_csv('locations.csv', header = None)
central_location = [11.552931,104.933636]

# divide the data into three clusters
kmeans = KMeans(n_clusters = 3, n_init = 10).fit(location.values)
pd.value_counts(kmeans.labels_)
ind_0 = kmeans.labels_ == 0
ind_1 = kmeans.labels_ == 1
ind_2 = kmeans.labels_ == 2

location_1 = location.iloc[ind_1,:]
location_2 = location.iloc[ind_2,:]

# Find out if a single delivery boy should cover points in cluster 1 and 2 or 
# two different deivery boys should cover it basad on the route minimzation 


dist_mat = mtsf.get_distance_matrix(location, central_location)

location_1.index
location_2.index

dist_center_location_1 = dist_mat[0,898]
dist_center_location_2 = dist_mat[0,1188]
dist_location_1_location_2 = dist_mat[898,1188]
            
# distance travelled when only one delivery boy goes to both cluster 1 and 2
dist_1_delivery_boy = dist_center_location_1 + dist_location_1_location_2 + dist_center_location_2

# distance travelled when only cluster 1 and cluster 2 are visited by two different delivery boys
dist_2_delivery_boy = 2*dist_center_location_1 + 2*dist_center_location_2

# since the distance covered by 1 delivery boy is much smaller, I chose only one delivery boy to cover cluster 1
# and cluster 2 and rest 24 to cover cluster 0

# Running optimization only for locations in cluster 0, only
location_0 = location.iloc[ind_0,:].copy()
population_size = 80
n_deliveryboy = 24
n_iteration = 20000
central_location = [11.552931,104.933636]
global_min, optimal_route, optimal_break, convergence = mtsf.run_optimization(population_size, n_deliveryboy, n_iteration, central_location, location_0)

# plot the convergence data
# distance with iteration
plt.plot(convergence[:,0], convergence[:,1])
plt.xlabel('Iteration')
plt.ylabel('Total distance (km)')
plt.show()
# running time with iteration
plt.plot(convergence[:,0], convergence[:,2]/60)
plt.xlabel('Iteration')
plt.ylabel('Time (mins)')
plt.show()








    