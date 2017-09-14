# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:18:36 2017

@author: tiwarir
"""

###############################################################################
# n_city is the number of cities including the central city
###############################################################################

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import time


def read_city_lat_long(filename):
    location_data = pd.read_csv(filename, header = None)
    return location_data


def deg2rad(deg):
    return deg * (math.pi/180)  


def get_distance_from_lat_long(lat1,lon1,lat2,lon2):
    R = 6371
    dLat = deg2rad(lat2-lat1)
    dLon = deg2rad(lon2-lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
                math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2))* \
                     math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d
    
def get_distance_matrix(location_data, central_location):
    location_data.loc[-1] = central_location
    location_data.index = location_data.index + 1
    location_data = location_data.sort_index()
    n_city, _ = location_data.shape
    dist_mat = np.zeros([n_city, n_city])
    for i in range(n_city):
        for j in range(i+1, n_city):
            lat1 = location_data.iat[i,0]
            lon1 = location_data.iat[i,1]
            lat2 = location_data.iat[j,0]
            lon2 = location_data.iat[j,1]            
            dist_mat[i,j] = get_distance_from_lat_long(lat1,lon1,lat2,lon2)
            dist_mat[j,i] = dist_mat[i,j]           
    return dist_mat
    

def calculate_total_trip_distance(dist_mat, route, breaks):
    n_city = len(route) 
    n_deliveryboy = len(breaks) + 1
    d = 0
    breaks = np.insert(breaks, 0, 0 )               # add the starting point
    breaks = np.append(breaks, n_city)              # add the ending point
    breaks = breaks.astype(int)
    for i in range(n_deliveryboy):
        i_route = route[breaks[i] : breaks[i+1]]
        i_route = np.insert(i_route, 0, 0)            # add central city at the begining
        i_route = np.append(i_route, 0)               # add central city at the end
        i_route_len = len(i_route)
        for j in range(i_route_len - 1):
            d += dist_mat[i_route[j], i_route[j+1]] 
    return d


def calculate_distance_for_whole_population(dist_mat, pop_route, pop_break):
    n, _ = pop_route.shape
    dist = np.zeros(n)
    for i in range(n):
        dist[i] = calculate_total_trip_distance(dist_mat, pop_route[i], pop_break[i])
    return dist
    


def create_temporay_break(n_city, n_deliveryboy):
    temporary_breaks = np.random.permutation(n_city - 1)
    n_breaks = n_deliveryboy - 1
    breaks = np.sort(temporary_breaks[:n_breaks])
    return breaks


def initialize_population(population_size, n_city, n_deliveryboy):
    n_breaks = n_deliveryboy - 1
    population_route = np.zeros((population_size,n_city - 1))
    population_breaks = np.zeros((population_size, n_breaks))
    
    population_route[0,:] = range(1, n_city)
    population_breaks[0,:] = create_temporay_break(n_city, n_deliveryboy)
    
    for i in range(1,population_size):
        population_route[i,:] = np.random.permutation(range(1,n_city))
        population_breaks[i,:] = create_temporay_break(n_city, n_deliveryboy)
        
    population_break = population_breaks.astype(int)
    population_route = population_route.astype(int) 
    
    return population_route, population_break
    
    
def update_population(population_route, population_breaks, population_distances):
    nrr, ncr = population_route.shape
    nrb, ncb = population_breaks.shape
    new_population_routes = np.zeros((nrr, ncr)) 
    new_population_breaks = np.zeros((nrb, ncb))
    
    random_order = np.random.permutation(nrr)
    
    for p in range(0, nrr, 8):
        ind = random_order[p:p+8]
        routes = population_route[ind,:]
        breaks = population_breaks[ind,:]
        distances = population_distances[ind]
        
        min_ind = np.argmin(distances)
        best_pop_route = routes[min_ind,:]
        best_pop_break = breaks[min_ind, :]
        tmp_pop_routes, tmp_pop_breaks = generate_new_solution_from_the_best_solutions(best_pop_route, best_pop_break)
        new_population_routes[p:p+8,:] = tmp_pop_routes
        new_population_breaks[p:p+8,:] = tmp_pop_breaks
    
    return new_population_routes, new_population_breaks
        
        
    
def generate_new_solution_from_the_best_solutions(best_pop_route, best_pop_break):
    n_route = len(best_pop_route) 
    n_break = len(best_pop_break)
    insertion_point = np.sort(np.ceil((n_route-1)*np.random.random((2))))
    insertion_point = insertion_point.astype(int)
    
    I = insertion_point[0]
    J = insertion_point[1]
    
    tmp_pop_route = np.zeros((8, n_route))
    tmp_pop_break = np.zeros((8, n_break))
    
    # create k solutions
    for k in range(8):
        tmp_pop_route[k,:] = best_pop_route
        tmp_pop_break[k,:] = best_pop_break
        if (k == 1):                # flipping
            ind_1 = range(I,J)
            ind_2 = range(J-1, I-1, -1)
            tmp_pop_route[k, ind_1] = tmp_pop_route[k, ind_2]
        elif ( k == 2):            # swapping 
            tmp_pop_route[k,[I,J]] = tmp_pop_route[k,[J,I]]
        elif (k == 3):            # sliding
            ind_1 = range(I,J)
            ind_2 = range(I+1,J)
            ind_2.append(I)
            tmp_pop_route[k,ind_1] = tmp_pop_route[k,ind_2]
        elif(k == 4 ):         # modify breaks
            tmp_pop_break[k,:] = create_temporay_break(n_route + 1, n_break + 1) 
        elif(k == 5): # flip, modify breaks
            ind_1 = range(I,J)
            ind_2 = range(J-1, I-1, -1)
            tmp_pop_route[k, ind_1] = tmp_pop_route[k, ind_2]
            tmp_pop_break[k,:] = create_temporay_break(n_route + 1, n_break + 1)            
        elif(k == 6):   # swap, modify breaks
            tmp_pop_route[k,[I,J]] = tmp_pop_route[k, [J,I]]
            tmp_pop_break[k,:] = create_temporay_break(n_route + 1, n_break + 1)
        elif (k == 7):
            ind_1 = range(I,J)
            ind_2 = range(I+1,J)
            ind_2.append(I)
            tmp_pop_route[k,ind_1] = tmp_pop_route[k,ind_2]
            tmp_pop_break[k,:] = create_temporay_break(n_route + 1, n_break + 1)
    return tmp_pop_route, tmp_pop_break


def run_optimization(population_size, n_deliveryboy, n_iteration, central_location, location_data):
    
#    location_data = read_city_lat_long(filename)
    
    dist_mat = get_distance_matrix(location_data, central_location)
    
    n_city, _ = dist_mat.shape
    
    pop_route, pop_break = initialize_population(population_size, n_city, n_deliveryboy)
    
    pop_distance = calculate_distance_for_whole_population(dist_mat, pop_route, pop_break)
    
    global_min = np.inf
    convergence = np.zeros((n_iteration, 3))
    
    start_time = time.time()
    for i in range(n_iteration):
        if (i == 0):
            pop_route, pop_break = initialize_population(population_size, n_city, n_deliveryboy)
            pop_distance = calculate_distance_for_whole_population(dist_mat, pop_route, pop_break)
        else:
            pop_route, pop_break = update_population(pop_route, pop_break, pop_distance)
            pop_distance = calculate_distance_for_whole_population(dist_mat, pop_route, pop_break)

        ind = np.argmin(pop_distance)
        
        if (pop_distance[ind] < global_min):
            global_min = pop_distance[ind]
            optimal_route = pop_route[ind]
            optimal_break = pop_break[ind]
        
        # here add a line to keep track of the optimal distance
        end_time = time.time()
        execution_time = end_time - start_time
        convergence[i] = [i, global_min, execution_time]
        if(i%100 == 0):
            print i, global_min
#            print 'Iteration No: ', i,  ' Global Minimum: ', global_min
                    
    return global_min, optimal_route, optimal_break, convergence

   
if __name__ == '__main__':
    population_size = 80
    n_deliveryboy = 25
    n_iteration = 5000
    central_location = [11.552931,104.933636]
    filename = 'locations.csv'  
    location_data = read_city_lat_long('locations.csv')
    global_min, optimal_route, optimal_break, convergence = run_optimization(population_size, n_deliveryboy, n_iteration, central_location, location_data)
    plt.plot(convergence[:,0], convergence[:,1])
    get_distance_matrix
#    pop_route, pop_break = initialize_population(10, 8, 3)
#    dist_matrix = np.random.randint(100, size=(11, 11)) 
#    d = calculate_total_trip_distance(dist_matrix, pop_route[1], pop_break[1])
#    population_distances = calculate_distance_for_whole_population(dist_matrix, pop_route, pop_break)
#    central_location = [11.552931,104.933636]

#    n_deliveryboy = 25
    dist_mat = get_distance_matrix(location_data, central_location)
#    n_city, _ = dist_mat.shape
#    population_size = 80  # should be a multiple of 8
#    population_route, population_break = initialize_population(population_size, n_city, n_deliveryboy)



    
