import numpy as np

from sim.system import Station

# test case 1
dist_array = 5 * np.random.randint(4, 12, size=(50, 50))
for i in range(50):
    for j in range(i, 50):
        if i == j:
            dist_array[i, i] = 0
        else:
            dist_array[i, j] = dist_array[j, i]
test_case_1 = {
    'stations': {i: Station(station_id=i,location=(i,i),capacity=50,capacity_opponent=50,num_self=i,num_opponent=50-i) for i in range(1,51)},
    'dist_array': dist_array,
    'mu_array': np.random.uniform(0, 4, (720, 50)),  # (time, station)
    'lambda_s_array': np.random.uniform(0, 2, (720, 50)),  # (time, station)
    'lambda_c_array': np.random.uniform(0, 2, (720, 50))  # (time, station)
}

# test case 2
dist_array = 5 * np.random.randint(4, 12, size=(50, 50))
for i in range(50):
    for j in range(i, 50):
        if i == j:
            dist_array[i, i] = 0
        else:
            dist_array[i, j] = dist_array[j, i]
test_case_2 = {
    'stations': {i: Station(station_id=i,location=(i,i),capacity=50,capacity_opponent=50,num_self=1,num_opponent=40) for i in range(1,51)},
    'dist_array': dist_array,
    'mu_array': np.random.uniform(0, 4, (720, 50)),  # (time, station)
    'lambda_s_array': np.random.uniform(0, 2, (720, 50)),  # (time, station)
    'lambda_c_array': np.random.uniform(0, 2, (720, 50))  # (time, station)
}