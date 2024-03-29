import pickle
import numpy as np

from simulation.system import Station
from order_preprocessing.sim.init import get_init_station

from simulation.consts import SELF_ARR_RATE_FIX, OPPO_ARR_RATE_FIX, DEP_FIX

# load data
with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\veh_distance_matrix.pkl',
          'rb') as file:
    dist_array = pickle.load(file)
with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\order_preprocessing\arr_s_array.pkl',
          'rb') as file:
    arr_s_array = SELF_ARR_RATE_FIX * pickle.load(file)
with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\order_preprocessing\arr_c_array.pkl',
          'rb') as file:
    arr_c_array = OPPO_ARR_RATE_FIX * pickle.load(file)
with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\order_preprocessing\dep_s_array.pkl',
          'rb') as file:
    dep_s_array = pickle.load(file)
with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\order_preprocessing\dep_c_array.pkl',
          'rb') as file:
    dep_c_array = pickle.load(file)
with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\station_list.pkl', 'rb') as file:
    station_list = pickle.load(file)
with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\station_list_25.pkl',
          'rb') as file:
    station_list_25 = pickle.load(file)
"""
with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\station_list_50.pkl',
          'rb') as file:
    station_list_50 = pickle.load(file)
with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\station_list_75.pkl',
          'rb') as file:
    station_list_75 = pickle.load(file)
with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\station_list_100.pkl',
          'rb') as file:
    station_list_100 = pickle.load(file)
"""


def get_part_stations(sel_stations: list, stations: list):

    tmp_station_dict = {}
    station_dict = get_init_station()
    for s in range(1, len(sel_stations) + 1):
        tmp_station_dict[s] = station_dict[stations.index(sel_stations[s - 1]) + 1]
    return tmp_station_dict


def get_part_dist_array(sel_stations: list, stations: list):

    tmp_array = np.zeros((len(sel_stations)+1, len(sel_stations)+1))
    idx_list = [0] + [stations.index(s)+1 for s in sel_stations]
    for s in range(tmp_array.shape[0]):
        for k in range(tmp_array.shape[0]):
            tmp_array[s, k] = dist_array[idx_list[s], idx_list[k]]
    return tmp_array


def get_part_mu_s_array(sel_stations: list, stations: list):
    idx_list = [stations.index(s) for s in sel_stations]
    tmp = dep_s_array[:, idx_list]
    return tmp


def get_part_mu_c_array(sel_stations: list, stations: list):
    idx_list = [stations.index(s) for s in sel_stations]
    tmp = dep_c_array[:, idx_list]
    return tmp


def get_part_lambda_s_array(sel_stations: list, stations: list):
    idx_list = [stations.index(s) for s in sel_stations]
    tmp = arr_s_array[:, idx_list]
    return tmp


def get_part_lambda_c_array(sel_stations: list, stations: list):
    idx_list = [stations.index(s) for s in sel_stations]
    tmp = arr_c_array[:, idx_list]
    return tmp


# test case in test
test_case_test = {
    'stations': {
        i: Station(station_id=i, location=(i, i), capacity=50, capacity_opponent=50, num_self=i, num_opponent=50 - i)
        for i in range(1, 51)},
    'dist_array': dist_array,
    'mu_s_array': np.random.uniform(0, 2, (720, 50)),  # (time, station)
    'mu_c_array': np.random.uniform(0, 2, (720, 50)),  # (time, station)
    'lambda_s_array': np.random.uniform(0, 2, (720, 50)),  # (time, station)
    'lambda_c_array': np.random.uniform(0, 2, (720, 50))  # (time, station)
}

# test case all
test_case_all = {
    'stations': get_init_station(),
    'dist_array': dist_array,
    'mu_s_array': dep_s_array,
    'mu_c_array': dep_c_array,
    'lambda_s_array': arr_s_array,
    'lambda_c_array': arr_c_array
}

# test case 25
test_case_25 = {
    'stations': get_part_stations(sel_stations=station_list_25, stations=station_list),
    'dist_array': get_part_dist_array(sel_stations=station_list_25, stations=station_list),
    'mu_s_array': get_part_mu_s_array(sel_stations=station_list_25, stations=station_list),
    'mu_c_array': get_part_mu_c_array(sel_stations=station_list_25, stations=station_list),
    'lambda_s_array': get_part_lambda_s_array(sel_stations=station_list_25, stations=station_list),
    'lambda_c_array': get_part_lambda_c_array(sel_stations=station_list_25, stations=station_list),
}
test_case_25['lambda_s_array'][:144, :] = test_case_25['lambda_s_array'][:144, :] / SELF_ARR_RATE_FIX
test_case_25['lambda_c_array'][:144, :] = test_case_25['lambda_c_array'][:144, :] / OPPO_ARR_RATE_FIX
test_case_25['mu_s_array'][144:, :] = test_case_25['mu_s_array'][144:, :] * DEP_FIX
test_case_25['mu_c_array'][144:, :] = test_case_25['mu_c_array'][144:, :] * DEP_FIX
