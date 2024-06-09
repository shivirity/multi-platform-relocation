import pickle
import numpy as np

from simulation.system import Station
from order_preprocessing.sim.init import get_init_station

from simulation.consts import (MORN_ARR_RATE_FIX, MORN_DEP_RATE_FIX, AFT_ARR_RATE_FIX, AFT_DEP_RATE_FIX,
                               OVERALL_RATE, SEED, SINGLE_LB, SINGLE_UB)

np.random.seed(SEED)

# load data
with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\veh_distance_matrix.pkl',
          'rb') as file:
    dist_array = pickle.load(file)
with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\order_preprocessing\arr_s_array.pkl',
          'rb') as file:
    arr_s_array = pickle.load(file)
with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\order_preprocessing\arr_c_array.pkl',
          'rb') as file:
    arr_c_array = pickle.load(file)
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
test_case_25['lambda_s_array'][:144, :] = test_case_25['lambda_s_array'][:144, :] * MORN_ARR_RATE_FIX * OVERALL_RATE
test_case_25['lambda_c_array'][:144, :] = test_case_25['lambda_c_array'][:144, :] * MORN_ARR_RATE_FIX * OVERALL_RATE
test_case_25['mu_s_array'][:144, :] = test_case_25['mu_s_array'][:144, :] * MORN_DEP_RATE_FIX * OVERALL_RATE
test_case_25['mu_c_array'][:144, :] = test_case_25['mu_c_array'][:144, :] * MORN_DEP_RATE_FIX * OVERALL_RATE

test_case_25['lambda_s_array'][144:, :] = test_case_25['lambda_s_array'][144:, :] * AFT_ARR_RATE_FIX * OVERALL_RATE
test_case_25['lambda_c_array'][144:, :] = test_case_25['lambda_c_array'][144:, :] * AFT_ARR_RATE_FIX * OVERALL_RATE
test_case_25['mu_s_array'][144:, :] = test_case_25['mu_s_array'][144:, :] * AFT_DEP_RATE_FIX * OVERALL_RATE
test_case_25['mu_c_array'][144:, :] = test_case_25['mu_c_array'][144:, :] * AFT_DEP_RATE_FIX * OVERALL_RATE

# 14 -> 168, 21 -> 252
for i in [1, 11, 18, 19]:  # plus dep
    test_case_25['mu_s_array'][168:, i-1] = test_case_25['mu_s_array'][168:, i-1] * 1.4
    test_case_25['mu_c_array'][168:, i-1] = test_case_25['mu_c_array'][168:, i-1] * 1.4
for i in [4]:
    test_case_25['mu_s_array'][168:, i-1] = test_case_25['mu_s_array'][168:, i-1] * 1.2
    test_case_25['mu_c_array'][168:, i-1] = test_case_25['mu_c_array'][168:, i-1] * 1.2
for i in [7, 13, 14, 16, 25]:  # plus dep - balanced
    test_case_25['mu_s_array'][168:, i-1] = test_case_25['mu_s_array'][168:, i-1] * 1.7
    test_case_25['mu_c_array'][168:, i-1] = test_case_25['mu_c_array'][168:, i-1] * 1.7
for i in [5, 8, 10, 12, 24]:  # plus arr
    test_case_25['lambda_s_array'][168:, i-1] = test_case_25['lambda_s_array'][168:, i-1] * 1.4
    test_case_25['lambda_c_array'][168:, i-1] = test_case_25['lambda_c_array'][168:, i-1] * 1.4

# 1.3 for test_case_25
# total_count = 0  # only for calculate.py
# change_count = 0
# # adjust mu_s_array and mu_c_array in test_case
# for i in range(test_case_25['mu_s_array'].shape[0]):
#     for j in range(test_case_25['mu_s_array'].shape[1]):
#         total_count += 1
#         old_val = test_case_25['mu_s_array'][i, j] + test_case_25['mu_c_array'][i, j]
#         test_case_25['mu_s_array'][i, j] = np.random.uniform(0, test_case_25['mu_s_array'][i, j])
#         if test_case_25['mu_s_array'][i, j] > 1.3:
#             test_case_25['mu_s_array'][i, j] = 1.3
#         test_case_25['mu_c_array'][i, j] = old_val - test_case_25['mu_s_array'][i, j]
#         change_count += 1
#         # if i in [1, 11, 18, 19]:
#         #     rand_num = np.random.uniform(-SINGLE_UB, -SINGLE_LB)
#         # elif i in [4]:
#         #     rand_num = np.random.uniform(-SINGLE_UB, -SINGLE_LB)
#         # elif i in [7, 13, 14, 16, 25]:
#         #     rand_num = np.random.uniform(-SINGLE_UB, -SINGLE_LB)
#         # elif i in [5, 8, 10, 12, 24]:
#         #     rand_num = np.random.uniform(SINGLE_LB, SINGLE_UB)
#         # else:
#         #     continue
#         # old_val = test_case_25['mu_s_array'][i, j] + test_case_25['mu_c_array'][i, j]
#         # # 将mu_s_array的值进行rand_num的调整，并保持mu_s_array+mu_c_array的值不变
#         # if test_case_25['mu_s_array'][i, j] * (1 + rand_num) >= 0 and \
#         #         test_case_25['mu_s_array'][i, j] * rand_num <= test_case_25['mu_c_array'][i, j]:
#         #     test_case_25['mu_c_array'][i, j] = (test_case_25['mu_c_array'][i, j] -
#         #                                         test_case_25['mu_s_array'][i, j] * rand_num)
#         #     test_case_25['mu_s_array'][i, j] = test_case_25['mu_s_array'][i, j] * (1 + rand_num)
#         #     change_count += 1
#         # new_val = test_case_25['mu_s_array'][i, j] + test_case_25['mu_c_array'][i, j]
#         # assert abs(old_val - new_val) < 1e-6
# print(f'change rate: {change_count / total_count}')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x_list = [i/2 for i in range(48)]
    for i in range(25):
        plt.figure(figsize=(8, 5), dpi=150)
        lambda_s_list = list(test_case_25['lambda_s_array'][:, i])
        mu_s_list = list(test_case_25['mu_s_array'][:, i])
        # 每两项求和
        lambda_s_list = [sum(lambda_s_list[i:i+6]) for i in range(0, len(lambda_s_list), 6)]
        mu_s_list = [sum(mu_s_list[i:i+6]) for i in range(0, len(mu_s_list), 6)]
        plt.plot(x_list, lambda_s_list, label='lambda_s')
        # plt.plot(test_case_25['lambda_c_array'][:, i], label='lambda_c')
        plt.plot(x_list, mu_s_list, label='mu_s')
        # plt.plot(test_case_25['mu_c_array'][:, i], label='mu_c')
        plt.legend()
        plt.title(f'Station {i+1}')
        plt.show()
