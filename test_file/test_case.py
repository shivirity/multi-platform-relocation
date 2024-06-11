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

# 14 -> 168, 21 -> 252, 22 -> 264
for i in [1, 11, 18, 19]:  # plus dep
    test_case_25['mu_s_array'][168:264, i-1] = test_case_25['mu_s_array'][168:264, i-1] * 1.4
    test_case_25['mu_c_array'][168:264, i-1] = test_case_25['mu_c_array'][168:264, i-1] * 1.4
    for j in range(264, test_case_25['mu_s_array'].shape[0]):
        test_case_25['mu_s_array'][j, i-1] = test_case_25['mu_s_array'][j, i-1] * ((1 - 1.4)/24 * j + 11.5 * 1.4 - 10.5)
        test_case_25['mu_c_array'][j, i-1] = test_case_25['mu_c_array'][j, i-1] * ((1 - 1.4)/24 * j + 11.5 * 1.4 - 10.5)
for i in [4]:
    test_case_25['mu_s_array'][168:264, i-1] = test_case_25['mu_s_array'][168:264, i-1] * 1.2
    test_case_25['mu_c_array'][168:264, i-1] = test_case_25['mu_c_array'][168:264, i-1] * 1.2
    for j in range(264, test_case_25['mu_s_array'].shape[0]):
        test_case_25['mu_s_array'][j, i-1] = test_case_25['mu_s_array'][j, i-1] * ((1 - 1.2)/24 * j + 11.5 * 1.2 - 10.5)
        test_case_25['mu_c_array'][j, i-1] = test_case_25['mu_c_array'][j, i-1] * ((1 - 1.2)/24 * j + 11.5 * 1.2 - 10.5)
for i in [7, 13, 14, 16, 25]:  # plus dep - balanced
    test_case_25['mu_s_array'][168:264, i-1] = test_case_25['mu_s_array'][168:264, i-1] * 1.7
    test_case_25['mu_c_array'][168:264, i-1] = test_case_25['mu_c_array'][168:264, i-1] * 1.7
    for j in range(264, test_case_25['mu_s_array'].shape[0]):
        test_case_25['mu_s_array'][j, i-1] = test_case_25['mu_s_array'][j, i-1] * ((1 - 1.7)/24 * j + 11.5 * 1.7 - 10.5)
        test_case_25['mu_c_array'][j, i-1] = test_case_25['mu_c_array'][j, i-1] * ((1 - 1.7)/24 * j + 11.5 * 1.7 - 10.5)
for i in [5, 8, 10, 12, 24]:  # plus arr
    test_case_25['lambda_s_array'][168:264, i-1] = test_case_25['lambda_s_array'][168:264, i-1] * 1.4
    test_case_25['lambda_c_array'][168:264, i-1] = test_case_25['lambda_c_array'][168:264, i-1] * 1.4
    for j in range(264, test_case_25['mu_s_array'].shape[0]):
        test_case_25['lambda_s_array'][j, i-1] = test_case_25['lambda_s_array'][j, i-1] * ((1 - 1.4)/24 * j + 11.5 * 1.4 - 10.5)
        test_case_25['lambda_c_array'][j, i-1] = test_case_25['lambda_c_array'][j, i-1] * ((1 - 1.4)/24 * j + 11.5 * 1.4 - 10.5)

# # 1.3 for test_case_25
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
#
# total_count = 0
# change_count = 0
# dep_idx_list = []
# dep_gap_list = []
# aft_dep_idx_list = []
# aft_dep_gap_list = []
# # adjust mu_s_array and mu_c_array in test_case
# for i in range(test_case_25['mu_s_array'].shape[0]):
#     for j in range(test_case_25['mu_s_array'].shape[1]):
#         old_val = test_case_25['mu_s_array'][i, j] + test_case_25['mu_c_array'][i, j]
#         gap = test_case_25['mu_s_array'][i, j] - test_case_25['lambda_s_array'][i, j]
#         if gap < 0:
#             if test_case_25['mu_s_array'][i, j] - 3 * gap <= old_val:
#                 test_case_25['mu_s_array'][i, j] -= 3 * gap
#                 test_case_25['mu_c_array'][i, j] += 3 * gap
#                 change_count += 1
#             elif test_case_25['mu_s_array'][i, j] - 2 * gap <= old_val:
#                 test_case_25['mu_s_array'][i, j] -= 2 * gap
#                 test_case_25['mu_c_array'][i, j] += 2 * gap
#                 change_count += 1
#             elif test_case_25['mu_s_array'][i, j] - 1 * gap <= old_val:
#                 test_case_25['mu_s_array'][i, j] -= 1 * gap
#                 test_case_25['mu_c_array'][i, j] += 1 * gap
#                 change_count += 1
#                 assert test_case_25['mu_s_array'][i, j] == test_case_25['lambda_s_array'][i, j], (
#                     f'{test_case_25["mu_s_array"][i, j]}'
#                     f'{test_case_25["lambda_s_array"][i, j]}')
#             else:
#                 pass
#         else:
#             # record
#             dep_idx_list.append((i, j))
#             dep_gap_list.append(gap)
#             if i >= 168:
#                 aft_dep_idx_list.append((i, j))
#                 aft_dep_gap_list.append(gap)
#             # if test_case_25['mu_s_array'][i, j] - 1 * gap >= 0:
#             #     test_case_25['mu_s_array'][i, j] -= 1 * gap
#             #     test_case_25['mu_c_array'][i, j] += 1 * gap
#             #     change_count += 1
#             #     assert test_case_25['mu_s_array'][i, j] == test_case_25['lambda_s_array'][i, j], (
#             #         f'{test_case_25["mu_s_array"][i, j]}'
#             #         f'{test_case_25["lambda_s_array"][i, j]}')
#             # else:
#             #     pass
#         total_count += 1
# thd = np.percentile(aft_dep_gap_list, 0.0001)
# print(f'length of aft_dep_gap_list: {len(aft_dep_gap_list)}')
# inds = [i for i, x in enumerate(aft_dep_gap_list) if x <= thd]
# aft_inds = [aft_dep_idx_list[i] for i in inds]
# print(f'length of aft_inds: {len(aft_inds)}')
# sel_aft_inds = [val for val in aft_inds if 17 * 12 <= val[0] <= 19 * 12]
# sel_n = 30
# if sel_n > len(sel_aft_inds):
#     raise ValueError("n cannot be greater than the length of the list")
# else:
#     indices = np.arange(len(sel_aft_inds))
#     sampled_indices = np.random.choice(indices, sel_n, replace=False)
#     sel_aft_inds = [sel_aft_inds[i] for i in sampled_indices]
# # aft_inds = []
# for ind in range(len(dep_gap_list)):
#     if dep_idx_list[ind] not in sel_aft_inds:
#         i, j = dep_idx_list[ind]
#         if test_case_25['mu_s_array'][i, j] - 1 * dep_gap_list[ind] >= 0:
#             test_case_25['mu_s_array'][i, j] -= 1 * dep_gap_list[ind]
#             test_case_25['mu_c_array'][i, j] += 1 * dep_gap_list[ind]
#             change_count += 1
#             assert test_case_25['mu_s_array'][i, j] == test_case_25['lambda_s_array'][i, j], (
#                 f'{test_case_25["mu_s_array"][i, j]}'
#                 f'{test_case_25["lambda_s_array"][i, j]}')
#     else:
#         i, j = dep_idx_list[ind]
#         old_val = test_case_25['mu_s_array'][i, j] + test_case_25['mu_c_array'][i, j]
#         test_case_25['mu_s_array'][i, j] = old_val * 2.5
#         test_case_25['mu_c_array'][i, j] = old_val - test_case_25['mu_s_array'][i, j]
#         change_count += 1
#         # if test_case_25['mu_s_array'][i, j] + 4 * dep_gap_list[ind] <= old_val:
#         #     test_case_25['mu_s_array'][i, j] += 4 * dep_gap_list[ind]
#         #     test_case_25['mu_c_array'][i, j] -= 4 * dep_gap_list[ind]
#         #     change_count += 1
#         # elif test_case_25['mu_s_array'][i, j] + 3 * dep_gap_list[ind] <= old_val:
#         #     test_case_25['mu_s_array'][i, j] += 3 * dep_gap_list[ind]
#         #     test_case_25['mu_c_array'][i, j] -= 3 * dep_gap_list[ind]
#         #     change_count += 1
#         # elif test_case_25['mu_s_array'][i, j] + 2 * dep_gap_list[ind] <= old_val:
#         #     test_case_25['mu_s_array'][i, j] += 2 * dep_gap_list[ind]
#         #     test_case_25['mu_c_array'][i, j] -= 2 * dep_gap_list[ind]
#         #     change_count += 1
#         # elif test_case_25['mu_s_array'][i, j] + 1 * dep_gap_list[ind] <= old_val:
#         #     test_case_25['mu_s_array'][i, j] += 1 * dep_gap_list[ind]
#         #     test_case_25['mu_c_array'][i, j] -= 1 * dep_gap_list[ind]
#         #     change_count += 1
# print(f'change rate: {change_count / total_count}')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x_list = [i/2 for i in range(48)]
    # for i in range(25):
    #     plt.figure(figsize=(8, 5), dpi=150)
    #     lambda_s_list = list(test_case_25['lambda_s_array'][:, i])
    #     mu_s_list = list(test_case_25['mu_s_array'][:, i])
    #     # 每两项求和
    #     lambda_s_list = [sum(lambda_s_list[i:i+6]) for i in range(0, len(lambda_s_list), 6)]
    #     mu_s_list = [sum(mu_s_list[i:i+6]) for i in range(0, len(mu_s_list), 6)]
    #     plt.plot(x_list, lambda_s_list, label='lambda_s')
    #     # plt.plot(test_case_25['lambda_c_array'][:, i], label='lambda_c')
    #     plt.plot(x_list, mu_s_list, label='mu_s')
    #     # plt.plot(test_case_25['mu_c_array'][:, i], label='mu_c')
    #     plt.legend()
    #     plt.title(f'Station {i+1}')
    #     plt.show()
    # draw the sum of all stations
    plt.figure(figsize=(8, 5), dpi=150)
    lambda_s_list = list(np.sum(test_case_25['lambda_s_array'], axis=1))
    lambda_c_list = list(np.sum(test_case_25['lambda_c_array'], axis=1))
    mu_s_list = list(np.sum(test_case_25['mu_s_array'], axis=1))
    mu_c_list = list(np.sum(test_case_25['mu_c_array'], axis=1))
    # 每两项求和
    lambda_s_list = [sum(lambda_s_list[i:i + 6]) for i in range(0, len(lambda_s_list), 6)]
    mu_s_list = [sum(mu_s_list[i:i + 6]) for i in range(0, len(mu_s_list), 6)]
    lambda_c_list = [sum(lambda_c_list[i:i + 6]) for i in range(0, len(lambda_c_list), 6)]
    mu_c_list = [sum(mu_c_list[i:i + 6]) for i in range(0, len(mu_c_list), 6)]
    plt.plot(x_list, lambda_s_list, label='lambda_s')
    plt.plot(x_list, [lambda_s_list[i] + lambda_c_list[i] for i in range(len(lambda_s_list))], label='lambda_total')
    plt.plot(x_list, mu_s_list, label='mu_s')
    plt.plot(x_list, [mu_s_list[i] + mu_c_list[i] for i in range(len(mu_s_list))], label='mu_total')
    plt.legend()
    plt.title(f'Station 1-25')
    plt.show()
