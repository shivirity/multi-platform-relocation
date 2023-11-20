import numpy as np
import pandas as pd
import pickle

from simulation.consts import CAP_S, CAP_C
from simulation.system import Station


def get_init_station() -> dict:

    NUM_BIKES = 3000  # 3900

    with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\station_list.pkl', 'rb') as file:
        station_list = pickle.load(file)
    with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\related_order.pkl', 'rb') as file:
        related_order = pickle.load(file)

    # by historical orders

    s_list, c_list = get_init_distribution(station_list=station_list, orders=related_order)
    sum_s, sum_c = sum(s_list), sum(c_list)
    sum_total = sum_s + sum_c
    s_list = [int(val * NUM_BIKES / sum_total) + 1 for val in s_list]
    c_list = [int(val * NUM_BIKES / sum_total) + 1 for val in c_list]


    # mean distribution
    '''
    s_list = [int(NUM_BIKES / len(station_list) / 3) for _ in range(len(station_list))]
    c_list = [int(NUM_BIKES / len(station_list) * 2 / 3) for _ in range(len(station_list))]
    '''

    # for order simulation
    '''
    tmp = {
        i: Station(station_id=i, location=(i, i), capacity=50, capacity_opponent=100,
                   num_self=max(s_list[i-1]+np.random.randint(low=-5, high=6), 0),
                   num_opponent=max(c_list[i-1]+np.random.randint(low=-5, high=6), 0))
        for i in range(1, 108)
    }
    '''

    # for real test
    tmp = {
        i: Station(station_id=i, location=(i, i), capacity=CAP_S, capacity_opponent=CAP_C,
                   num_self=max(s_list[i-1], 0),
                   num_opponent=max(c_list[i-1], 0))
        for i in range(1, 108)
    }

    return tmp


def get_init_distribution(station_list: list, orders: pd.DataFrame):
    # orders = orders.sort_values(by="START_STEP", ascending=True).reset_index(drop=True)
    s_list, c_list = [], []
    com = 'hello'
    station_list = list(station_list)
    arr_s, arr_c = np.zeros((96, len(station_list))), np.zeros((96, len(station_list)))
    former_start_t = -1
    for _, record in orders.iterrows():
        bool_self = (com in record['CompanyId'])
        start_t, end_t, start, end = record['START_STEP'], record['END_STEP'], \
                                     record['start_station_id'], record['end_station_id']
        '''
        if start_t < former_start_t:
            break
        '''
        # assert start_t >= former_start_t, f'{start_t, former_start_t}'
        former_start_t = start_t
        if start not in station_list:
            end_idx = station_list.index(end)
            # end in station_list
            if bool_self:
                # hello bike
                arr_s[end_t:, end_idx] += 1
            else:
                # others
                arr_c[end_t:, end_idx] += 1
        else:
            if end not in station_list:
                start_idx = station_list.index(start)
                if bool_self:
                    arr_s[start_t:, start_idx] -= 1
                else:
                    arr_c[start_t:, start_idx] -= 1
            else:
                start_idx = station_list.index(start)
                end_idx = station_list.index(end)
                if bool_self:
                    arr_s[start_t:, start_idx] -= 1
                    arr_s[end_t:, end_idx] -= 1
                else:
                    arr_c[start_t:, start_idx] -= 1
                    arr_c[end_t:, end_idx] -= 1

    for station in range(len(station_list)):
        s_list.append(-min(list(arr_s[:, station])))
        c_list.append(-min(list(arr_c[:, station])))

    s_list = [max(val, 0) for val in s_list]
    c_list = [max(val, 0) for val in c_list]

    return s_list, c_list


if __name__ == '__main__':

    with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\station_list.pkl', 'rb') as file:
        station_list = pickle.load(file)
    with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\related_order.pkl', 'rb') as file:
        related_order = pickle.load(file)

    num_of_bikes = 5000

    s_list, c_list = get_init_distribution(station_list=station_list, orders=related_order)
    sum_s, sum_c = sum(s_list), sum(c_list)
    sum_total = sum_s + sum_c
    s_list = [int(val * num_of_bikes / sum_total) + 1 for val in s_list]
    c_list = [int(val * num_of_bikes / sum_total) + 1 for val in c_list]
