import pickle
import random
import time

import numpy as np

from sim.consts import *
from order_preprocessing.sim.init import get_init_station


class Order_Simulation:
    def __init__(self, stations: dict, dist_array: np.ndarray, arr_s_array: np.ndarray, dep_s_array: np.ndarray):
        """
        Order_Simulation类，生成按比例分配的单量

        :param stations: station dict
        :param dist_array: distance matrix
        :param arr_s_array: arrival rate for orders (self)
        :param dep_s_array: dep rate for orders (self)
        """

        # system const
        self.dist = dist_array

        # station dictionary
        self.stations = stations

        # simulation duration (1min per unit)
        self.sim_start_time = 5 * 60
        self.sim_end_time = 22 * 60

        # simulation const (1min per unit)
        self.min_step = 5

        # simulation variable
        self.t = self.sim_start_time  # system time

        # number record
        self.success = 0
        self.full_loss = 0
        self.empty_loss = 0

        self.work_success = 0
        self.work_full_loss = 0
        self.work_empty_loss = 0

        # arr and dep array for out_orders
        self.arr_s_array = arr_s_array
        self.dep_s_array = dep_s_array

        # arr & dep number recording
        self.arr_s_num = 0
        self.dep_s_num = 0

        # array generation
        self.arr_s_list = []
        self.dep_s_list = []

    @property
    def self_list(self):
        return [station.num_self for station in self.stations.values()]

    def generate_orders(self):
        """
        生成 time min 内订单

        :return: list(num_change_list)
        """
        num_change_list, arr_s_list, dep_s_list = [], [], []

        for station in self.stations.keys():
            # first arrive
            arr_s = np.random.poisson(self.arr_s_array[int(self.t / self.min_step / 3), station - 1])
            num_s = int(min((arr_s + self.stations[station].num_self), self.stations[station].cap))
            if num_s == self.stations[station].cap:
                self.full_loss += (arr_s + self.stations[station].num_self + arr_s - num_s)

                if self.t >= RE_START_T:
                    self.work_full_loss += (arr_s + self.stations[station].num_self + arr_s - num_s)

            # record arrival
            arr_s_list.append(num_s - self.stations[station].num_self)

            # next departure
            dep_s = np.random.poisson(self.dep_s_array[int(self.t / self.min_step/ 3), station - 1])
            suc_num = min(num_s, dep_s)
            self.success += suc_num
            if self.t >= RE_START_T:
                self.work_success += suc_num
            if num_s < dep_s:
                self.empty_loss += (dep_s - num_s)
                if self.t >= RE_START_T:
                    self.work_empty_loss += (dep_s - num_s)

            if suc_num > 0.01:
                dep_s_list.append(suc_num)
                num_change_list.append(num_s - suc_num - self.stations[station].num_self)
            else:
                num_change_list.append(num_s - self.stations[station].num_self)
                dep_s_list.append(0)

            assert arr_s_list[-1] - dep_s_list[-1] == num_change_list[-1], f'{(arr_s_list[-1], dep_s_list[-1], num_change_list[-1])}'

        return list(num_change_list), list(arr_s_list), list(dep_s_list)

    def run(self):
        """
        仿真运行主函数

        :return:
        """
        # change stage_info and simulation log
        # self._log.append(self.simulation_log_format(self.stations))

        # start simulation
        while self.t < self.sim_end_time:
            self.step()

    def apply_num_change(self, num_change_list):
        for station in self.stations.keys():
            self.stations[station].num_self += num_change_list[station - 1]
            assert self.stations[station].num_self >= 0

    def step(self):
        """
        步进函数，前进一个最小步（5min）

        :return:
        """

        num_change_list, arr_s_list, dep_s_list = self.generate_orders()

        # record arr & dep
        arr_s = sum(arr_s_list)
        self.arr_s_num += arr_s
        dep_s = sum(dep_s_list)
        self.dep_s_num += dep_s

        # record arr & dep array
        self.arr_s_list.append(arr_s_list)
        self.dep_s_list.append(dep_s_list)

        # num_change
        self.apply_num_change(num_change_list)
        # step forward
        self.t += self.min_step


def load_data() -> dict:
    split_day = 3
    # distance_matrix
    with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\distance_matrix.pkl', 'rb') as file:
        distance_matrix = pickle.load(file)
    distance_matrix = np.floor(distance_matrix/5) * 5  # in minute
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if distance_matrix[i, j] == 0:
                distance_matrix[i, j] += 5
    # arr_s_array
    with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\order_preprocessing\arr_s_array.pkl', 'rb') as file:
        arr_s_array = pickle.load(file)
        # dim as 5 days
        arr_s_array = arr_s_array / split_day

    # dep_s_array
    with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\order_preprocessing\dep_s_array.pkl', 'rb') as file:
        dep_s_array = pickle.load(file)
        dep_s_array = dep_s_array / split_day

    reduction_rate = 1

    return {
        'dist_array': distance_matrix,
        'arr_s_array': arr_s_array / reduction_rate,
        'dep_s_array': dep_s_array,
    }


if __name__ == '__main__':

    data = load_data()

    stations = get_init_station()
    data['stations'] = stations

    station_num = len(stations.keys())

    avg_order, avg_work_order = 0, 0
    avg_loss, avg_work_loss = 0, 0
    avg_full, avg_work_full = 0, 0

    rep = 10
    test = 0
    st = time.process_time()
    for _ in range(rep):
        problem = Order_Simulation(**data)
        problem.run()
        avg_order += problem.success
        avg_loss += problem.empty_loss
        avg_full += problem.full_loss
        avg_work_order += problem.work_success
        avg_work_loss += problem.work_empty_loss
        avg_work_full += problem.work_full_loss

    en = time.process_time()
    print('Running time: %s Seconds' % (en - st))

    avg_order /= rep
    avg_loss /= rep
    avg_full /= rep
    avg_work_order /= rep
    avg_work_loss /= rep
    avg_work_full /= rep
