import pickle
import random
import time
from collections import Counter
from queue import PriorityQueue
import numpy as np

from order_preprocessing.sim.init import get_init_station


class Order_Simulation:
    def __init__(self, stations: dict, dist_array: np.ndarray, arr_s_array: np.ndarray,
                 arr_c_array: np.ndarray, dep_array: np.ndarray, lam_array: np.ndarray,
                 des_dict: dict, des_prob_dict: dict):
        """
        Order_Simulation类，生成按比例分配的单量

        :param stations: station dict
        :param dist_array: distance matrix
        :param arr_s_array: arrival rate for orders (out, self)
        :param arr_c_array: arrival rate for orders (out, opponent)
        :param dep_array: arrival rate for orders (out)
        :param lam_array: dep rate for orders (indexed by time and station, in)
        :param des_dict: dep destination for orders (indexed by time and station, in)
        :param des_prob_dict: dep probability for orders (indexed by time and station, in)
        """

        # system const
        self.dist = dist_array

        # station dictionary
        self.stations = stations

        # simulation duration (1min per unit)
        self.sim_start_time = 0
        self.sim_end_time = 24 * 60

        # simulation const (1min per unit)
        self.min_step = 5

        # simulation variable
        self.t = self.sim_start_time  # system time

        # log
        self.success = 0
        self.success_opponent = 0
        self.full_loss = 0

        # queue for orders
        self.order_queue = PriorityQueue()  # (arr_t, end_loc, num, belong(1: self, 0: oppo))

        # arr and dep array for out_orders
        self.arr_s_array = arr_s_array
        self.arr_c_array = arr_c_array
        self.dep_array = dep_array

        # inner order generation
        self.lam_array = lam_array
        self.des_dict = des_dict
        self.des_prob_dict = des_prob_dict

        # arr & dep number recording
        self.arr_s_num = 0
        self.dep_s_num = 0
        self.arr_c_num = 0
        self.dep_c_num = 0

        # array generation
        self.arr_s_list = []
        self.arr_c_list = []
        self.dep_s_list = []
        self.dep_c_list = []

    def get_incluster_orders(self) -> dict:
        """to generate number of arrival"""
        # lam = lam_array[self.t/self.min_step, stations] notice: this is inner orders
        # des_list = des_dict[(self.t/self.min_step, stations)] notice: return a list
        # des_prob_list = des_prob_dict[(self.t/self.min_step, stations)]
        order_dict = {}
        for station in self.stations.keys():
            lam = self.lam_array[int(self.t / self.min_step / 3), station - 1]
            dep_num = np.random.poisson(lam)
            if dep_num > 0.01:
                start_list = []
                for _ in range(dep_num):
                    des = np.random.choice(
                        a=self.des_dict[(int(self.t / self.min_step / 3), station - 1)],
                        p=self.des_prob_dict[(int(self.t / self.min_step / 3), station - 1)]
                    )  # station_list.index
                    start_list.append(des+1)
                order_dict[station] = start_list

        return order_dict
        # return dict
        # start_loc: [end_loc1, end_loc2, end_loc3, ...]
        # return

    def generate_orders(self):
        """
        生成 time min 内订单

        :return: list(num_change_list)
        """
        num_change_list, arr_s_list, arr_c_list, dep_s_list, dep_c_list = [], [], [], [], []

        in_orders = self.get_incluster_orders()  # dep at this time
        # first in
        arr_s_dict, arr_c_dict = {}, {}
        while not self.order_queue.empty():
            arr_t = self.order_queue.queue[0][0]
            if self.t >= arr_t:
                # (arr_t, end_loc, num, belong(1: self, 0: oppo))
                arr_info = self.order_queue.get()
                if arr_info[3] > 0.9:
                    if arr_info[1] in arr_s_dict.keys():
                        arr_s_dict[arr_info[1]] += arr_info[2]
                    else:
                        arr_s_dict[arr_info[1]] = arr_info[2]
                else:
                    if arr_info[1] in arr_c_dict.keys():
                        arr_c_dict[arr_info[1]] += arr_info[2]
                    else:
                        arr_c_dict[arr_info[1]] = arr_info[2]
            else:
                break

        for station in self.stations.keys():
            arr_s, arr_c = np.random.poisson(self.arr_s_array[int(self.t / self.min_step / 3), station - 1]), \
                           np.random.poisson(self.arr_c_array[int(self.t / self.min_step / 3), station - 1])

            arr_s += arr_s_dict.get(station, 0)
            arr_c += arr_c_dict.get(station, 0)

            num_s = int(min((arr_s + self.stations[station].num_self), self.stations[station].cap))
            num_c = int(min((arr_c + self.stations[station].num_opponent), self.stations[station].cap_opponent))

            # record arrival
            arr_s_list.append(num_s - self.stations[station].num_self)
            arr_c_list.append(num_c - self.stations[station].num_opponent)

            # next out
            out_dep = np.random.poisson(self.dep_array[int(self.t / self.min_step/3), station - 1])
            in_dep = in_orders.get(station, -1)
            if isinstance(in_dep, list):
                in_dep = list(in_dep)
            else:
                in_dep = []
            des_list = [-1 for _ in range(out_dep)] + in_dep
            bike_list = [1 for _ in range(num_s)] + [0 for _ in range(num_c)]
            suc_num = min(len(des_list), len(bike_list))
            if suc_num > 0:
                dep, bike = random.sample(des_list, suc_num), random.sample(bike_list, suc_num)
                dep_s, dep_c = sum(bike), suc_num - sum(bike)

                # dep_s
                dep_s_list.append(dep_s)
                dep_c_list.append(dep_c)

                use_pair = zip(dep, bike)  # (-1, 1)
                counter = Counter(use_pair)
                for pair in list(counter):
                    if pair[0] > 0:
                        # (arr_t, end_loc, num, belong(1: self, 0: oppo))
                        assert self.dist[station - 1, pair[0] - 1] > -0.01
                        self.order_queue.put(
                            (self.t + self.dist[station - 1, pair[0] - 1], pair[0], counter[pair], pair[1])
                        )
                num_change_list.append(
                    (num_s - dep_s - self.stations[station].num_self,
                     num_c - dep_c - self.stations[station].num_opponent))
            else:
                num_change_list.append((0, 0))
                dep_s_list.append(0)
                dep_c_list.append(0)

        return list(num_change_list), list(arr_s_list), list(arr_c_list), list(dep_s_list), list(dep_c_list)

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
            self.stations[station].change_num(num_change_list[station - 1])

    def step(self):
        """
        步进函数，前进一个最小步（5min）

        :return:
        """

        num_change_list, arr_s_list, arr_c_list, dep_s_list, dep_c_list = self.generate_orders()

        # record arr & dep
        arr_s = sum(arr_s_list)
        self.arr_s_num += arr_s
        arr_c = sum(arr_c_list)
        self.arr_c_num += arr_c
        dep_s = sum(dep_s_list)
        self.dep_s_num += dep_s
        dep_c = sum(dep_c_list)
        self.dep_c_num += dep_c

        # record arr & dep array
        self.arr_s_list.append(arr_s_list)
        self.arr_c_list.append(arr_c_list)
        self.dep_s_list.append(dep_s_list)
        self.dep_c_list.append(dep_c_list)

        # num_change
        self.apply_num_change(num_change_list)
        # step forward
        self.t += self.min_step


def load_data() -> dict:
    split_day = 3
    # distance_matrix
    with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\distance_matrix.pkl', 'rb') as file:
        distance_matrix = pickle.load(file)
    distance_matrix = np.floor(distance_matrix/5) * 5
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            if distance_matrix[i, j] == 0:
                distance_matrix[i, j] += 5
    # arr_s_array
    with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\arr_s_array.pkl', 'rb') as file:
        arr_s_array = pickle.load(file)
        # dim as 5 days
        arr_s_array = arr_s_array / split_day
    # arr_c_array
    with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\arr_c_array.pkl', 'rb') as file:
        arr_c_array = pickle.load(file)
        arr_c_array = arr_c_array / split_day
    # dep_array
    with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\dep_array.pkl', 'rb') as file:
        dep_array = pickle.load(file)
        dep_array = dep_array / split_day
    # lam_array
    with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\lam_array.pkl', 'rb') as file:
        lam_array = pickle.load(file)
        lam_array = lam_array / split_day
    # des_dict
    with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\des_dict.pkl', 'rb') as file:
        des_dict = pickle.load(file)
    # des_prob_dict
    with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\data\des_prob_dict.pkl', 'rb') as file:
        des_prob_dict = pickle.load(file)

    return {
        'dist_array': distance_matrix,
        'arr_s_array': arr_s_array,
        'arr_c_array': arr_c_array,
        'dep_array': dep_array,
        'lam_array': lam_array,
        'des_dict': des_dict,
        'des_prob_dict': des_prob_dict,
    }


if __name__ == '__main__':

    data = load_data()
    '''
    stations = get_init_station()
    data['stations'] = stations

    station_num = len(stations.keys())

    arr_s_array, arr_c_array, dep_s_array, dep_c_array = \
        np.zeros((288, station_num)), np.zeros((288, station_num)), \
        np.zeros((288, station_num)), np.zeros((288, station_num))

    rep = 10000
    test = 0
    st = time.process_time()
    for _ in range(rep):
        problem = Order_Simulation(**data)
        problem.run()
        arr_s_array += np.array(problem.arr_s_list)
        arr_c_array += np.array(problem.arr_c_list)
        dep_s_array += np.array(problem.dep_s_list)
        dep_c_array += np.array(problem.dep_c_list)

        if _ % 1000 == 0:
            print(f'finish {_}/10000.')

    en = time.process_time()
    print('Running time: %s Seconds' % (en - st))

    arr_s_array /= rep
    arr_c_array /= rep
    dep_s_array /= rep
    dep_c_array /= rep

    # dump information
    with open("arr_s_array.pkl", "wb") as f:
        pickle.dump(arr_s_array, f)
    with open("arr_c_array.pkl", "wb") as f:
        pickle.dump(arr_c_array, f)
    with open("dep_s_array.pkl", "wb") as f:
        pickle.dump(dep_s_array, f)
    with open("dep_c_array.pkl", "wb") as f:
        pickle.dump(dep_c_array, f)
    '''

