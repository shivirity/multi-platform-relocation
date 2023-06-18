import time
import random
import numpy as np


class Order_Simulation:
    def __init__(self, stations: dict, dist_array: np.ndarray, arr_array: np.ndarray, prob_array: np.ndarray):
        """
        Order_Simulation类，生成按比例分配的单量

        :param stations: dict of Station object
        :param dist_array: distance matrix
        """
        # system const
        self.dist = dist_array
        self.arr = arr_array
        self.prob = prob_array

        # station dictionary
        self.stations = stations

        # simulation duration (1min per unit)
        self.sim_start_time = 0
        self.sim_end_time = 12 * 60

        # simulation const (1min per unit)
        self.min_step = 5

        # simulation variable
        self.t = self.sim_start_time  # system time

        # log
        self._log = []
        self.print_log = True
        self.success = 0
        self.success_opponent = 0
        self.full_loss = 0

    def generate_orders(self):
        """
        生成 time min 内订单

        :return: list(num_change_list), list(success_list), list(success_opponent_list), list(full_list)
        """
        # todo 加栈控制订单到达
        num_change_list, success_list, success_opponent_list, full_list = [], [], [], []
        for station in self.stations.keys():
            arr_s, arr_c = np.random.poisson(self.lambda_s_array[int(self.t), station-1]*gene_t/MIN_STEP), \
                           np.random.poisson(self.lambda_c_array[int(self.t), station-1]*gene_t/MIN_STEP)
            dep = np.random.poisson(self.mu_array[int(self.t), station-1]*gene_t/MIN_STEP)
            if arr_s + self.stations[station].num_self > self.stations[station].cap:
                num_s = self.stations[station].cap
                full_list.append(arr_s + self.stations[station].num_self - self.stations[station].cap)
            else:
                num_s = int(arr_s + self.stations[station].num_self)
                full_list.append(0)
            num_c = int(min(self.stations[station].num_opponent + arr_c, self.stations[station].cap_opponent))
            bike_list = [1 for _ in range(num_s)] + [0 for _ in range(num_c)]
            if len(bike_list) >= dep:
                dep_s = sum(random.sample(bike_list, dep))
                dep_c = dep - dep_s
            else:
                dep_s = self.stations[station].num_self
                dep_c = self.stations[station].num_opponent
            success_list.append(dep_s)
            success_opponent_list.append(dep_c)
            num_change_list.append(
                (num_s-dep_s-self.stations[station].num_self, num_c-dep_c-self.stations[station].num_opponent))
        return list(num_change_list), list(success_list), list(success_opponent_list), list(full_list)

    def run(self):
        """
        仿真运行主函数

        :return:
        """
        # change stage_info and simulation log
        self._log.append(self.simulation_log_format(self.stations))

        # start simulation
        while self.t < self.sim_end_time:
            self.step()

    def apply_num_change(self, num_change_list):
        for station in self.stations.keys():
            self.stations[station].change_num(num_change_list[station-1])

    def step(self):
        """
        步进函数，前进一个最小步（5min）

        :return:
        """

        while self.t < self.sim_end_time:
            # simulation log for current time
            num_change_list, success_list, success_opponent_list, full_list = self.generate_orders()
            # num_change
            self.apply_num_change(num_change_list)
            # success_record
            sum_success = sum(success_list)
            self.success += sum_success
            # success_opponent_record
            sum_success_oppo = sum(success_opponent_list)
            self.success_opponent += sum_success_oppo
            # full_loss_record
            sum_full_loss = sum(full_list)
            self.full_loss += sum_full_loss
            # step forward
            self.t += self.min_step

        # write log
        self._log.append(self.simulation_log_format(self.stations))


if __name__ == '__main__':
    pass
