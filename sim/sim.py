import copy
import random
import numpy as np

from sim.consts import *

random.seed(SEED)
np.random.seed(SEED)


class Simulation:
    def __init__(self, stations: dict, dist_array: np.ndarray,
                 mu_array: np.ndarray, lambda_s_array: np.ndarray, lambda_c_array: np.ndarray):
        """
        Simulation类

        :param stations: dict of Station object
        :param dist_array: distance matrix
        :param mu_array: demand(departing) rate for every time_idx and station
        :param lambda_s_array: supply(arriving) rate for every time_idx and station from platform-self
        :param lambda_c_array: supply(arriving) rate for every time_idx and station from platform-opponent
        """
        # system const
        self.mu_array = mu_array
        self.lambda_s_array = lambda_s_array
        self.lambda_c_array = lambda_c_array
        self.dist = dist_array

        # station dictionary
        self.stations = stations

        # simulation duration (1min per unit)
        self.sim_start_time = SIM_START_T
        self.sim_end_time = SIM_END_T

        # simulation variable
        self.t = 0  # system time
        self.last_t = 0  # latest log time

        # stage variable
        self.stage = 0
        self.stage_info = []

        # relocation vehicle variable
        # current_loc, next_loc, load
        self.veh_info = [0, None, 0]

        # system performance
        self.success = 0  # number of successful orders
        self.success_list = []
        self.full_loss = 0  # number of orders that lost because of full station
        self.full_loss_list = []

        # log
        self._log = []

    @property
    def log(self):
        return self._log

    @staticmethod
    def stage_info_format(stage, time, veh_loc, veh_load):
        return {'stage': stage, 'time': time, 'veh_loc': veh_loc, 'veh_load': veh_load}

    def simulation_log_format(self, stations_dict):
        new_log = {'t': self.t}
        for key, value in stations_dict.items():
            new_log[key] = (value.num_self, value.num_opponent)
        return new_log

    def decide_inventory(self):
        return -1

    def decide_route(self):
        return 0

    def decide_time(self, route_dec: int):
        """
        根据下一站决策，返回step的时长

        :param route_dec:
        :return: step的时长(int)
        """
        current_station = self.veh_info[0]
        if route_dec == current_station:  # stay at current station
            return STAY_TIME
        else:
            return self.dist[current_station, route_dec]

    def apply_decision(self, inv: int, route: int):
        pass

    def generate_orders(self):
        """
        生成订单

        :return:
        """
        num_change_list, success_list, full_list = [], [], []
        for station in self.stations.keys():
            arr_s, arr_c = np.random.poisson(self.lambda_s_array[int(self.t), station-1]), \
                           np.random.poisson(self.lambda_c_array[int(self.t), station-1])
            dep = np.random.poisson(self.mu_array[int(self.t), station-1])
            if arr_s + self.stations[station].num_self >= self.stations[station].cap:
                num_s = self.stations[station].cap
                full_list.append(arr_s + self.stations[station].num_self - self.stations[station].cap)
            else:
                num_s = arr_s + self.stations[station].num_self
                full_list.append(0)
            num_c = max(self.stations[station].num_opponent + arr_c, self.stations[station].cap_opponent)
            bike_list = [1 for _ in range(num_s)] + [0 for _ in range(num_c)]
            if len(bike_list) >= dep:
                dep_s = sum(random.sample(bike_list, dep))
                dep_c = dep - dep_s
            else:
                dep_s = self.stations[station].num_self
                dep_c = self.stations[station].num_opponent
            success_list.append(dep_s)
            num_change_list.append(
                (num_s-dep_s-self.stations[station].num_self, num_c-dep_c-self.stations[station].num_opponent))
        return list(num_change_list), list(success_list), list(full_list)

    def apply_num_change(self, num_change_list):
        for station in self.stations.keys():
            self.stations[station].change_num(num_change_list[station-1])

    def step(self, end_t: int):
        """
        步进函数

        :return:
        """
        end_t += self.t
        while self.t < end_t:
            # simulation log for current time
            self._log.append(self.simulation_log_format(self.stations))
            num_change_list, success_list, full_list = self.generate_orders()
            # num_change
            self.apply_num_change(num_change_list)
            # success_record
            sum_success = sum(success_list)
            self.success += sum_success
            self.success_list.append(sum_success)
            # full_loss_record
            sum_full_loss = sum(full_list)
            self.full_loss += sum_full_loss
            self.full_loss_list.append(sum_full_loss)
            # step forward
            self.t += MIN_STEP
        # before operation
        self.veh_info[0] = self.veh_info[1]  # current_loc = next_loc
        self._log.append(self.simulation_log_format(self.stations))

    def run(self):
        """
        仿真运行主函数

        :return:
        """
        # change stage_info and simulation log
        self.stage_info.append(self.stage_info_format(0, 0, 0, 0))
        self._log.append(self.simulation_log_format(self.stations))

        # start simulation
        while self.t < self.sim_end_time:

            if self.t:
                self.stage += 1
            self.stage_info.append(
                self.stage_info_format(stage=self.stage, time=self.t, veh_loc=self.veh_info[0], veh_load=self.veh_info[2]))
            # decisions at current stage
            inv_dec = self.decide_inventory()  # [10, 20, 30, 40, -1]
            route_dec = self.decide_route()  # 下一前往站点
            t_dec = self.decide_time(route_dec)  # 向前步进若干步，单位：min

            # change next_loc and load in apply_decision
            # self.apply_decision(inv=inv_dec, route=route_dec)  # 这里改车辆状态
            self.veh_info[1] = route_dec

            self.stage_info.append(
                self.stage_info_format(stage=self.stage, time=self.t, veh_loc=self.veh_info[0], veh_load=self.veh_info[2]))

            self.step(end_t=t_dec)

    def print_simulation_log(self):
        f = open("simulation_log.txt", "w")
        for line in self.log:
            f.write(str(line) + '\n')
        f.close()

    def print_stage_log(self):
        f = open("stage_log.txt", "w")
        for line in self.stage_info:
            f.write(str(line) + '\n')
        f.close()
