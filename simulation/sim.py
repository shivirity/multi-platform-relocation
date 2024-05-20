import copy
import logging
import pickle
import random
import time
import numpy as np

from simulation.consts import *

import os
import sys

# sys.path.append(rf'{os.path.abspath(os.path.join(os.getcwd(), "../.."))}\ALNS-for-multi-platform-relocation')
# sys.path.append(rf'route_extension\bph')

# from alns import get_relocation_routes
from route_extension.route_extension_algo import get_REA_routes_test
from route_extension.cg_re_algo import get_CG_REA_routes, get_DP_routes_greedy, get_exact_routes
from route_extension.bph.BaP_algo import get_routes_branch_and_price

random.seed(SEED)
np.random.seed(SEED)


class Simulation:
    def __init__(self, stations: dict, dist_array: np.ndarray,
                 mu_s_array: np.ndarray, mu_c_array: np.ndarray, lambda_s_array: np.ndarray,
                 lambda_c_array: np.ndarray, **kwargs):
        """
        Simulation类.

        :param stations: dict of Station object
        :param dist_array: distance matrix
        :param mu_s_array: demand(departing) rate for every time_idx and station from platform-self
        :param mu_c_array: demand(departing) rate for every time_idx and station from platform-opponent
        :param lambda_s_array: supply(arriving) rate for every time_idx and station from platform-self
        :param lambda_c_array: supply(arriving) rate for every time_idx and station from platform-opponent
        """
        # system const
        self.mu_s_array = mu_s_array
        self.mu_c_array = mu_c_array
        self.mu_array = mu_s_array + mu_c_array
        self.lambda_s_array = lambda_s_array
        self.lambda_c_array = lambda_c_array
        self.dist = dist_array * DIST_FIX
        self.dist[0, 7] = 1 * DIST_FIX  # avoid distance equals 0
        self.dist[7, 0] = 1 * DIST_FIX  # avoid distance equals 0

        # station dictionary
        self.stations = stations

        # simulation duration (1min per unit)
        self.sim_start_time = SIM_START_T
        self.sim_end_time = SIM_END_T

        # simulation variable
        self.t = self.sim_start_time  # system time

        # stage variable
        self.stage = 0
        self.stage_info = []

        # relocation vehicle variable
        # current_loc, next_loc, time_left, load
        self.num_of_veh = NUM_VEHICLES
        self.veh_info = [[0, None, None, 0] for _ in range(self.num_of_veh)]  # vehicle info

        # system performance
        self.success = 0  # number of successful orders
        self.success_work = 0  # number of successful orders after relocation van starts to work till sim ends
        self.success_work_till_done = 0  # number of successful orders after vehicle starts to work till work done
        self.success_work_till_done_list = []
        self.success_list = []
        self.success_work_list = []
        self.success_opponent = 0  # number of successful orders from opponent platform
        self.success_opponent_list = []
        self.full_loss = 0  # number of orders that lost because of station is full
        self.full_loss_list = []
        self.empty_loss = 0  # number of orders that lost because of lack of bikes (single/both platforms)
        self.empty_loss_list = []

        self.veh_distance = [0 for _ in range(self.num_of_veh)]  # total distance of relocation vehicle
        self.return_count_time = [0 for _ in range(self.num_of_veh)]  # number of times that vehicle returns to depot

        # policy
        # single is True: 'None', 'STR', 'rollout', 'GLA', 'MINLP'
        # single is False: 'random', 'MINLP', 'REA_test', 'DP_test'
        self.policy = None
        self.single = kwargs['single'] if 'single' in kwargs.keys() else False
        # bool, False means decide with multi-information, True means decide with single-info

        # log
        self._log = []
        self.print_action = False  # print the action of relocation vehicle

        # for single rollout
        self.single_full_list = None  # record return loss for every station in single rollout
        self.single_empty_list = None  # record rental loss for every station in single rollout

        # offline training property
        # self.random_choice_to_init_B = False
        # self.cost_list = []
        # self.dec_time_list = []
        # self.basis_func_property = []
        # self.func_dict = kwargs['func_dict'] if 'func_dict' in kwargs.keys() else None
        # self.MLP_model = kwargs['MLP_model'] if 'MLP_model' in kwargs.keys() else None
        self.cost_after_work = 0
        # self.func_var_dict = self.init_func_var_dict()
        # online VFA property
        self.best_val_list = []

        # test esd
        self.test_esd = 0
        self.test_esd_till_work_done = 0

        # MINLP mode
        self.ei_s_arr = kwargs['ei_s_arr'] if 'ei_s_arr' in kwargs.keys() else None
        self.ei_c_arr = kwargs['ei_c_arr'] if 'ei_c_arr' in kwargs.keys() else None
        self.esd_arr = kwargs['esd_arr'] if 'esd_arr' in kwargs.keys() else None
        self.last_dec_t = None
        self.future_dec_dict = None

        # MLP test
        # self.nn_var_list = ['time', 'veh_load', 'des_inv']
        # for i in range(1, 26):
        #     self.nn_var_list.append(f'veh_loc_{i}')
        # for i in range(1, 26):
        #     self.nn_var_list.append(f'num_self_{i}')
        # for i in range(1, 26):
        #     self.nn_var_list.append(f'num_oppo_{i}')
        # for i in range(1, 26):
        #     self.nn_var_list.append(f'orders_till_sim_end_{i}')
        # for i in range(1, 26):
        #     self.nn_var_list.append(f'bikes_s_arr_till_sim_end{i}')
        # for i in range(1, 26):
        #     self.nn_var_list.append(f'bikes_c_arr_till_sim_end{i}')

    @property
    def self_list(self):
        return [station.num_self for station in self.stations.values()]

    @property
    def oppo_list(self):
        return [station.num_opponent for station in self.stations.values()]

    @property
    def log(self):
        return self._log

    @property
    def func_var_dict(self):
        """给offline_VFA中的每个变量定义唯一的名字"""
        var_dict = {
            'const': 1,  # const
            'veh_load': self.veh_info[2],  # load on the relocation vehicle
        }
        return var_dict

    @staticmethod
    def stage_info_format(stage: int, time: int, veh_loc: list, veh_next_loc: list, veh_load: list):
        return {'stage': stage, 'time': time, 'veh_loc': veh_loc, 'veh_next_loc': veh_next_loc, 'veh_load': veh_load}

    @staticmethod
    def get_station_inv(station_inv: int, inv_dec: int, load: int):
        """
        计算站点决策后库存（不包含depot）

        :param station_inv: 站点原有库存
        :param inv_dec: 站点库存决策
        :param load: vehicle运载量
        :return: 站内决策后库存
        """
        if inv_dec == -1:
            return station_inv
        else:
            if inv_dec > station_inv:
                ins = min(inv_dec - station_inv, load)
            elif inv_dec < station_inv:
                ins = max(inv_dec - station_inv, load - VEH_CAP)
            else:
                ins = 0
            return station_inv + ins

    def simulation_log_format(self, stations_dict):
        new_log = {'t': self.t}
        for key, value in stations_dict.items():
            new_log[key] = (value.num_self, value.num_opponent)
        return new_log

    def get_estimated_order(self, step_t: int, num_self: list, num_oppo: list, start_t: int) -> float:
        """
        返回当前动作的价值函数

        :param step_t: 向前估计的时间步长
        :param num_self: 初始状态的自身平台库存列表
        :param num_oppo: 初始状态的竞对平台库存列表
        :param start_t: 初始状态的时间
        :return:
        """
        tmp_t, order_exp = 0, 0
        if self.single is False:  # multi-platform info case
            num_self_list, num_oppo_list = list(num_self), list(num_oppo)
            while tmp_t < step_t:
                # arrive
                num_self_list = [
                    min(num_self_list[i - 1] + self.lambda_s_array[int((start_t + tmp_t) / MIN_STEP), i - 1],
                        self.stations[i].cap) for i in range(1, len(self.stations) + 1)]
                num_oppo_list = [
                    min(num_oppo_list[i - 1] + self.lambda_c_array[int((start_t + tmp_t) / MIN_STEP), i - 1],
                        self.stations[i].cap_opponent) for i in range(1, len(self.stations) + 1)]
                num_self_list_a, num_oppo_list_a = list(num_self_list), list(num_oppo_list)
                # departure
                num_self_list = [max(
                    num_self_list[i - 1] - num_self_list[i - 1] / (num_self_list[i - 1] + num_oppo_list[i - 1]) *
                    self.mu_array[int((start_t + tmp_t) / MIN_STEP), i - 1], 0)
                                 if num_self_list[i - 1] + num_oppo_list[i - 1] > 0 else num_self_list[i - 1]
                                 for i in range(1, len(self.stations) + 1)
                                 ]
                num_oppo_list = [max(
                    num_oppo_list[i - 1] - num_oppo_list[i - 1] / (num_self_list[i - 1] + num_oppo_list[i - 1]) *
                    self.mu_array[int((start_t + tmp_t) / MIN_STEP), i - 1], 0)
                                 if num_self_list[i - 1] + num_oppo_list[i - 1] > 0 else num_oppo_list[i - 1]
                                 for i in range(1, len(self.stations) + 1)
                                 ]
                order_exp += sum([num_self_list_a[i] - num_self_list[i] for i in range(len(num_self_list))])
                tmp_t += MIN_STEP
        else:  # single-platform info case
            num_self_list = list(num_self)
            while tmp_t < step_t:
                # arrive
                num_self_list = [
                    min(num_self_list[i - 1] + self.lambda_s_array[int((start_t + tmp_t) / MIN_STEP), i - 1],
                        self.stations[i].cap) for i in range(1, len(self.stations) + 1)]
                num_self_list_a = list(num_self_list)
                # departure
                num_self_list = [max(
                    num_self_list[i - 1] - self.mu_s_array[int((start_t + tmp_t) / MIN_STEP), i - 1], 0)
                    for i in range(1, len(self.stations) + 1)
                ]
                order_exp += sum([num_self_list_a[i] - num_self_list[i] for i in range(len(num_self_list))])
                tmp_t += MIN_STEP

        return order_exp

    def get_MINLP_dist_mat(self):
        """返回MINLP模型中的距离矩阵"""
        dist_mat = np.zeros((len(self.stations) + 1, len(self.stations) + 1))
        for i in range(dist_mat.shape[0]):
            for j in range(dist_mat.shape[1]):
                if i == 0:
                    dist_mat[i, j] = int((self.dist[0, j] - 0.2) / MIN_RUN_STEP) + 1 if self.dist[0, j] > 0 else 0
                else:
                    if i == j:
                        dist_mat[i, j] = 0
                    else:
                        dist_mat[i, j] = (int((self.dist[i, j] - 0.2) / MIN_RUN_STEP) + 1 if self.dist[
                                                                                                 i, j] > 0 else 0) + 1
        return dist_mat

    def decide_action_multi_info(self):
        """
        决策当前站点目标库存水平和下一站点决策

        :return: 决策字典, {'inv': inv_dec, 'route': route_dec}
        """
        # todo: 重构决策函数, self.veh_info
        # random
        if self.policy == 'random':
            dec_list = []
            for veh in range(self.num_of_veh):
                if self.veh_info[veh][2] == 0 or self.veh_info[veh][2] is None:  # time to decide
                    cur_station = self.veh_info[veh][0]
                    if cur_station:
                        inv_levels = [i * self.stations[cur_station].cap for i in DEC_LEVELS]
                        inv_tmp, inv_state = [], []
                        for i in range(len(inv_levels)):
                            if not i:
                                inv_tmp.append(inv_levels[i])
                                inv_state.append(
                                    self.get_station_inv(self.stations[cur_station].num_self, inv_levels[i], self.veh_info[veh][3]))
                            else:
                                inv_state_tmp = self.get_station_inv(
                                    self.stations[cur_station].num_self, inv_levels[i], self.veh_info[veh][3])
                                if inv_state_tmp not in inv_state:
                                    inv_tmp.append(inv_levels[i])
                                    inv_state.append(inv_state_tmp)
                        inv_levels = inv_tmp
                        inv_dec = random.sample(inv_levels, 1)[0]
                    else:
                        inv_dec = -1  # happens if the vehicle is at depot
                    route_dec = random.sample([i for i in self.stations.keys() if i != cur_station], 1)[0]
                    dec_list.append({'inv': inv_dec, 'route': route_dec})
                else:
                    dec_list.append({'inv': None, 'route': None})

        elif self.policy == 'MINLP':

            if self.last_dec_t is None:  # at depot
                assert self.t == RE_START_T
                self.last_dec_t = self.t  # 第一次决策
                # closest to the planned amount of loading/unloading
                self.future_dec_dict, _, __ = get_relocation_routes(
                    num_of_van=1,
                    van_location=[0],
                    van_dis_left=[0],
                    van_load=[0],
                    c_s=CAP_S,
                    c_v=VEH_CAP,
                    cur_t=round(self.t / MIN_RUN_STEP),
                    t_p=round(T_PLAN / MIN_RUN_STEP),
                    t_f=round(T_FORE / MIN_RUN_STEP),
                    t_roll=round(T_ROLL / MIN_RUN_STEP),
                    c_mat=self.get_MINLP_dist_mat(),
                    ei_s_arr=self.ei_s_arr,
                    ei_c_arr=self.ei_c_arr,
                    esd_arr=self.esd_arr,
                    x_s_arr=[val.num_self for val in self.stations.values()],
                    x_c_arr=[val.num_opponent for val in self.stations.values()],
                    alpha=ALPHA,
                    plot=False,
                    mode='multi' if self.single is False else 'single',
                    time_limit=MINLP_TIME_LIMIT
                )

                # REA test
                st = time.time()
                ___ = get_REA_routes_test(
                    num_of_van=1,
                    van_location=[0],
                    van_dis_left=[0],
                    van_load=[0],
                    c_s=CAP_S,
                    c_v=VEH_CAP,
                    cur_t=round(self.t / MIN_RUN_STEP),
                    t_p=round(T_PLAN / MIN_RUN_STEP),
                    t_f=round(T_FORE / MIN_RUN_STEP),
                    t_roll=round(T_ROLL / MIN_RUN_STEP),
                    c_mat=self.get_MINLP_dist_mat(),
                    ei_s_arr=self.ei_s_arr,
                    ei_c_arr=self.ei_c_arr,
                    esd_arr=self.esd_arr,
                    x_s_arr=[val.num_self for val in self.stations.values()],
                    x_c_arr=[val.num_opponent for val in self.stations.values()],
                    alpha=ALPHA,
                    est_ins=0,
                    branch=2,
                )
                ed = time.time()
                print(f'REA time cost: {ed - st}')

                assert self.future_dec_dict['n_r'][0][0] == 0 and self.future_dec_dict['routes'][0][0] == 0
                inv_dec = -1
                route_dec = self.future_dec_dict['routes'][0][1]
                if self.print_action:
                    print(
                        f'expected inventory: {self.future_dec_dict["exp_inv"][0][0]}, expected target inventory: {self.future_dec_dict["exp_target_inv"][0][0]}')

            else:  # at stations
                cur_station, cur_load = self.veh_info[0], self.veh_info[2]
                if self.t == RE_END_T:
                    realized_ins = min(cur_load, self.stations[cur_station].cap - self.stations[cur_station].num_self)
                    inv_dec = self.stations[cur_station].num_self + realized_ins
                    route_dec = cur_station
                else:
                    if self.t - self.last_dec_t < T_ROLL:
                        cur_ind = round(self.t / MIN_RUN_STEP - self.future_dec_dict['start_time'])
                        if self.future_dec_dict['loc'][0][cur_ind] == cur_station:
                            planned_ins = self.future_dec_dict['n_r'][0][cur_ind]
                            if planned_ins > 0:
                                realized_ins = min(
                                    planned_ins, cur_load,
                                    self.stations[cur_station].cap - self.stations[cur_station].num_self)
                                inv_dec = self.stations[cur_station].num_self + realized_ins
                            elif planned_ins < 0:
                                realized_ins = min(
                                    -planned_ins, self.stations[cur_station].num_self, VEH_CAP - cur_load)
                                inv_dec = self.stations[cur_station].num_self - realized_ins
                            else:
                                inv_dec = self.stations[cur_station].num_self
                            cur_route_ind = self.future_dec_dict['routes'][0].index(cur_station)
                            if cur_route_ind == len(self.future_dec_dict['routes'][0]) - 1:
                                print('cannot cover t_rolling')
                                print(
                                    f'time: {self.t}, cur_station: {cur_station}, start_time: {self.future_dec_dict["start_time"]}, routes: {self.future_dec_dict["loc"][0]}')
                                route_dec = cur_station
                            else:
                                route_dec = self.future_dec_dict['routes'][0][cur_route_ind + 1]

                            if self.print_action:
                                print(
                                    f'expected inventory: {self.future_dec_dict["exp_inv"][0][cur_ind]}, expected target inventory: {self.future_dec_dict["exp_target_inv"][0][cur_ind]}')

                        else:  # remain at the last station, ins sequence cannot cover t_rolling
                            assert cur_station == self.future_dec_dict['routes'][0][
                                -1], f'{cur_station}, {self.future_dec_dict["routes"]}'
                            inv_dec, route_dec = self.stations[cur_station].num_self, cur_station

                    else:  # update dict

                        rea_test_ins = self.future_dec_dict['n_r'][0][self.future_dec_dict['loc'][0].index(cur_station)]
                        self.last_dec_t = self.t
                        self.future_dec_dict, _, __ = get_relocation_routes(
                            num_of_van=1,
                            van_location=[cur_station],
                            van_dis_left=[0],
                            van_load=[cur_load],
                            c_s=CAP_S,
                            c_v=VEH_CAP,
                            cur_t=round(self.t / MIN_RUN_STEP),
                            t_p=round(T_PLAN / MIN_RUN_STEP),
                            t_f=round(T_FORE / MIN_RUN_STEP),
                            t_roll=round(T_ROLL / MIN_RUN_STEP),
                            c_mat=self.get_MINLP_dist_mat(),
                            ei_s_arr=self.ei_s_arr,
                            ei_c_arr=self.ei_c_arr,
                            esd_arr=self.esd_arr,
                            x_s_arr=[val.num_self for val in self.stations.values()],
                            x_c_arr=[val.num_opponent for val in self.stations.values()],
                            alpha=ALPHA,
                            plot=False,
                            mode='multi' if self.single is False else 'single',
                            time_limit=MINLP_TIME_LIMIT
                        )

                        # # REA test
                        # st = time.time()
                        # ___ = get_REA_routes_test(
                        #     num_of_van=1,
                        #     van_location=[cur_station],
                        #     van_dis_left=[0],
                        #     van_load=[cur_load],
                        #     c_s=CAP_S,
                        #     c_v=VEH_CAP,
                        #     cur_t=round(self.t / MIN_RUN_STEP),
                        #     t_p=round(T_PLAN / MIN_RUN_STEP),
                        #     t_f=round(T_FORE / MIN_RUN_STEP),
                        #     t_roll=round(T_ROLL / MIN_RUN_STEP),
                        #     c_mat=self.get_MINLP_dist_mat(),
                        #     ei_s_arr=self.ei_s_arr,
                        #     ei_c_arr=self.ei_c_arr,
                        #     esd_arr=self.esd_arr,
                        #     x_s_arr=[val.num_self for val in self.stations.values()],
                        #     x_c_arr=[val.num_opponent for val in self.stations.values()],
                        #     alpha=ALPHA,
                        #     est_ins=rea_test_ins,
                        #     branch=2,
                        # )
                        # ed = time.time()
                        # print(f'REA time cost: {ed - st}')

                        assert self.future_dec_dict['loc'][0][0] == cur_station
                        planned_ins = self.future_dec_dict['n_r'][0][0]
                        if planned_ins > 0:
                            realized_ins = min(
                                planned_ins, cur_load,
                                self.stations[cur_station].cap - self.stations[cur_station].num_self)
                            inv_dec = self.stations[cur_station].num_self + realized_ins
                        elif planned_ins < 0:
                            realized_ins = min(
                                -planned_ins, self.stations[cur_station].num_self, VEH_CAP - cur_load)
                            inv_dec = self.stations[cur_station].num_self - realized_ins
                        else:
                            inv_dec = self.stations[cur_station].num_self
                        route_dec = self.future_dec_dict['routes'][0][1]
                        if self.print_action:
                            print(
                                f'expected inventory: {self.future_dec_dict["exp_inv"][0][0]}, expected target inventory: {self.future_dec_dict["exp_target_inv"][0][0]}')

        elif self.policy == 'REA_test':
            if self.last_dec_t is None:  # at depot
                assert self.t == RE_START_T
                self.last_dec_t = self.t  # first decision
                # closest to the planned amount of loading/unloading
                self.future_dec_dict = get_CG_REA_routes(
                    num_of_van=1,
                    van_location=[0],
                    van_dis_left=[0],
                    van_load=[0],
                    c_s=CAP_S,
                    c_v=VEH_CAP,
                    cur_t=round(self.t / MIN_RUN_STEP),
                    t_p=round(T_PLAN / MIN_RUN_STEP),
                    t_f=round(T_FORE / MIN_RUN_STEP),
                    t_roll=round(T_ROLL / MIN_RUN_STEP),
                    c_mat=self.get_MINLP_dist_mat(),
                    ei_s_arr=self.ei_s_arr,
                    ei_c_arr=self.ei_c_arr,
                    esd_arr=self.esd_arr,
                    x_s_arr=[val.num_self for val in self.stations.values()],
                    x_c_arr=[val.num_opponent for val in self.stations.values()],
                    alpha=ALPHA,
                    est_ins=0,
                    branch=2
                )
                print(self.future_dec_dict)
                inv_dec = -1
                route_dec = self.future_dec_dict['routes'][0][1]
                if self.print_action:
                    print(f'expected inventory: at depot, expected target inventory: at depot')
            else:  # at station
                cur_station, cur_load = self.veh_info[0], self.veh_info[2]
                if self.t == RE_END_T:
                    realized_ins = min(cur_load, self.stations[cur_station].cap - self.stations[cur_station].num_self)
                    inv_dec = self.stations[cur_station].num_self + realized_ins  # drop all the bikes here
                    route_dec = cur_station
                else:
                    if self.t - self.last_dec_t < T_ROLL:  # before rolling
                        cur_ind = round(self.t / MIN_RUN_STEP - self.future_dec_dict['start_time'])
                        if self.future_dec_dict['loc'][0][cur_ind] == cur_station:
                            planned_ins = self.future_dec_dict['n_r'][0][cur_ind]
                            if planned_ins > 0:
                                realized_ins = min(planned_ins, cur_load,
                                                   self.stations[cur_station].cap - self.stations[cur_station].num_self)
                                inv_dec = self.stations[cur_station].num_self + realized_ins
                            elif planned_ins < 0:
                                realized_ins = min(-planned_ins, self.stations[cur_station].num_self,
                                                   VEH_CAP - cur_load)
                                inv_dec = self.stations[cur_station].num_self - realized_ins
                            else:
                                inv_dec = self.stations[cur_station].num_self
                            cur_route_ind = self.future_dec_dict['routes'][0].index(cur_station)
                            if cur_route_ind == len(self.future_dec_dict['routes'][0]) - 1:
                                print('cannot cover t_rolling')
                                print(
                                    f'time: {self.t}, cur_station: {cur_station}, start_time: {self.future_dec_dict["start_time"]}, routes: {self.future_dec_dict["loc"][0]}')
                                route_dec = cur_station
                            else:
                                route_dec = self.future_dec_dict['routes'][0][cur_route_ind + 1]

                            if self.print_action:
                                print(
                                    f'expected inventory: {self.future_dec_dict["exp_inv"][0][cur_ind]}, '
                                    f'expected target inventory: {self.future_dec_dict["exp_target_inv"][0][cur_ind]}')
                        else:  # remain at the last station, ins sequence cannot cover rolling time
                            assert cur_station == self.future_dec_dict['routes'][0][-1], \
                                f'{cur_station}, {self.future_dec_dict["routes"]}'
                            inv_dec, route_dec = self.stations[cur_station].num_self, cur_station

                    else:  # update dict
                        scheduled_ins = self.future_dec_dict['n_r'][0][
                            self.future_dec_dict['loc'][0].index(cur_station)]
                        self.last_dec_t = self.t
                        self.future_dec_dict = get_CG_REA_routes(
                            num_of_van=1,
                            van_location=[cur_station],
                            van_dis_left=[0],
                            van_load=[cur_load],
                            c_s=CAP_S,
                            c_v=VEH_CAP,
                            cur_t=round(self.t / MIN_RUN_STEP),
                            t_p=round(T_PLAN / MIN_RUN_STEP),
                            t_f=round(T_FORE / MIN_RUN_STEP),
                            t_roll=round(T_ROLL / MIN_RUN_STEP),
                            c_mat=self.get_MINLP_dist_mat(),
                            ei_s_arr=self.ei_s_arr,
                            ei_c_arr=self.ei_c_arr,
                            esd_arr=self.esd_arr,
                            x_s_arr=[val.num_self for val in self.stations.values()],
                            x_c_arr=[val.num_opponent for val in self.stations.values()],
                            alpha=ALPHA,
                            est_ins=scheduled_ins,
                            branch=2
                        )
                        print(self.future_dec_dict)
                        assert self.future_dec_dict['loc'][0][0] == cur_station
                        planned_ins = self.future_dec_dict['n_r'][0][0]
                        if planned_ins > 0:
                            realized_ins = min(
                                planned_ins, cur_load,
                                self.stations[cur_station].cap - self.stations[cur_station].num_self)
                            inv_dec = self.stations[cur_station].num_self + realized_ins
                        elif planned_ins < 0:
                            realized_ins = min(
                                -planned_ins, self.stations[cur_station].num_self, VEH_CAP - cur_load)
                            inv_dec = self.stations[cur_station].num_self - realized_ins
                        else:
                            inv_dec = self.stations[cur_station].num_self
                        if len(self.future_dec_dict['routes'][0]) == 1:
                            route_dec = cur_station
                        else:
                            route_dec = self.future_dec_dict['routes'][0][1]
                        if self.print_action:
                            print(
                                f'expected inventory: {self.future_dec_dict["exp_inv"][0][0]}, '
                                f'expected target inventory: {self.future_dec_dict["exp_target_inv"][0][0]}')

        elif self.policy == 'DP_test' or self.policy == 'exact_test':
            dec_list = []
            if self.last_dec_t is None:  # all at depot
                assert self.t == RE_START_T
                self.last_dec_t = self.t  # first decision
                # closest to the planned amount of loading/unloading
                if self.policy == 'DP_test':  # todo: test branch and price using the 'DP_test' structure
                    # self.future_dec_dict = get_DP_routes_greedy(
                    #     num_of_van=self.num_of_veh,
                    #     van_location=[0 for _ in range(self.num_of_veh)],
                    #     van_dis_left=[0 for _ in range(self.num_of_veh)],
                    #     van_load=[0 for _ in range(self.num_of_veh)],
                    #     c_s=CAP_S,
                    #     c_v=VEH_CAP,
                    #     cur_t=round(self.t / MIN_RUN_STEP),
                    #     t_p=round(T_PLAN / MIN_RUN_STEP),
                    #     t_f=round(T_FORE / MIN_RUN_STEP),
                    #     t_roll=round(T_ROLL / MIN_RUN_STEP),
                    #     c_mat=self.get_MINLP_dist_mat(),
                    #     ei_s_arr=self.ei_s_arr,
                    #     ei_c_arr=self.ei_c_arr,
                    #     esd_arr=self.esd_arr,
                    #     x_s_arr=[val.num_self for val in self.stations.values()],
                    #     x_c_arr=[val.num_opponent for val in self.stations.values()],
                    #     alpha=ALPHA,
                    # )
                    self.future_dec_dict = get_routes_branch_and_price(
                        num_of_van=self.num_of_veh,
                        van_location=[0 for _ in range(self.num_of_veh)],
                        van_dis_left=[0 for _ in range(self.num_of_veh)],
                        van_load=[0 for _ in range(self.num_of_veh)],
                        c_s=CAP_S,
                        c_v=VEH_CAP,
                        cur_t=round(self.t / MIN_RUN_STEP),
                        t_p=round(T_PLAN / MIN_RUN_STEP),
                        t_f=round(T_FORE / MIN_RUN_STEP),
                        t_roll=round(T_ROLL / MIN_RUN_STEP),
                        c_mat=self.get_MINLP_dist_mat(),
                        ei_s_arr=self.ei_s_arr,
                        ei_c_arr=self.ei_c_arr,
                        esd_arr=self.esd_arr,
                        x_s_arr=[val.num_self for val in self.stations.values()],
                        x_c_arr=[val.num_opponent for val in self.stations.values()],
                        alpha=ALPHA,
                        est_ins=[0 for _ in range(self.num_of_veh)]
                    )
                    # import pickle
                    # with open('test_van_location.pkl', 'rb') as f:
                    #     test_van_location = pickle.load(f)
                    # with open('test_van_dis_left.pkl', 'rb') as f:
                    #     test_dis_left = pickle.load(f)
                    # with open('test_van_load.pkl', 'rb') as f:
                    #     test_van_load = pickle.load(f)
                    # with open('test_cur_t.pkl', 'rb') as f:
                    #     test_cur_t = pickle.load(f)
                    # with open('test_x_s_arr.pkl', 'rb') as f:
                    #     test_x_s_arr = pickle.load(f)
                    # with open('test_x_c_arr.pkl', 'rb') as f:
                    #     test_x_c_arr = pickle.load(f)
                    # with open('test_scheduled_ins.pkl', 'rb') as f:
                    #     test_est_ins = pickle.load(f)
                    # self.future_dec_dict = get_routes_branch_and_price(
                    #     num_of_van=self.num_of_veh,
                    #     van_location=test_van_location,
                    #     van_dis_left=test_dis_left,
                    #     van_load=test_van_load,
                    #     c_s=CAP_S,
                    #     c_v=VEH_CAP,
                    #     cur_t=test_cur_t,
                    #     t_p=round(T_PLAN / MIN_RUN_STEP),
                    #     t_f=round(T_FORE / MIN_RUN_STEP),
                    #     t_roll=round(T_ROLL / MIN_RUN_STEP),
                    #     c_mat=self.get_MINLP_dist_mat(),
                    #     ei_s_arr=self.ei_s_arr,
                    #     ei_c_arr=self.ei_c_arr,
                    #     esd_arr=self.esd_arr,
                    #     x_s_arr=test_x_s_arr,
                    #     x_c_arr=test_x_c_arr,
                    #     alpha=ALPHA,
                    #     est_ins=test_est_ins
                    # )

                else:
                    assert False, f'to fill mode: exact_test'
                # self.future_dec_dict = {'objective': 4190.522352925615, 'start_time': 84, 'routes': [[0, 5, 4, 6, 19, 12, 16], [0, 15, 7, 10, 8, 18]], 'exp_inv': [[0, 34.76888403354017, None, 1.3884099068276705, None, None, 36.34512042804721, None, 11.54645116635093, None, 38.70774301315821, None, 0.9211436567541507], [0, 0, 26.04215368940173, None, 1.0744787598225505, None, None, None, 30.01515586957035, None, 38.375582624506166, None, 4.419167753156481]], 'exp_target_inv': [[0, 10, None, 26, None, None, 11, None, 37, None, 14, None, 26], [0, 0, 1, None, 26, None, None, None, 19, None, 24, None, 29]], 'loc': [[0, 5, None, 4, None, None, 6, None, 19, None, 12, None, 16], [0, 0, 15, None, 7, None, None, None, 10, None, 8, None, 18]], 'n_r': [[0, -25, None, 25, None, None, -25, None, 25, None, -25, None, 25], [0, -100, -25, None, 25, None, None, None, -11, None, -14, None, 25]]}
                print(self.future_dec_dict)
                for veh in range(self.num_of_veh):
                    inv_dec = -1
                    if self.future_dec_dict['loc'][veh][1] != 0:
                        route_dec = self.future_dec_dict['routes'][veh][1]
                    else:  # initial stay at depot
                        route_dec = 0
                    dec_list.append({'inv': inv_dec, 'route': route_dec})
                    if self.print_action:
                        print(f'Vehicle {veh}: at depot')
            else:  # at station
                if self.t == RE_END_T:
                    for veh in range(self.num_of_veh):
                        if self.veh_info[veh][2] == 0:  # time to decide
                            cur_station, cur_load = self.veh_info[veh][0], self.veh_info[veh][3]
                            realized_ins = min(cur_load, self.stations[cur_station].cap - self.stations[cur_station].num_self)
                            inv_dec = self.stations[cur_station].num_self + realized_ins
                            route_dec = cur_station
                            dec_list.append({'inv': inv_dec, 'route': route_dec})
                        else:
                            dec_list.append({'inv': None, 'route': None})
                elif self.t - self.last_dec_t < T_ROLL:
                    for veh in range(self.num_of_veh):
                        assert self.veh_info[veh][2] is not None
                        if self.veh_info[veh][2] == 0:  # time to decide
                            cur_station, cur_load = self.veh_info[veh][0], self.veh_info[veh][3]
                            cur_ind = round(self.t / MIN_RUN_STEP - self.future_dec_dict['start_time'])
                            if self.future_dec_dict['loc'][veh][cur_ind] == cur_station:
                                planned_ins = self.future_dec_dict['n_r'][veh][cur_ind]
                                if planned_ins < -99:  # stay
                                    inv_dec = -1
                                else:
                                    if planned_ins > 0:
                                        realized_ins = min(planned_ins, cur_load,
                                                           self.stations[cur_station].cap - self.stations[cur_station].num_self)
                                        inv_dec = self.stations[cur_station].num_self + realized_ins
                                    elif planned_ins < 0:
                                        realized_ins = min(-planned_ins, self.stations[cur_station].num_self,
                                                           VEH_CAP - cur_load)
                                        inv_dec = self.stations[cur_station].num_self - realized_ins
                                    else:
                                        inv_dec = self.stations[cur_station].num_self  # do no repositioning
                                if cur_ind < len(self.future_dec_dict['loc'][veh]) - 1:
                                    if self.future_dec_dict['loc'][veh][cur_ind + 1] == cur_station:
                                        route_dec = cur_station  # stay at current station
                                    else:
                                        # assert self.future_dec_dict['loc'][veh][cur_ind + 1] is None, f"{self.future_dec_dict['loc'][veh][cur_ind + 1]}"
                                        cur_route_ind = self.future_dec_dict['routes'][veh].index(cur_station)
                                        route_dec = self.future_dec_dict['routes'][veh][cur_route_ind + 1]
                                        # fix inv decision
                                        if cur_station > 0:
                                            inv_dec = inv_dec if inv_dec > -0.5 else self.stations[cur_station].num_self
                                        else:
                                            inv_dec = -1
                                else:
                                    print('cannot cover t_rolling')
                                    route_dec = cur_station
                                if self.print_action:
                                    print(
                                        f'Vehicle {veh}: expected inventory: {self.future_dec_dict["exp_inv"][veh][cur_ind]}, '
                                        f'expected target inventory: {self.future_dec_dict["exp_target_inv"][veh][cur_ind]}')
                                dec_list.append({'inv': inv_dec, 'route': route_dec})
                            else:  # remain at the last station, ins sequence cannot cover rolling time
                                assert False
                        else:
                            dec_list.append({'inv': None, 'route': None})
                else:  # update dict
                    self.last_dec_t = self.t
                    if self.policy == 'DP_test':
                        # self.future_dec_dict = get_DP_routes_greedy(
                        #     num_of_van=self.num_of_veh,
                        #     van_location=[self.veh_info[veh][1] for veh in range(self.num_of_veh)],
                        #     van_dis_left=[round(self.veh_info[veh][2]/MIN_RUN_STEP) for veh in range(self.num_of_veh)],
                        #     van_load=[self.veh_info[veh][3] for veh in range(self.num_of_veh)],
                        #     c_s=CAP_S,
                        #     c_v=VEH_CAP,
                        #     cur_t=round(self.t / MIN_RUN_STEP),
                        #     t_p=round(T_PLAN / MIN_RUN_STEP),
                        #     t_f=round(T_FORE / MIN_RUN_STEP),
                        #     t_roll=round(T_ROLL / MIN_RUN_STEP),
                        #     c_mat=self.get_MINLP_dist_mat(),
                        #     ei_s_arr=self.ei_s_arr,
                        #     ei_c_arr=self.ei_c_arr,
                        #     esd_arr=self.esd_arr,
                        #     x_s_arr=[val.num_self for val in self.stations.values()],
                        #     x_c_arr=[val.num_opponent for val in self.stations.values()],
                        #     alpha=ALPHA,
                        # )
                        scheduled_ins = [self.future_dec_dict['n_r'][veh][
                                             self.future_dec_dict['loc'][veh].index(
                                                 self.veh_info[veh][1])] for veh in range(self.num_of_veh)]
                        print(f'scheduled_ins: {scheduled_ins}')
                        self.future_dec_dict = get_routes_branch_and_price(
                            num_of_van=self.num_of_veh,
                            van_location=[self.veh_info[veh][1] for veh in range(self.num_of_veh)],
                            van_dis_left=[round(self.veh_info[veh][2]/MIN_RUN_STEP) for veh in range(self.num_of_veh)],
                            van_load=[self.veh_info[veh][3] for veh in range(self.num_of_veh)],
                            c_s=CAP_S,
                            c_v=VEH_CAP,
                            cur_t=round(self.t / MIN_RUN_STEP),
                            t_p=round(T_PLAN / MIN_RUN_STEP),
                            t_f=round(T_FORE / MIN_RUN_STEP),
                            t_roll=round(T_ROLL / MIN_RUN_STEP),
                            c_mat=self.get_MINLP_dist_mat(),
                            ei_s_arr=self.ei_s_arr,
                            ei_c_arr=self.ei_c_arr,
                            esd_arr=self.esd_arr,
                            x_s_arr=[val.num_self for val in self.stations.values()],
                            x_c_arr=[val.num_opponent for val in self.stations.values()],
                            alpha=ALPHA,
                            est_ins=scheduled_ins
                        )
                    else:
                        assert False, f'to fill mode: exact_test'
                    print(self.future_dec_dict)
                    for veh in range(self.num_of_veh):
                        assert self.veh_info[veh][2] is not None
                        if self.veh_info[veh][2] == 0:  # time to decide
                            cur_station, cur_load = self.veh_info[veh][0], self.veh_info[veh][3]
                            planned_ins = self.future_dec_dict['n_r'][veh][0]
                            cur_ind = round(self.t / MIN_RUN_STEP - self.future_dec_dict['start_time'])
                            if planned_ins < -99:  # stay
                                inv_dec = -1
                            else:
                                if planned_ins > 0:
                                    realized_ins = min(planned_ins, cur_load,
                                                       self.stations[cur_station].cap - self.stations[cur_station].num_self)
                                    inv_dec = self.stations[cur_station].num_self + realized_ins
                                elif planned_ins < 0:
                                    realized_ins = min(-planned_ins, self.stations[cur_station].num_self,
                                                       VEH_CAP - cur_load)
                                    inv_dec = self.stations[cur_station].num_self - realized_ins
                                else:
                                    inv_dec = self.stations[cur_station].num_self
                            if len(self.future_dec_dict['routes'][veh]) == 1:
                                route_dec = cur_station
                            else:
                                if cur_ind < len(self.future_dec_dict['loc'][veh]) - 1:
                                    if self.future_dec_dict['loc'][veh][cur_ind + 1] == cur_station:
                                        route_dec = cur_station
                                    else:
                                        assert self.future_dec_dict['loc'][veh][cur_ind + 1] is None, f'{veh}, {cur_ind}'
                                        route_dec = self.future_dec_dict['routes'][veh][1]
                                        # fix inv decision
                                        inv_dec = inv_dec if inv_dec > -0.5 else self.stations[cur_station].num_self
                                else:
                                    print('cannot cover t_rolling')
                                    route_dec = cur_station
                            if self.print_action:
                                print(
                                    f'Vehicle {veh}: expected inventory: {self.future_dec_dict["exp_inv"][veh][0]}, '
                                    f'expected target inventory: {self.future_dec_dict["exp_target_inv"][veh][0]}')
                            dec_list.append({'inv': inv_dec, 'route': route_dec})
                        else:
                            dec_list.append({'inv': None, 'route': None})

        else:
            print('policy type error.')
            assert False

        # assert route_dec is not None
        # return {'inv': inv_dec, 'route': route_dec}
        return dec_list

    def decide_action_single_info(self):
        """
        决策当前站点目标库存水平和下一站点决策（单平台信息）

        :return: 决策字典, {'inv': inv_dec, 'route': route_dec}
        """
        # todo: 重构
        # do nothing
        if self.policy is None:
            dec_list = []
            for veh in range(self.num_of_veh):
                veh_dec_list = {'inv': -1, 'route': self.veh_info[veh][0]}
                dec_list.append(veh_dec_list)

        elif self.policy == 'STR':
            cur_station, cur_load = self.veh_info[0], self.veh_info[2]
            if cur_station:
                cur_inv = self.stations[cur_station].num_self
                # shortage
                if cur_inv < round(GAMMA * self.stations[cur_station].cap):
                    inv_dec = min(round(GAMMA * self.stations[cur_station].cap), cur_inv + cur_load)
                    load_after_ins = cur_load - (inv_dec - cur_inv)
                # surplus
                elif cur_inv > round((1 - GAMMA) * self.stations[cur_station].cap):
                    inv_dec = max(round((1 - GAMMA) * self.stations[cur_station].cap), cur_inv - (VEH_CAP - cur_load))
                    load_after_ins = cur_load + cur_inv - inv_dec
                # balanced
                else:
                    inv_dec = -1
                    load_after_ins = cur_load
                pot_stations = [i for i in self.stations.keys() if i != cur_station]
                if 0 < load_after_ins < VEH_CAP:
                    imb_stations = [i for i in pot_stations
                                    if self.stations[i].num_self > (1 - GAMMA) * self.stations[i].cap or self.stations[
                                        i].num_self < GAMMA * self.stations[i].cap]
                elif load_after_ins == 0:
                    imb_stations = [i for i in pot_stations if
                                    self.stations[i].num_self > (1 - GAMMA) * self.stations[i].cap]
                elif load_after_ins == VEH_CAP:
                    imb_stations = [i for i in pot_stations if
                                    self.stations[i].num_self < GAMMA * self.stations[i].cap]
                else:
                    imb_stations = []
                # 有可以前往的站点
                if imb_stations:
                    dis_list = [self.dist[cur_station, i] for i in imb_stations]
                    route_dec_idx = \
                        random.sample([i for i in range(len(imb_stations)) if dis_list[i] == min(dis_list)], 1)[0]
                    route_dec = imb_stations[route_dec_idx]
                # 没有可以前往的站点
                else:
                    route_dec = cur_station
            else:  # at depot
                inv_dec = -1
                pot_stations = [i for i in self.stations.keys()]
                surplus_stations = [i for i in pot_stations if
                                    self.stations[i].num_self > (1 - GAMMA) * self.stations[i].cap]
                # 有可以前往的站点
                if surplus_stations:
                    dis_list = [self.dist[0, i] for i in surplus_stations]
                    route_dec_idx = \
                        random.sample([i for i in range(len(surplus_stations)) if dis_list[i] == min(dis_list)], 1)[0]
                    route_dec = surplus_stations[route_dec_idx]
                # 没有可以前往的站点
                else:
                    route_dec = 0

        elif self.policy == 'GLA':  # 2023 TS baseline
            dec_list = []
            former_route_dec = []
            next_visit_stations = [veh_info[1] for veh_info in self.veh_info if veh_info[2] is not None and veh_info[2] > 0]
            for veh in range(self.num_of_veh):
                if self.veh_info[veh][2] == 0 or self.veh_info[veh][2] is None:
                    cur_station, cur_load = self.veh_info[veh][0], self.veh_info[veh][3]
                    if cur_station:
                        cur_inv = self.stations[cur_station].num_self
                        # inv decision
                        dep = sum(
                            self.mu_s_array[int(self.t / MIN_STEP):int(self.t / MIN_STEP + GLA_HORIZON / MIN_STEP),
                            cur_station - 1])
                        arr = sum(
                            self.lambda_s_array[int(self.t / MIN_STEP):int(self.t / MIN_STEP + GLA_HORIZON / MIN_STEP),
                            cur_station - 1])
                        net_demand = int(dep - arr) + 1 if dep > arr else int(dep - arr)
                        if net_demand >= cur_inv:
                            inv_dec = min(net_demand, self.stations[cur_station].cap)
                            load_after_ins = cur_load - min(inv_dec - cur_inv, cur_load)
                        else:
                            inv_dec = max(net_demand, 0)
                            load_after_ins = cur_load + min(cur_inv - inv_dec, VEH_CAP - cur_load)
                        # route decision
                        rate = load_after_ins / VEH_CAP
                        stations = [i for i in self.stations.keys() if i != cur_station]
                        # cannot go to same stations
                        stations = [val for val in stations if
                                    (val not in former_route_dec) and (val not in next_visit_stations)]
                        random.shuffle(stations)
                        if rate <= GLA_delta:  # load
                            num_self_list = [self.stations[station].num_self for station in stations]
                            route_dec = stations[num_self_list.index(max(num_self_list))]
                            former_route_dec.append(route_dec)
                        else:  # unload
                            net_demand_list = [
                                round(
                                    sum(
                                        self.mu_s_array[
                                        int(self.t / MIN_STEP):int(self.t / MIN_STEP + GLA_HORIZON / MIN_STEP),
                                        station - 1]
                                    ) -
                                    sum(
                                        self.lambda_s_array[
                                        int(self.t / MIN_STEP):int(self.t / MIN_STEP + GLA_HORIZON / MIN_STEP),
                                        station - 1]
                                    )
                                )
                                for station in stations
                            ]
                            route_dec = stations[net_demand_list.index(max(net_demand_list))]
                            former_route_dec.append(route_dec)

                    else:  # at depot
                        inv_dec = -1
                        stations = [i for i in self.stations.keys() if i != cur_station]
                        stations = [val for val in stations if val not in former_route_dec]
                        random.shuffle(stations)
                        num_self_list = [self.stations[station].num_self for station in stations]
                        route_dec = stations[num_self_list.index(max(num_self_list))]
                        former_route_dec.append(route_dec)
                    dec_list.append({'inv': inv_dec, 'route': route_dec})
                else:
                    dec_list.append({'inv': None, 'route': None})

        elif self.policy == 'MINLP':

            if self.last_dec_t is None:  # at depot
                assert self.t == RE_START_T
                self.last_dec_t = self.t  # 第一次决策
                # closest to the planned amount of loading/unloading
                self.future_dec_dict, _, __ = get_relocation_routes(
                    num_of_van=1,
                    van_location=[0],
                    van_dis_left=[0],
                    van_load=[0],
                    c_s=CAP_S,
                    c_v=VEH_CAP,
                    cur_t=round(self.t / MIN_RUN_STEP),
                    t_p=round(T_PLAN / MIN_RUN_STEP),
                    t_f=round(T_FORE / MIN_RUN_STEP),
                    t_roll=round(T_ROLL / MIN_RUN_STEP),
                    c_mat=self.get_MINLP_dist_mat(),
                    ei_s_arr=self.ei_s_arr,
                    ei_c_arr=self.ei_c_arr,
                    esd_arr=self.esd_arr,
                    x_s_arr=[val.num_self for val in self.stations.values()],
                    x_c_arr=[val.num_opponent for val in self.stations.values()],
                    alpha=ALPHA,
                    plot=False,
                    mode='multi' if self.single is False else 'single',
                    time_limit=MINLP_TIME_LIMIT
                )

                # REA test
                st = time.time()
                ___ = get_REA_routes_test(
                    num_of_van=1,
                    van_location=[0],
                    van_dis_left=[0],
                    van_load=[0],
                    c_s=CAP_S,
                    c_v=VEH_CAP,
                    cur_t=round(self.t / MIN_RUN_STEP),
                    t_p=round(T_PLAN / MIN_RUN_STEP),
                    t_f=round(T_FORE / MIN_RUN_STEP),
                    t_roll=round(T_ROLL / MIN_RUN_STEP),
                    c_mat=self.get_MINLP_dist_mat(),
                    ei_s_arr=self.ei_s_arr,
                    ei_c_arr=self.ei_c_arr,
                    esd_arr=self.esd_arr,
                    x_s_arr=[val.num_self for val in self.stations.values()],
                    x_c_arr=[val.num_opponent for val in self.stations.values()],
                    alpha=ALPHA,
                    est_ins=0,
                    branch=2,
                )
                ed = time.time()
                print(f'REA time cost: {ed - st}')

                assert self.future_dec_dict['n_r'][0][0] == 0 and self.future_dec_dict['routes'][0][0] == 0
                inv_dec = -1
                route_dec = self.future_dec_dict['routes'][0][1]
                if self.print_action:
                    print(
                        f'expected inventory: {self.future_dec_dict["exp_inv"][0][0]}, expected target inventory: {self.future_dec_dict["exp_target_inv"][0][0]}')

            else:  # at stations
                cur_station, cur_load = self.veh_info[0], self.veh_info[2]
                if self.t == RE_END_T:
                    realized_ins = min(cur_load, self.stations[cur_station].cap - self.stations[cur_station].num_self)
                    inv_dec = self.stations[cur_station].num_self + realized_ins
                    route_dec = cur_station
                else:
                    if self.t - self.last_dec_t < T_ROLL:
                        cur_ind = round(self.t / MIN_RUN_STEP - self.future_dec_dict['start_time'])
                        if self.future_dec_dict['loc'][0][cur_ind] == cur_station:
                            planned_ins = self.future_dec_dict['n_r'][0][cur_ind]
                            if planned_ins > 0:
                                realized_ins = min(
                                    planned_ins, cur_load,
                                    self.stations[cur_station].cap - self.stations[cur_station].num_self)
                                inv_dec = self.stations[cur_station].num_self + realized_ins
                            elif planned_ins < 0:
                                realized_ins = min(
                                    -planned_ins, self.stations[cur_station].num_self, VEH_CAP - cur_load)
                                inv_dec = self.stations[cur_station].num_self - realized_ins
                            else:
                                inv_dec = self.stations[cur_station].num_self
                            cur_route_ind = self.future_dec_dict['routes'][0].index(cur_station)
                            if cur_route_ind == len(self.future_dec_dict['routes'][0]) - 1:
                                print('cannot cover t_rolling')
                                print(
                                    f'time: {self.t}, cur_station: {cur_station}, start_time: {self.future_dec_dict["start_time"]}, routes: {self.future_dec_dict["loc"][0]}')
                                route_dec = cur_station
                            else:
                                route_dec = self.future_dec_dict['routes'][0][cur_route_ind + 1]

                            if self.print_action:
                                print(
                                    f'expected inventory: {self.future_dec_dict["exp_inv"][0][cur_ind]}, expected target inventory: {self.future_dec_dict["exp_target_inv"][0][cur_ind]}')

                        else:  # remain at the last station, ins sequence cannot cover t_rolling
                            assert cur_station == self.future_dec_dict['routes'][0][
                                -1], f'{cur_station}, {self.future_dec_dict["routes"]}'
                            inv_dec, route_dec = self.stations[cur_station].num_self, cur_station

                    else:  # update dict
                        self.last_dec_t = self.t
                        self.future_dec_dict, _, __ = get_relocation_routes(
                            num_of_van=1,
                            van_location=[cur_station],
                            van_dis_left=[0],
                            van_load=[cur_load],
                            c_s=CAP_S,
                            c_v=VEH_CAP,
                            cur_t=round(self.t / MIN_RUN_STEP),
                            t_p=round(T_PLAN / MIN_RUN_STEP),
                            t_f=round(T_FORE / MIN_RUN_STEP),
                            t_roll=round(T_ROLL / MIN_RUN_STEP),
                            c_mat=self.get_MINLP_dist_mat(),
                            ei_s_arr=self.ei_s_arr,
                            ei_c_arr=self.ei_c_arr,
                            esd_arr=self.esd_arr,
                            x_s_arr=[val.num_self for val in self.stations.values()],
                            x_c_arr=[val.num_opponent for val in self.stations.values()],
                            alpha=ALPHA,
                            plot=False,
                            mode='multi' if self.single is False else 'single',
                            time_limit=MINLP_TIME_LIMIT
                        )

                        # REA test
                        st = time.time()
                        ___ = get_REA_routes_test(
                            num_of_van=1,
                            van_location=[0],
                            van_dis_left=[0],
                            van_load=[0],
                            c_s=CAP_S,
                            c_v=VEH_CAP,
                            cur_t=round(self.t / MIN_RUN_STEP),
                            t_p=round(T_PLAN / MIN_RUN_STEP),
                            t_f=round(T_FORE / MIN_RUN_STEP),
                            t_roll=round(T_ROLL / MIN_RUN_STEP),
                            c_mat=self.get_MINLP_dist_mat(),
                            ei_s_arr=self.ei_s_arr,
                            ei_c_arr=self.ei_c_arr,
                            esd_arr=self.esd_arr,
                            x_s_arr=[val.num_self for val in self.stations.values()],
                            x_c_arr=[val.num_opponent for val in self.stations.values()],
                            alpha=ALPHA,
                            est_ins=self.future_dec_dict['n_r'][0][0],
                            branch=2,
                        )
                        ed = time.time()
                        print(f'REA time cost: {ed - st}')

                        assert self.future_dec_dict['loc'][0][0] == cur_station
                        planned_ins = self.future_dec_dict['n_r'][0][0]
                        if planned_ins > 0:
                            realized_ins = min(
                                planned_ins, cur_load,
                                self.stations[cur_station].cap - self.stations[cur_station].num_self)
                            inv_dec = self.stations[cur_station].num_self + realized_ins
                        elif planned_ins < 0:
                            realized_ins = min(
                                -planned_ins, self.stations[cur_station].num_self, VEH_CAP - cur_load)
                            inv_dec = self.stations[cur_station].num_self - realized_ins
                        else:
                            inv_dec = self.stations[cur_station].num_self
                        route_dec = self.future_dec_dict['routes'][0][1]

                        if self.print_action:
                            print(
                                f'expected inventory: {self.future_dec_dict["exp_inv"][0][0]}, expected target inventory: {self.future_dec_dict["exp_target_inv"][0][0]}')

        else:
            print('policy type error.')
            assert False

        # assert route_dec is not None
        # return {'inv': inv_dec, 'route': route_dec}
        return dec_list

    def update_time_left(self, cur_station: int, next_station: int) -> int:
        """return time left after decision"""
        if cur_station == next_station:
            return STAY_TIME
        elif cur_station == 0:
            assert next_station != 0
            time_step = int((self.dist[0, next_station] - 0.2) / MIN_RUN_STEP) + 1 if self.dist[
                                                                                          0, next_station] > 0 else 0
            return time_step * MIN_RUN_STEP
        else:
            time_step = int((self.dist[cur_station, next_station] - 0.2) / MIN_RUN_STEP) + 1 if self.dist[
                                                                                                    cur_station, next_station] > 0 else 0
            return time_step * MIN_RUN_STEP + MIN_RUN_STEP

    def decide_time(self) -> int:
        """
        根据下一站决策，返回step的时长

        :return: step的时长(int)
        """
        return min([veh_info[2] for veh_info in self.veh_info])

    def apply_decision_multi_info(self, inv_dec_list: list, route_dec_list: list):
        """
        改变当前站点库存和车辆载量，时间转移

        :param inv_dec_list: inventory decision list
        :param route_dec_list: route decision (station id)
        :return:
        """
        cur_station_list, cur_load_list = ([veh_info[0] for veh_info in self.veh_info],
                                           [veh_info[3] for veh_info in self.veh_info])
        # cur_station, cur_load = self.veh_info[0], self.veh_info[2]
        duration_list = [0 for _ in range(self.num_of_veh)]
        for veh in range(len(self.veh_info)):
            cur_station, cur_load = cur_station_list[veh], cur_load_list[veh]
            inv_dec, route_dec = inv_dec_list[veh], route_dec_list[veh]
            if not cur_station:
                if inv_dec is not None:
                    assert inv_dec < 0  # must decide to move or on the way
                    self.veh_info[veh][1] = route_dec
                    self.veh_info[veh][2] = self.update_time_left(cur_station=cur_station, next_station=route_dec)
            else:
                if inv_dec is not None:
                    if inv_dec < 0:  # do nothing
                        self.veh_info[veh][1] = route_dec
                        self.veh_info[veh][2] = self.update_time_left(cur_station=cur_station, next_station=route_dec)
                        duration_list[veh] = STAY_TIME
                    else:
                        if inv_dec > self.stations[cur_station].num_self:
                            ins = min(inv_dec - self.stations[cur_station].num_self, cur_load)
                        elif inv_dec < self.stations[cur_station].num_self:
                            ins = max(inv_dec - self.stations[cur_station].num_self, cur_load - VEH_CAP)
                        else:
                            ins = 0
                        # post decision state
                        # station
                        self.stations[cur_station].num_self += ins
                        # vehicle
                        self.veh_info[veh][1] = route_dec
                        self.veh_info[veh][2] = self.update_time_left(cur_station=cur_station, next_station=route_dec)
                        self.veh_info[veh][3] = self.veh_info[veh][3] - ins
                        # time
                        duration_list[veh] = CONST_OPERATION
        # adjust stay or operation time
        if max(duration_list) > 0:
            assert max(duration_list) == CONST_OPERATION
            count_t = min(self.sim_end_time - self.t, CONST_OPERATION)
            step_end_t = self.t + count_t
            while self.t < step_end_t:
                num_change_list, success_list, success_opponent_list, full_list, empty_list = \
                    self.generate_orders(gene_t=MIN_STEP)
                # num change
                self.apply_num_change(num_change_list)
                # success record
                sum_success = sum(success_list)
                self.success += sum_success
                self.success_list.append(sum_success)
                if self.t >= RECORD_WORK_T:
                    self.success_work += sum_success
                    self.success_work_list.append(sum_success)
                    if self.t < RE_END_T:
                        self.success_work_till_done += sum_success
                        self.success_work_till_done_list.append(sum_success)
                else:
                    self.success_work_list.append(0)
                # success opponent record
                sum_success_oppo = sum(success_opponent_list)
                self.success_opponent += sum_success_oppo
                self.success_opponent_list.append(sum_success_oppo)
                # full loss record
                sum_full_loss = sum(full_list)
                self.full_loss += sum_full_loss
                self.full_loss_list.append(sum_full_loss)
                # empty loss record
                sum_empty_loss = sum(empty_list)
                self.empty_loss += sum_empty_loss
                self.empty_loss_list.append(sum_empty_loss)

                self.t += MIN_STEP

            # update time left
            for veh in range(len(self.veh_info)):
                self.veh_info[veh][2] -= count_t
                assert self.veh_info[veh][2] >= 0

    def apply_decision_single_info(self, inv_dec: int, route_dec: int):
        """
        改变当前站点库存和车辆载量，用于single information时的流转量估计

        :param inv_dec: inventory decision
        :param route_dec: route decision (station id)
        :return:
        """
        assert self.single is True
        cur_station, cur_load = self.veh_info[0], self.veh_info[2]
        if not cur_station:
            assert inv_dec < 0
            self.veh_info[1] = route_dec
        else:
            if inv_dec < 0:  # do nothing
                self.veh_info[1] = route_dec
            else:
                if inv_dec > self.stations[cur_station].num_self:
                    ins = min(inv_dec - self.stations[cur_station].num_self, cur_load)
                elif inv_dec < self.stations[cur_station].num_self:
                    ins = max(inv_dec - self.stations[cur_station].num_self, cur_load - VEH_CAP)
                else:
                    ins = 0
                # post decision state
                # station
                self.stations[cur_station].num_self += ins
                # vehicle
                self.veh_info[1], self.veh_info[2] = route_dec, self.veh_info[2] - ins
                # time
                # operation_duration = OPERATION_TIME * abs(ins)
                operation_duration = CONST_OPERATION
                # 操作时流转数量
                count_t = min(self.sim_end_time - self.t, operation_duration)
                assert count_t == operation_duration
                num_change_list, full_list, empty_list = self.generate_orders(gene_t=count_t, single=True)
                # num_change
                self.apply_num_change(num_change_list)
                # full_loss_record
                sum_full_loss = sum(full_list)
                self.full_loss += sum_full_loss
                # empty_loss_record
                sum_empty_loss = sum(empty_list)
                self.empty_loss += sum_empty_loss

                self.t += operation_duration

    def generate_orders(self, gene_t=MIN_STEP, single=False):
        """
        生成 time min 内订单

        :param gene_t: 生成xx分钟内的订单
        :param single: 是否只考虑单平台的订单, True表示是, False表示否
        :return: list(num_change_list), list(success_list), list(success_opponent_list), list(full_list)
        """
        if not single:  # single=False
            num_change_list, success_list, success_opponent_list, full_list, empty_list = [], [], [], [], []
            for station in self.stations.keys():
                arr_s, arr_c = np.random.poisson(
                    sum(self.lambda_s_array[int(self.t / MIN_STEP):int((self.t + gene_t) / MIN_STEP), station - 1])
                ), np.random.poisson(
                    sum(self.lambda_c_array[int(self.t / MIN_STEP):int((self.t + gene_t) / MIN_STEP), station - 1])
                )
                dep = np.random.poisson(
                    sum(self.mu_array[int(self.t / MIN_STEP): int((self.t + gene_t) / MIN_STEP), station - 1])
                )
                if arr_s + self.stations[station].num_self > self.stations[station].cap:
                    num_s = self.stations[station].cap
                    full_list.append(arr_s + self.stations[station].num_self - self.stations[station].cap)
                else:
                    num_s = int(arr_s + self.stations[station].num_self)
                    full_list.append(0)
                num_c = int(min(self.stations[station].num_opponent + arr_c, self.stations[station].cap_opponent))
                bike_list = [1 for _ in range(num_s)] + [0 for _ in range(num_c)]
                if len(bike_list) < 0.1:
                    empty_list.append(dep)
                else:
                    empty_list.append(0)
                if len(bike_list) >= dep:
                    dep_s = sum(random.sample(bike_list, dep))
                    dep_c = dep - dep_s
                else:
                    dep_s = self.stations[station].num_self
                    dep_c = self.stations[station].num_opponent
                success_list.append(dep_s)
                success_opponent_list.append(dep_c)
                num_change_list.append(
                    (num_s - dep_s - self.stations[station].num_self,
                     num_c - dep_c - self.stations[station].num_opponent))
            return \
                list(num_change_list), list(success_list), list(success_opponent_list), list(full_list), list(
                    empty_list)

        else:  # single=True
            assert self.single
            num_change_list, full_list, empty_list = [], [], []
            for station in self.stations.keys():
                arr_s = np.random.poisson(
                    sum(self.lambda_s_array[int(self.t / MIN_STEP):int((self.t + gene_t) / MIN_STEP), station - 1]))
                dep = np.random.poisson(
                    sum(self.mu_s_array[int(self.t / MIN_STEP):int((self.t + gene_t) / MIN_STEP), station - 1]))
                if arr_s + self.stations[station].num_self > self.stations[station].cap:
                    num_s = self.stations[station].cap
                    full_list.append(arr_s + self.stations[station].num_self - self.stations[station].cap)
                else:
                    num_s = int(arr_s + self.stations[station].num_self)
                    full_list.append(0)
                if num_s < 0.1:
                    empty_list.append(dep)
                else:
                    empty_list.append(0)
                dep_s = min(num_s, dep)
                num_change_list.append((num_s - dep_s - self.stations[station].num_self, 0))

            return list(num_change_list), list(full_list), list(empty_list)

    def apply_num_change(self, num_change_list):
        for station in self.stations.keys():
            self.stations[station].change_num(num_change_list[station - 1])

    def step_multi_info(self, end_t: int):
        """
        步进函数 for multi-information

        :return:
        """
        end_t += self.t
        while self.t < end_t and self.t < self.sim_end_time:
            # simulation log for current time
            self._log.append(self.simulation_log_format(self.stations))
            num_change_list, success_list, success_opponent_list, full_list, empty_list = self.generate_orders()
            # num_change
            self.apply_num_change(num_change_list)
            # success_record
            sum_success = sum(success_list)
            self.success += sum_success
            self.success_list.append(sum_success)
            if self.t >= RECORD_WORK_T:
                self.success_work += sum_success
                self.success_work_list.append(sum_success)
                # if self.single is False:
                #     if round((self.t - RE_START_T) / MIN_RUN_STEP) < 36:
                #         self.test_esd += \
                #             sum([self.esd_arr[s - 1, round((self.t - RE_START_T) / MIN_RUN_STEP), round(
                #                 (self.t + MIN_RUN_STEP - RE_START_T) / MIN_RUN_STEP), self.stations[s].num_self,
                #             self.stations[
                #                 s].num_opponent] for s in range(1, 26)])
                #         self.test_esd_till_work_done += \
                #             sum([self.esd_arr[s - 1, round((self.t - RE_START_T) / MIN_RUN_STEP), round(
                #                 (self.t + MIN_RUN_STEP - RE_START_T) / MIN_RUN_STEP), self.stations[s].num_self,
                #             self.stations[
                #                 s].num_opponent] for s in range(1, 26)])
                #
                #     elif round((self.t - RE_START_T) / MIN_RUN_STEP) == 36:
                #         self.test_esd += \
                #             sum([self.esd_arr[s - 1, round((self.t - RE_START_T) / MIN_RUN_STEP) - 1, -1, self.stations[
                #                 s].num_self,
                #             self.stations[s].num_opponent] for s in range(1, 26)])
                # else:
                #     if round((self.t - RE_START_T) / MIN_RUN_STEP) < 36:
                #         self.test_esd += \
                #             sum([self.esd_arr[s - 1, round((self.t - RE_START_T) / MIN_RUN_STEP), round(
                #                 (self.t + MIN_RUN_STEP - RE_START_T) / 10), self.stations[s].num_self] for s in range(1, 26)])
                #         self.test_esd_till_work_done += \
                #             sum([self.esd_arr[s - 1, round((self.t - RE_START_T) / MIN_RUN_STEP), round(
                #                 (self.t + MIN_RUN_STEP - RE_START_T) / 10), self.stations[s].num_self] for s in
                #                  range(1, 26)])
                #
                #     elif round((self.t - RE_START_T) / MIN_RUN_STEP) == 36:
                #         self.test_esd += \
                #             sum([self.esd_arr[s - 1, round((self.t - RE_START_T) / MIN_RUN_STEP) - 1, -1, self.stations[s].num_self] for s in range(1, 26)])

                if self.t < RE_END_T:
                    self.success_work_till_done += sum_success
                    self.success_work_till_done_list.append(sum_success)
            else:
                self.success_work_list.append(0)
            # success_opponent_record
            sum_success_oppo = sum(success_opponent_list)
            self.success_opponent += sum_success_oppo
            self.success_opponent_list.append(sum_success_oppo)
            # full_loss_record
            sum_full_loss = sum(full_list)
            self.full_loss += sum_full_loss
            self.full_loss_list.append(sum_full_loss)
            # empty_loss_record
            sum_empty_loss = sum(empty_list)
            self.empty_loss += sum_empty_loss
            self.empty_loss_list.append(sum_empty_loss)
            # step forward
            self.t += MIN_STEP
            # update time left
            for veh in range(len(self.veh_info)):
                if self.veh_info[veh][1] is not None:
                    self.veh_info[veh][2] -= MIN_STEP
                    assert self.veh_info[veh][2] >= 0
        # vehicle statistics
        for veh in range(self.num_of_veh):
            if self.veh_info[veh][1] is not None and self.veh_info[veh][2] == 0:  # arrived
                move_dist = (int((self.dist[self.veh_info[veh][0], self.veh_info[veh][1]] - 0.2) / MIN_RUN_STEP) + 1) * MIN_RUN_STEP \
                    if self.veh_info[veh][0] != self.veh_info[veh][1] else 0
                self.veh_distance[veh] += move_dist
                self.veh_info[veh][0] = self.veh_info[veh][1]  # current_loc = next_loc
            if self.t > RE_END_T and self.return_count_time[veh] < 0.8:
                if self.veh_info[veh][2] == 0:  # arrived
                    return_dist = (int((self.dist[self.veh_info[veh][0], self.veh_info[veh][1]] - 0.2) / MIN_RUN_STEP) + 1) * MIN_RUN_STEP \
                        if self.veh_info[veh][0] != self.veh_info[veh][1] else 0
                    cur_station, cur_load = self.veh_info[veh][0], self.veh_info[veh][3]
                    assert cur_station > 0 if self.policy is not None else True
                    # put all the bikes at current station
                    if self.policy is not None:
                        self.stations[cur_station].num_self += cur_load
                        self.veh_info[veh][3] = 0
                    self.veh_distance[veh] += return_dist
                    self.return_count_time[veh] += 1
        self._log.append(self.simulation_log_format(self.stations))

    def run(self):
        """
        仿真运行主函数

        :return:
        """
        # change stage_info and simulation log
        self.stage_info.append(self.stage_info_format(0, 0,
                                                      [info[0] for info in self.veh_info],
                                                      [info[1] for info in self.veh_info],
                                                      [info[3] for info in self.veh_info]))
        self._log.append(self.simulation_log_format(self.stations))

        # start simulation
        while self.t < self.sim_end_time:

            if self.t:
                self.stage += 1
            self.stage_info.append(
                self.stage_info_format(
                    stage=self.stage, time=self.t, veh_loc=[info[0] for info in self.veh_info],
                    veh_next_loc=[info[1] for info in self.veh_info], veh_load=[info[3] for info in self.veh_info]))

            if RE_START_T <= self.t <= RE_END_T:
                # decisions at current stage
                dec_start = time.process_time()
                assert isinstance(self.single, bool)
                # dec_dict_format: [{'inv: int, 'route': int}, None] (dict for decision, None for no decision)
                if not self.single:  # multi-info
                    dec_list = self.decide_action_multi_info()
                else:  # single-info
                    dec_list = self.decide_action_single_info()
                dec_end = time.process_time()
                inv_dec_list, route_dec_list = ([i['inv'] for i in dec_list if i is not None],
                                                [i['route'] for i in dec_list if i is not None])
                t_left_list = [veh[2] for veh in self.veh_info]
                if self.print_action:
                    if self.t == 1200:
                        logging.debug(f'1200: {inv_dec_list}, {route_dec_list}')
                    for veh in range(len(self.veh_info)):
                        if inv_dec_list[veh] is not None:
                            assert t_left_list[veh] == 0 or t_left_list[veh] is None  # first entry
                            inv_dec, route_dec = inv_dec_list[veh], route_dec_list[veh]
                            if self.veh_info[veh][0] > 0:
                                print(
                                    f'({int(dec_end - dec_start)}s) Vehicle {veh}: Decision done at {int((self.t / 60 * 100)) / 100} ' +
                                    f'with inv={inv_dec}(cur_inv={round(self.stations[self.veh_info[veh][0]].num_self)}/{self.stations[self.veh_info[veh][0]].num_opponent}) ' +
                                    f'and route={route_dec}(from station={self.veh_info[veh][0]}) and vehicle load={self.veh_info[veh][3]} ' +
                                    f'before operation.')
                            else:
                                print(
                                    f'({int(dec_end - dec_start)}s) Vehicle {veh}: Decision done at {int((self.t / 60 * 100)) / 100} ' +
                                    f'and route={route_dec}(from depot) at depot')
                # change next_loc and load in apply_decision
                self.apply_decision_multi_info(inv_dec_list=inv_dec_list, route_dec_list=route_dec_list)
                t_dec = self.decide_time()  # 向前步进若干步，单位：min
            elif self.t > RE_END_T:
                t_dec = STAY_TIME
                for veh in range(self.num_of_veh):
                    if self.veh_info[veh][2] == 0:
                        self.veh_info[veh][2] += STAY_TIME
                num_self_list = [val.num_self for val in self.stations.values()]
                num_oppo_list = [val.num_opponent for val in self.stations.values()]
                self.cost_after_work += ORDER_INCOME_UNIT * self.get_estimated_order(step_t=t_dec,
                                                                                     num_self=num_self_list,
                                                                                     num_oppo=num_oppo_list,
                                                                                     start_t=self.t)
            else:
                t_dec = STAY_TIME

            self.stage_info.append(
                self.stage_info_format(
                    stage=self.stage, time=self.t, veh_loc=[info[0] for info in self.veh_info],
                    veh_next_loc=[info[1] for info in self.veh_info], veh_load=[info[3] for info in self.veh_info]))
            self.step_multi_info(end_t=t_dec)

    def print_simulation_log(self):
        f = open("test_log/simulation_log.txt", "w")
        for line in self.log:
            f.write(str(line) + '\n')
        f.close()

    def print_stage_log(self):
        f = open("test_log/stage_log.txt", "w")
        for line in self.stage_info:
            f.write(str(line) + '\n')
        f.close()
