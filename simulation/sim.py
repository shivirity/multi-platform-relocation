import copy
import time
import random
import pickle
import numpy as np

from simulation.consts import *

random.seed(SEED)
np.random.seed(SEED)

with open(r'D:\Desktop\Multi-platform EBSS operations\multi-platform-relocation\expectation_calculation\ESD_array.pkl', 'rb') as f:
    esd_arr = pickle.load(f)


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
        # current_loc, next_loc, load
        self.veh_info = [0, None, 0]

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

        self.veh_distance = 0  # total distance of relocation vehicle
        self.return_count_time = 0  # number of times that vehicle returns to depot

        # policy
        # single is True: 'STR', 'rollout', 'GLA'
        # single is False: 'None', 'random', 'online_VFA'
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
        self.random_choice_to_init_B = False
        self.cost_list = []
        self.dec_time_list = []
        self.basis_func_property = []
        self.func_dict = kwargs['func_dict'] if 'func_dict' in kwargs.keys() else None
        self.MLP_model = kwargs['MLP_model'] if 'MLP_model' in kwargs.keys() else None
        self.cost_after_work = 0
        # self.func_var_dict = self.init_func_var_dict()
        # online VFA property
        self.best_val_list = []

        # test esd
        self.test_esd = 0

        # MLP test
        self.nn_var_list = ['time', 'veh_load', 'des_inv']
        for i in range(1, 26):
            self.nn_var_list.append(f'veh_loc_{i}')
        for i in range(1, 26):
            self.nn_var_list.append(f'num_self_{i}')
        for i in range(1, 26):
            self.nn_var_list.append(f'num_oppo_{i}')
        for i in range(1, 26):
            self.nn_var_list.append(f'orders_till_sim_end_{i}')
        for i in range(1, 26):
            self.nn_var_list.append(f'bikes_s_arr_till_sim_end{i}')
        for i in range(1, 26):
            self.nn_var_list.append(f'bikes_c_arr_till_sim_end{i}')

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
    def stage_info_format(stage, time, veh_loc, veh_next_loc, veh_load):
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

    def get_estimate_value_linear(self, inv_dec: int, route_dec: int) -> float:
        """返回离线训练时当前动作的总价值函数（估计）"""
        assert self.policy in ['offline_VFA_train', 'online_VFA']
        cur_station, cur_load = self.veh_info[0], self.veh_info[2]
        if cur_station:  # at station
            if inv_dec > self.stations[cur_station].num_self:
                ins = min(inv_dec - self.stations[cur_station].num_self, cur_load)
            elif inv_dec < self.stations[cur_station].num_self:
                ins = max(inv_dec - self.stations[cur_station].num_self, cur_load - VEH_CAP)
            else:
                ins = 0
            num_self_list = [val.num_self for val in self.stations.values()]
            num_oppo_list = [val.num_opponent for val in self.stations.values()]
            num_self_list[cur_station - 1] += ins
            on_route_t = 5 * (int((self.dist[
                                       cur_station, route_dec] - 0.2) / 5) + 1) if cur_station != route_dec else 0  # time on route
            cur_step_t = CONST_OPERATION + on_route_t if cur_station != route_dec else STAY_TIME
            # cost at current step
            order_exp = self.get_estimated_order(
                step_t=cur_step_t, num_self=num_self_list, num_oppo=num_oppo_list, start_t=self.t
            )
            # cost after current step
            cost_after = 0
            # instruction fix
            var_dict = self.get_post_decision_var_dict(inv_dec=inv_dec, route_dec=route_dec)
            # calculate cost after
            policy_t_idx = int(self.t / POLICY_DURATION)
            for key, value in self.func_dict[policy_t_idx].items():
                cost_after += value * var_dict[key]
            # route cost
            route_cost = on_route_t * DISTANCE_COST_UNIT

            return ORDER_INCOME_UNIT * order_exp + cost_after - route_cost

        else:  # at depot
            assert inv_dec < 0 and route_dec != 0
            num_self_list = [val.num_self for val in self.stations.values()]
            num_oppo_list = [val.num_opponent for val in self.stations.values()]
            cur_step_t = 5 * (int((self.dist[cur_station, route_dec] - 0.2) / 5) + 1)  # time on route
            # cost at current step
            order_exp = self.get_estimated_order(
                step_t=cur_step_t, num_self=num_self_list, num_oppo=num_oppo_list, start_t=self.t
            )
            # cost after current step
            veh_load, cost_after = cur_load, 0
            # instruction fix
            var_dict = self.get_post_decision_var_dict(inv_dec=inv_dec, route_dec=route_dec)
            # calculate cost after
            policy_t_idx = int(self.t / POLICY_DURATION)
            for key, value in self.func_dict[policy_t_idx].items():
                cost_after += value * var_dict[key]
            # route cost
            route_cost = cur_step_t * DISTANCE_COST_UNIT

            assert isinstance(cost_after - route_cost, float), f'cost_after={cost_after}, route_cost={route_cost}'
            return ORDER_INCOME_UNIT * order_exp + cost_after - route_cost

    def get_estimate_value_MLP(self, inv_dec: int, route_dec: int) -> float:
        assert self.single is False and self.policy == 'MLP_test'
        cur_station, cur_load = self.veh_info[0], self.veh_info[2]
        if cur_station:  # at station
            if inv_dec > self.stations[cur_station].num_self:
                ins = min(inv_dec - self.stations[cur_station].num_self, cur_load)
            elif inv_dec < self.stations[cur_station].num_self:
                ins = max(inv_dec - self.stations[cur_station].num_self, cur_load - VEH_CAP)
            else:
                ins = 0
            num_self_list = [val.num_self for val in self.stations.values()]
            num_oppo_list = [val.num_opponent for val in self.stations.values()]
            num_self_list[cur_station - 1] += ins
            on_route_t = 5 * (int((self.dist[
                                       cur_station, route_dec] - 0.2) / 5) + 1) if cur_station != route_dec else 0  # time on route
            cur_step_t = CONST_OPERATION + on_route_t if cur_station != route_dec else STAY_TIME
            # cost at current step
            order_exp = self.get_estimated_order(
                step_t=cur_step_t, num_self=num_self_list, num_oppo=num_oppo_list, start_t=self.t
            )
            # cost after current step
            cost_after = 0
            # instruction fix
            var_dict = self.get_post_decision_var_dict(inv_dec=inv_dec, route_dec=route_dec)
            # calculate cost after
            x_input = np.array([var_dict[key] for key in self.nn_var_list]).reshape(1, -1)
            cost_after = float(self.MLP_model.predict(x_input)[0])
            # route cost
            route_cost = on_route_t * DISTANCE_COST_UNIT

            return ORDER_INCOME_UNIT * order_exp + cost_after - route_cost

        else:  # at depot
            assert inv_dec < 0 and route_dec != 0
            num_self_list = [val.num_self for val in self.stations.values()]
            num_oppo_list = [val.num_opponent for val in self.stations.values()]
            cur_step_t = 5 * (int((self.dist[cur_station, route_dec] - 0.2) / 5) + 1)  # time on route
            # cost at current step
            order_exp = self.get_estimated_order(
                step_t=cur_step_t, num_self=num_self_list, num_oppo=num_oppo_list, start_t=self.t
            )
            # cost after current step
            veh_load, cost_after = cur_load, 0
            # instruction fix
            var_dict = self.get_post_decision_var_dict(inv_dec=inv_dec, route_dec=route_dec)
            # calculate cost after
            x_input = np.array([var_dict[key] for key in self.nn_var_list]).reshape(1, -1)
            cost_after = float(self.MLP_model.predict(x_input)[0])
            # route cost
            route_cost = cur_step_t * DISTANCE_COST_UNIT

            assert isinstance(cost_after - route_cost, float), f'cost_after={cost_after}, route_cost={route_cost}'
            return ORDER_INCOME_UNIT * order_exp + cost_after - route_cost

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

    def get_post_decision_var_dict(self, inv_dec: int, route_dec: int) -> dict:
        """返回离线训练时当前动作后的变量字典"""
        # todo 所有操作都要在计算post-value function的时候修正
        var_dict = dict(self.func_var_dict)
        cur_station, cur_load = self.veh_info[0], self.veh_info[2]
        if cur_station:  # at station
            if inv_dec > self.stations[cur_station].num_self:
                ins = min(inv_dec - self.stations[cur_station].num_self, cur_load)
            elif inv_dec < self.stations[cur_station].num_self:
                ins = max(inv_dec - self.stations[cur_station].num_self, cur_load - VEH_CAP)
            else:
                ins = 0

            veh_load = cur_load - ins
            if 'veh_load' in var_dict.keys():
                var_dict['veh_load'] = veh_load

        else:
            assert inv_dec < 0 and route_dec != 0
            ins = 0
            veh_load = cur_load
            if 'veh_load' in var_dict.keys():
                var_dict['veh_load'] = veh_load

        # binary route decision
        # for i in range(1, len(self.stations) + 1):
        #     var_dict[f'veh_des_{i}'] = 1 if route_dec == i else 0

        # binary vehicle location
        for i in range(1, len(self.stations) + 1):
            var_dict[f'veh_loc_{i}'] = 1 if cur_station == i else 0

        # destination station inventory
        var_dict[f'des_inv'] = self.stations[route_dec].num_self

        # time spent on the route
        on_route_t = 5 * (int((self.dist[
                                   cur_station, route_dec] - 0.2) / 5) + 1) if cur_station != route_dec else 0  # time on route
        cur_step_t = CONST_OPERATION + on_route_t if cur_station != route_dec else MIN_RUN_STEP
        # step time
        var_dict['step_t'] = cur_step_t
        var_dict['time'] = self.t + cur_step_t

        for i in range(1, len(self.stations) + 1):
            # orders till sim ends
            order = self.mu_array[int((self.t + cur_step_t) / MIN_STEP):int(SIM_END_T / MIN_STEP), i - 1].sum()
            next_2_hour_order = self.mu_array[
                                int((self.t + cur_step_t) / MIN_STEP):int((self.t + cur_step_t + 120) / MIN_STEP),
                                i - 1].sum()
            arr_s = self.lambda_s_array[int((self.t + cur_step_t) / MIN_STEP):int(SIM_END_T / MIN_STEP), i - 1].sum()
            arr_c = self.lambda_c_array[int((self.t + cur_step_t) / MIN_STEP):int(SIM_END_T / MIN_STEP), i - 1].sum()

            var_dict[f'bikes_s_arr_till_sim_end{i}'], var_dict[f'bikes_c_arr_till_sim_end{i}'] = arr_s, arr_c
            var_dict[f'orders_till_sim_end_{i}'] = order
            # number of bikes from our platform at stations (proportion)
            """
            if i != cur_station:
                if self.stations[i].num_self + self.stations[i].num_opponent > 0:
                    var_dict[f'self_order_proportion_till_ends_{i}'] = \
                        self.stations[i].num_self / (self.stations[i].num_self + self.stations[i].num_opponent) * order
                    var_dict[f'self_order_proportion_next_2_hours_{i}'] = \
                        self.stations[i].num_self / (
                                    self.stations[i].num_self + self.stations[i].num_opponent) * next_2_hour_order
                else:
                    var_dict[f'self_order_proportion_till_ends_{i}'] = 0
                    var_dict[f'self_order_proportion_next_2_hours_{i}'] = 0
            else:
                assert cur_station > 0
                if self.stations[i].num_self + self.stations[i].num_opponent + ins > 0:
                    var_dict[f'self_order_proportion_till_ends_{i}'] = \
                        (self.stations[i].num_self + ins) / (
                                    self.stations[i].num_self + ins + self.stations[i].num_opponent) * order
                    var_dict[f'self_order_proportion_next_2_hours_{i}'] = \
                        (self.stations[i].num_self + ins) / (
                                    self.stations[i].num_self + ins + self.stations[i].num_opponent) * next_2_hour_order
                else:
                    var_dict[f'self_order_proportion_till_ends_{i}'] = 0
                    var_dict[f'self_order_proportion_next_2_hours_{i}'] = 0
            """
        """
        for i in range(1, len(self.stations) + 1):
            next_2_hour_s_arr = self.lambda_s_array[
                                int((self.t + cur_step_t) / MIN_STEP):int((self.t + cur_step_t + 120) / MIN_STEP),
                                i - 1].sum()
            next_2_hour_c_arr = self.lambda_c_array[
                                int((self.t + cur_step_t) / MIN_STEP):int((self.t + cur_step_t + 120) / MIN_STEP),
                                i - 1].sum()
            next_2_hour_dep = self.mu_array[
                              int((self.t + cur_step_t) / MIN_STEP):int((self.t + cur_step_t + 120) / MIN_STEP),
                              i - 1].sum()
            if i != cur_station:
                if self.stations[i].num_self + next_2_hour_s_arr + self.stations[i].num_opponent + next_2_hour_c_arr > 0:
                    var_dict[f'net_demand_in_2_hours_{i}'] = \
                        next_2_hour_dep * \
                        ((self.stations[i].num_self + next_2_hour_s_arr) / (
                                self.stations[i].num_self + next_2_hour_s_arr + self.stations[
                            i].num_opponent + next_2_hour_c_arr)) \
                        - next_2_hour_s_arr
                else:
                    var_dict[f'net_demand_in_2_hours_{i}'] = 0
            else:
                if self.stations[i].num_self + ins + next_2_hour_s_arr + self.stations[i].num_opponent + next_2_hour_c_arr > 0:
                    var_dict[f'net_demand_in_2_hours_{i}'] = \
                        next_2_hour_dep * \
                        ((self.stations[i].num_self + ins + next_2_hour_s_arr) / (
                                self.stations[i].num_self + ins + next_2_hour_s_arr + self.stations[
                            i].num_opponent + next_2_hour_c_arr)) \
                        - next_2_hour_s_arr
                else:
                    var_dict[f'net_demand_in_2_hours_{i}'] = 0
        """

        # number of bikes from our platform at stations
        for i in range(1, len(self.stations) + 1):
            if i != cur_station:
                var_dict[f'num_self_{i}'] = self.stations[i].num_self
                var_dict[f'num_oppo_{i}'] = self.stations[i].num_opponent
            else:
                var_dict[f'num_self_{i}'] = self.stations[i].num_self + ins
                var_dict[f'num_oppo_{i}'] = self.stations[i].num_opponent

        return var_dict

    def decide_action_multi_info(self):
        """
        决策当前站点目标库存水平和下一站点决策

        :return: 决策字典, {'inv': inv_dec, 'route': route_dec}
        """
        # random
        if self.policy == 'random':
            cur_station = self.veh_info[0]
            if cur_station:
                inv_levels = [i * self.stations[cur_station].cap for i in DEC_LEVELS]
                inv_tmp, inv_state = [], []
                for i in range(len(inv_levels)):
                    if not i:
                        inv_tmp.append(inv_levels[i])
                        inv_state.append(
                            self.get_station_inv(self.stations[cur_station].num_self, inv_levels[i], self.veh_info[2]))
                    else:
                        inv_state_tmp = self.get_station_inv(
                            self.stations[cur_station].num_self, inv_levels[i], self.veh_info[2])
                        if inv_state_tmp not in inv_state:
                            inv_tmp.append(inv_levels[i])
                            inv_state.append(inv_state_tmp)
                inv_levels = inv_tmp

                inv_dec = random.sample(inv_levels, 1)[0]
            else:
                inv_dec = -1  # only happens when the vehicle is at depot
            route_dec = random.sample([i for i in self.stations.keys() if i != cur_station], 1)[0]

        # rollout
        elif self.policy == 'rollout':
            cur_station = self.veh_info[0]
            if cur_station:
                inv_levels = [i * self.stations[cur_station].cap for i in DEC_LEVELS]
                inv_tmp, inv_state = [], []
                for i in range(len(inv_levels)):
                    if not i:
                        inv_tmp.append(inv_levels[i])
                        inv_state.append(
                            self.get_station_inv(self.stations[cur_station].num_self, inv_levels[i], self.veh_info[2]))
                    else:
                        inv_state_tmp = self.get_station_inv(
                            self.stations[cur_station].num_self, inv_levels[i], self.veh_info[2])
                        if inv_state_tmp not in inv_state:
                            inv_tmp.append(inv_levels[i])
                            inv_state.append(inv_state_tmp)
                inv_levels = inv_tmp

                route_choose = [i for i in self.stations.keys()]
                route_success = {}
                for inv in inv_levels:
                    for route in route_choose:
                        rep_sim_h_success = []
                        for _ in range(ROLLOUT_SIM_TIMES):
                            rep_sim = copy.deepcopy(self)
                            rep_sim.apply_decision_multi_info(inv_dec=inv, route_dec=route)
                            t_dec = rep_sim.decide_time(route)  # 向前步进若干步, in rollout, 单位：min
                            rep_sim.step_multi_info(end_t=t_dec)
                            rep_success = rep_sim.run_replication_multi_info(base_policy='multi-STR')
                            rep_sim_h_success.append(rep_success)
                        route_success[(inv, route)] = np.mean(np.array(rep_sim_h_success, dtype=np.single))
                route_dec = max(route_success, key=lambda x: route_success[x])
                inv_dec, route_dec = route_dec[0], route_dec[1]
            else:
                route_success = []
                route_choose = [i for i in self.stations.keys()]
                for route in route_choose:
                    rep_sim = copy.deepcopy(self)
                    rep_sim.apply_decision_multi_info(-1, route)
                    rep_sim_h_success = []
                    for _ in range(ROLLOUT_SIM_TIMES):
                        t_dec = rep_sim.decide_time(route)  # 向前步进若干步, in rollout, 单位：min
                        rep_sim.step_multi_info(end_t=t_dec)
                        rep_success = rep_sim.run_replication_multi_info(base_policy='multi-STR')
                        rep_sim_h_success.append(rep_success)
                    route_success.append(np.mean(np.array(rep_sim_h_success, dtype=np.single)))
                route_dec = route_success.index(max(route_success))
                inv_dec = -1

        # do nothing
        elif self.policy is None:
            cur_station = self.veh_info[0]
            inv_dec, route_dec = -1, cur_station

        # short-term relocation
        elif self.policy == 'multi-STR':  # no new rules has been added
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
            else:
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

        # offline value function approximation training process
        elif self.policy == 'offline_VFA_train':
            cur_station, cur_load = self.veh_info[0], self.veh_info[2]
            if cur_station:  # at stations
                # choice set (without decision levels)
                inv_options = [i for i in range(int(SAFETY_INV_LB * self.stations[cur_station].cap),
                                                int(SAFETY_INV_UB * self.stations[cur_station].cap) + 1)]
                min_inv_option = max(0, cur_load + self.stations[cur_station].num_self - VEH_CAP)
                max_inv_option = min(self.stations[cur_station].cap, cur_load + self.stations[cur_station].num_self)
                inv_options = [i for i in inv_options if min_inv_option <= i <= max_inv_option]
                station_options = [i for i in self.stations.keys()]
                best_dec, best_val = None, -np.inf
                for inv in inv_options:
                    for station in station_options:
                        # cost + estimated value
                        est_val = self.get_estimate_value_linear(inv_dec=inv, route_dec=station)
                        if est_val > best_val:
                            best_dec, best_val = (inv, station), est_val
                # epsilon-greedy
                if random.random() < EPSILON or self.random_choice_to_init_B:
                    inv_dec, route_dec = \
                        int(random.sample(inv_options, 1)[0]), int(random.sample(station_options, 1)[0])
                else:
                    inv_dec, route_dec = best_dec[0], best_dec[1]

                post_dec_var_dict = self.get_post_decision_var_dict(inv_dec=inv_dec, route_dec=route_dec)
                # estimate current cost
                if inv_dec > self.stations[cur_station].num_self:
                    ins = min(inv_dec - self.stations[cur_station].num_self, cur_load)
                elif inv_dec < self.stations[cur_station].num_self:
                    ins = max(inv_dec - self.stations[cur_station].num_self, cur_load - VEH_CAP)
                else:
                    ins = 0
                num_self_list = [val.num_self for val in self.stations.values()]
                num_oppo_list = [val.num_opponent for val in self.stations.values()]
                num_self_list[cur_station - 1] += ins
                on_route_t = 5 * (int((self.dist[
                                           cur_station, route_dec] - 0.2) / 5) + 1) if cur_station != route_dec else 0
                cur_step_t = CONST_OPERATION + on_route_t if cur_station != route_dec else MIN_RUN_STEP
                # cost at current step
                order_exp = self.get_estimated_order(
                    step_t=cur_step_t, num_self=num_self_list, num_oppo=num_oppo_list, start_t=self.t
                )
                self.cost_list.append(ORDER_INCOME_UNIT * order_exp - UNIT_TRAVEL_COST * on_route_t)
                self.basis_func_property.append(dict(post_dec_var_dict))
                self.dec_time_list.append(self.t)

            else:  # at depot
                inv_dec = -1
                station_options = [i for i in self.stations.keys()]
                best_dec, best_val = None, -np.inf
                for station in station_options:
                    est_val = self.get_estimate_value_linear(inv_dec=inv_dec, route_dec=station)
                    if est_val > best_val:
                        best_dec, best_val = station, est_val
                # epsilon-greedy
                if random.random() < EPSILON or self.random_choice_to_init_B:
                    route_dec = int(random.sample(station_options, 1)[0])
                else:
                    route_dec = best_dec

                post_dec_var_dict = self.get_post_decision_var_dict(inv_dec=inv_dec, route_dec=route_dec)

                num_self_list = [val.num_self for val in self.stations.values()]
                num_oppo_list = [val.num_opponent for val in self.stations.values()]
                cur_step_t = 5 * (int((self.dist[cur_station, route_dec] - 0.2) / 5) + 1)  # time on route
                # cost at current step
                order_exp = self.get_estimated_order(
                    step_t=cur_step_t, num_self=num_self_list, num_oppo=num_oppo_list, start_t=self.t
                )

                self.cost_list.append(ORDER_INCOME_UNIT * order_exp - UNIT_TRAVEL_COST * cur_step_t)
                self.basis_func_property.append(dict(post_dec_var_dict))
                self.dec_time_list.append(self.t)

        elif self.policy == 'online_VFA':
            cur_station, cur_load = self.veh_info[0], self.veh_info[2]
            if cur_station:  # at stations
                # choice set (without decision levels)
                inv_options = [i for i in range(int(SAFETY_INV_LB * self.stations[cur_station].cap),
                                                int(SAFETY_INV_UB * self.stations[cur_station].cap) + 1)]
                min_inv_option = max(0, cur_load + self.stations[cur_station].num_self - VEH_CAP)
                max_inv_option = min(self.stations[cur_station].cap, cur_load + self.stations[cur_station].num_self)
                inv_options = [i for i in inv_options if min_inv_option <= i <= max_inv_option]
                station_options = [i for i in self.stations.keys()]
                best_dec, best_val = None, -np.inf
                for inv in inv_options:
                    for station in station_options:
                        # cost + estimated value
                        est_val = self.get_estimate_value_linear(inv_dec=inv, route_dec=station)
                        if est_val > best_val:
                            best_dec, best_val = (inv, station), est_val
                self.best_val_list.append(best_val + sum(self.cost_list))
                inv_dec, route_dec = best_dec[0], best_dec[1]

                # estimate current cost
                if inv_dec > self.stations[cur_station].num_self:
                    ins = min(inv_dec - self.stations[cur_station].num_self, cur_load)
                elif inv_dec < self.stations[cur_station].num_self:
                    ins = max(inv_dec - self.stations[cur_station].num_self, cur_load - VEH_CAP)
                else:
                    ins = 0
                num_self_list = [val.num_self for val in self.stations.values()]
                num_oppo_list = [val.num_opponent for val in self.stations.values()]
                num_self_list[cur_station - 1] += ins
                on_route_t = 5 * (int((self.dist[
                                           cur_station, route_dec] - 0.2) / 5) + 1) if cur_station != route_dec else 0
                cur_step_t = CONST_OPERATION + on_route_t if cur_station != route_dec else MIN_RUN_STEP
                # cost at current step
                order_exp = self.get_estimated_order(
                    step_t=cur_step_t, num_self=num_self_list, num_oppo=num_oppo_list, start_t=self.t
                )
                self.cost_list.append(ORDER_INCOME_UNIT * order_exp - UNIT_TRAVEL_COST * on_route_t)

            else:  # at depot
                inv_dec = -1
                station_options = [i for i in self.stations.keys()]
                best_dec, best_val = None, -np.inf
                for station in station_options:
                    est_val = self.get_estimate_value_linear(inv_dec=inv_dec, route_dec=station)
                    if est_val > best_val:
                        best_dec, best_val = station, est_val
                self.best_val_list.append(best_val + sum(self.cost_list))
                route_dec = best_dec

                # estimate current cost
                num_self_list = [val.num_self for val in self.stations.values()]
                num_oppo_list = [val.num_opponent for val in self.stations.values()]
                cur_step_t = 5 * (int((self.dist[cur_station, route_dec] - 0.2) / 5) + 1)  # time on route
                # cost at current step
                order_exp = self.get_estimated_order(
                    step_t=cur_step_t, num_self=num_self_list, num_oppo=num_oppo_list, start_t=self.t
                )

                self.cost_list.append(ORDER_INCOME_UNIT * order_exp - UNIT_TRAVEL_COST * cur_step_t)

        elif self.policy == 'MLP_test':
            cur_station, cur_load = self.veh_info[0], self.veh_info[2]
            if cur_station:  # at stations
                # choice set (without decision levels)
                inv_options = [i for i in range(int(SAFETY_INV_LB * self.stations[cur_station].cap),
                                                int(SAFETY_INV_UB * self.stations[cur_station].cap) + 1)]
                min_inv_option = max(0, cur_load + self.stations[cur_station].num_self - VEH_CAP)
                max_inv_option = min(self.stations[cur_station].cap, cur_load + self.stations[cur_station].num_self)
                inv_options = [i for i in inv_options if min_inv_option <= i <= max_inv_option]
                station_options = [i for i in self.stations.keys()]
                best_dec, best_val = None, -np.inf
                for inv in inv_options:
                    for station in station_options:
                        # cost + estimated value
                        est_val = self.get_estimate_value_MLP(inv_dec=inv, route_dec=station)
                        if est_val > best_val:
                            best_dec, best_val = (inv, station), est_val
                self.best_val_list.append(best_val + sum(self.cost_list))
                inv_dec, route_dec = best_dec[0], best_dec[1]

                # estimate current cost
                if inv_dec > self.stations[cur_station].num_self:
                    ins = min(inv_dec - self.stations[cur_station].num_self, cur_load)
                elif inv_dec < self.stations[cur_station].num_self:
                    ins = max(inv_dec - self.stations[cur_station].num_self, cur_load - VEH_CAP)
                else:
                    ins = 0
                num_self_list = [val.num_self for val in self.stations.values()]
                num_oppo_list = [val.num_opponent for val in self.stations.values()]
                num_self_list[cur_station - 1] += ins
                on_route_t = 5 * (int((self.dist[
                                           cur_station, route_dec] - 0.2) / 5) + 1) if cur_station != route_dec else 0
                cur_step_t = CONST_OPERATION + on_route_t if cur_station != route_dec else MIN_RUN_STEP
                # cost at current step
                order_exp = self.get_estimated_order(
                    step_t=cur_step_t, num_self=num_self_list, num_oppo=num_oppo_list, start_t=self.t
                )
                self.cost_list.append(ORDER_INCOME_UNIT * order_exp - UNIT_TRAVEL_COST * on_route_t)

            else:  # at depot
                inv_dec = -1
                station_options = [i for i in self.stations.keys()]
                best_dec, best_val = None, -np.inf
                for station in station_options:
                    est_val = self.get_estimate_value_MLP(inv_dec=inv_dec, route_dec=station)
                    if est_val > best_val:
                        best_dec, best_val = station, est_val
                self.best_val_list.append(best_val + sum(self.cost_list))
                route_dec = best_dec

                # estimate current cost
                num_self_list = [val.num_self for val in self.stations.values()]
                num_oppo_list = [val.num_opponent for val in self.stations.values()]
                cur_step_t = 5 * (int((self.dist[cur_station, route_dec] - 0.2) / 5) + 1)  # time on route
                # cost at current step
                order_exp = self.get_estimated_order(
                    step_t=cur_step_t, num_self=num_self_list, num_oppo=num_oppo_list, start_t=self.t
                )

                self.cost_list.append(ORDER_INCOME_UNIT * order_exp - UNIT_TRAVEL_COST * cur_step_t)

        else:
            print('policy type error.')
            assert False

        assert route_dec is not None

        return {'inv': inv_dec, 'route': route_dec}

    def decide_action_single_info(self):
        """
        决策当前站点目标库存水平和下一站点决策（单平台信息）

        :return: 决策字典, {'inv': inv_dec, 'route': route_dec}
        """
        if self.policy == 'STR':
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

        elif self.policy == 'offline_VFA_train':
            cur_station, cur_load = self.veh_info[0], self.veh_info[2]
            if cur_station:  # at stations
                # choice set (without decision levels)
                inv_options = [i for i in range(int(SAFETY_INV_LB * self.stations[cur_station].cap),
                                                int(SAFETY_INV_UB * self.stations[cur_station].cap) + 1)]
                min_inv_option = max(0, cur_load + self.stations[cur_station].num_self - VEH_CAP)
                max_inv_option = min(self.stations[cur_station].cap, cur_load + self.stations[cur_station].num_self)
                inv_options = [i for i in inv_options if min_inv_option <= i <= max_inv_option]
                station_options = [i for i in self.stations.keys()]
                best_dec, best_val = None, -np.inf
                for inv in inv_options:
                    for station in station_options:
                        # cost + estimated value
                        est_val = self.get_estimate_value_linear(inv_dec=inv, route_dec=station)
                        if est_val > best_val:
                            best_dec, best_val = (inv, station), est_val
                # epsilon-greedy
                if random.random() < EPSILON or self.random_choice_to_init_B:
                    inv_dec, route_dec = \
                        int(random.sample(inv_options, 1)[0]), int(random.sample(station_options, 1)[0])
                else:
                    inv_dec, route_dec = best_dec[0], best_dec[1]

                post_dec_var_dict = self.get_post_decision_var_dict(inv_dec=inv_dec, route_dec=route_dec)
                # estimate current cost
                if inv_dec > self.stations[cur_station].num_self:
                    ins = min(inv_dec - self.stations[cur_station].num_self, cur_load)
                elif inv_dec < self.stations[cur_station].num_self:
                    ins = max(inv_dec - self.stations[cur_station].num_self, cur_load - VEH_CAP)
                else:
                    ins = 0
                num_self_list = [val.num_self for val in self.stations.values()]
                num_oppo_list = [val.num_opponent for val in self.stations.values()]
                num_self_list[cur_station - 1] += ins
                on_route_t = 5 * (int((self.dist[
                                           cur_station, route_dec] - 0.2) / 5) + 1) if cur_station != route_dec else 0
                cur_step_t = CONST_OPERATION + on_route_t if cur_station != route_dec else MIN_RUN_STEP
                # cost at current step
                order_exp = self.get_estimated_order(
                    step_t=cur_step_t, num_self=num_self_list, num_oppo=num_oppo_list, start_t=self.t
                )
                self.cost_list.append(ORDER_INCOME_UNIT * order_exp - UNIT_TRAVEL_COST * on_route_t)
                self.basis_func_property.append(dict(post_dec_var_dict))
                self.dec_time_list.append(self.t)

            else:  # at depot
                inv_dec = -1
                station_options = [i for i in self.stations.keys()]
                best_dec, best_val = None, -np.inf
                for station in station_options:
                    est_val = self.get_estimate_value_linear(inv_dec=inv_dec, route_dec=station)
                    if est_val > best_val:
                        best_dec, best_val = station, est_val
                # epsilon-greedy
                if random.random() < EPSILON or self.random_choice_to_init_B:
                    route_dec = int(random.sample(station_options, 1)[0])
                else:
                    route_dec = best_dec

                post_dec_var_dict = self.get_post_decision_var_dict(inv_dec=inv_dec, route_dec=route_dec)

                num_self_list = [val.num_self for val in self.stations.values()]
                num_oppo_list = [val.num_opponent for val in self.stations.values()]
                cur_step_t = 5 * (int((self.dist[cur_station, route_dec] - 0.2) / 5) + 1)  # time on route
                # cost at current step
                order_exp = self.get_estimated_order(
                    step_t=cur_step_t, num_self=num_self_list, num_oppo=num_oppo_list, start_t=self.t
                )

                self.cost_list.append(ORDER_INCOME_UNIT * order_exp - UNIT_TRAVEL_COST * cur_step_t)
                self.basis_func_property.append(dict(post_dec_var_dict))
                self.dec_time_list.append(self.t)

        elif self.policy == 'online_VFA':
            cur_station, cur_load = self.veh_info[0], self.veh_info[2]
            if cur_station:  # at stations
                # choice set (without decision levels)
                inv_options = [i for i in range(int(SAFETY_INV_LB * self.stations[cur_station].cap),
                                                int(SAFETY_INV_UB * self.stations[cur_station].cap) + 1)]
                min_inv_option = max(0, cur_load + self.stations[cur_station].num_self - VEH_CAP)
                max_inv_option = min(self.stations[cur_station].cap, cur_load + self.stations[cur_station].num_self)
                inv_options = [i for i in inv_options if min_inv_option <= i <= max_inv_option]
                station_options = [i for i in self.stations.keys()]
                best_dec, best_val = None, -np.inf
                for inv in inv_options:
                    for station in station_options:
                        # cost + estimated value
                        est_val = self.get_estimate_value_linear(inv_dec=inv, route_dec=station)
                        if est_val > best_val:
                            best_dec, best_val = (inv, station), est_val
                self.best_val_list.append(best_val + sum(self.cost_list))
                inv_dec, route_dec = best_dec[0], best_dec[1]

                # estimate current cost
                if inv_dec > self.stations[cur_station].num_self:
                    ins = min(inv_dec - self.stations[cur_station].num_self, cur_load)
                elif inv_dec < self.stations[cur_station].num_self:
                    ins = max(inv_dec - self.stations[cur_station].num_self, cur_load - VEH_CAP)
                else:
                    ins = 0
                num_self_list = [val.num_self for val in self.stations.values()]
                num_oppo_list = [val.num_opponent for val in self.stations.values()]
                num_self_list[cur_station - 1] += ins
                on_route_t = 5 * (int((self.dist[
                                           cur_station, route_dec] - 0.2) / 5) + 1) if cur_station != route_dec else 0
                cur_step_t = CONST_OPERATION + on_route_t if cur_station != route_dec else MIN_RUN_STEP
                # cost at current step
                order_exp = self.get_estimated_order(
                    step_t=cur_step_t, num_self=num_self_list, num_oppo=num_oppo_list, start_t=self.t
                )
                self.cost_list.append(ORDER_INCOME_UNIT * order_exp - UNIT_TRAVEL_COST * on_route_t)

            else:  # at depot
                inv_dec = -1
                station_options = [i for i in self.stations.keys()]
                best_dec, best_val = None, -np.inf
                for station in station_options:
                    est_val = self.get_estimate_value_linear(inv_dec=inv_dec, route_dec=station)
                    if est_val > best_val:
                        best_dec, best_val = station, est_val
                self.best_val_list.append(best_val + sum(self.cost_list))
                route_dec = best_dec

                # estimate current cost
                num_self_list = [val.num_self for val in self.stations.values()]
                num_oppo_list = [val.num_opponent for val in self.stations.values()]
                cur_step_t = 5 * (int((self.dist[cur_station, route_dec] - 0.2) / 5) + 1)  # time on route
                # cost at current step
                order_exp = self.get_estimated_order(
                    step_t=cur_step_t, num_self=num_self_list, num_oppo=num_oppo_list, start_t=self.t
                )

                self.cost_list.append(ORDER_INCOME_UNIT * order_exp - UNIT_TRAVEL_COST * cur_step_t)

        elif self.policy == 'rollout':  # 2019 COR
            cur_station = self.veh_info[0]
            if cur_station:
                inv_levels = [i * self.stations[cur_station].cap for i in DEC_LEVELS]
                inv_tmp, inv_state = [], []
                # 缩小决策空间, 去除产生同样库存的决策水平
                for i in range(len(inv_levels)):
                    if not i:
                        inv_tmp.append(inv_levels[i])
                        inv_state.append(
                            self.get_station_inv(self.stations[cur_station].num_self, inv_levels[i], self.veh_info[2]))
                    else:
                        inv_state_tmp = self.get_station_inv(
                            self.stations[cur_station].num_self, inv_levels[i], self.veh_info[2])
                        if inv_state_tmp not in inv_state:
                            inv_tmp.append(inv_levels[i])
                            inv_state.append(inv_state_tmp)
                inv_levels = inv_tmp

                inv_fail, station_rental, station_return = \
                    [], [0 for _ in range(len(self.stations.keys()))], [0 for _ in range(len(self.stations.keys()))]
                for inv in inv_levels:
                    # len(ROLLOUT_SIM_TIMES)
                    rep_inv_fail = []
                    for _ in range(ROLLOUT_SIM_TIMES):
                        rep_sim = copy.deepcopy(self)
                        rep_sim.sim_end_time = self.t + SINGLE_ROLLOUT_HORIZON
                        rep_sim.apply_decision_single_info(inv_dec=inv, route_dec=cur_station)
                        rep_sim.single_full_list, rep_sim.single_empty_list = [], []
                        rep_fail, rep_rental, rep_return = rep_sim.run_replication_single_info(base_policy=None)
                        rep_inv_fail.append(rep_fail)
                        station_rental = [a + b for a, b in zip(station_rental, rep_rental)]
                        station_return = [a + b for a, b in zip(station_return, rep_return)]
                    inv_fail.append(sum(rep_inv_fail))
                station_rental = [val / (len(inv_levels) * ROLLOUT_SIM_TIMES) for val in station_rental]
                station_return = [val / (len(inv_levels) * ROLLOUT_SIM_TIMES) for val in station_return]

                inv_dec = inv_levels[inv_fail.index(min(inv_fail))]
                pre_rental = self.stations[cur_station].num_self + self.veh_info[2] - \
                             self.get_station_inv(self.stations[cur_station].num_self, inv_dec, self.veh_info[2])
                pre_return = VEH_CAP - pre_rental
                station_rental, station_return = \
                    [min(val, pre_rental) for val in station_rental], [min(val, pre_return) for val in station_return]
                station_max = [max(a, b) for a, b in zip(station_rental, station_return)]
                station_max[cur_station - 1] = -32
                default_station = list(self.stations.keys())
                random.shuffle(default_station)
                default_max = [station_max[val - 1] for val in default_station]
                route_dec = default_station[default_max.index(max(default_max))]
            else:
                station_rental, station_return = \
                    [0 for _ in range(len(self.stations.keys()))], [0 for _ in range(len(self.stations.keys()))]
                for _ in range(ROLLOUT_SIM_TIMES):
                    rep_sim = copy.deepcopy(self)
                    rep_sim.sim_end_time = self.t + SINGLE_ROLLOUT_HORIZON
                    rep_sim.apply_decision_single_info(inv_dec=-1, route_dec=cur_station)
                    rep_sim.single_full_list, rep_sim.single_empty_list = [], []
                    v, rep_rental, rep_return = rep_sim.run_replication_single_info(base_policy=None)
                    station_rental = [a + b for a, b in zip(station_rental, rep_rental)]
                    station_return = [a + b for a, b in zip(station_return, rep_return)]
                station_rental = [val / ROLLOUT_SIM_TIMES for val in station_rental]
                station_return = [val / ROLLOUT_SIM_TIMES for val in station_return]
                inv_dec = -1
                pre_rental, pre_return = 0, VEH_CAP
                station_rental, station_return = \
                    [min(val, pre_rental) for val in station_rental], [min(val, pre_return) for val in station_return]
                station_max = [max(a, b) for a, b in zip(station_rental, station_return)]
                default_station = list(self.stations.keys())
                random.shuffle(default_station)
                default_max = [station_max[val - 1] for val in default_station]
                route_dec = default_station[default_max.index(max(default_max))]

        elif self.policy == 'GLA':  # 2023 TS baseline
            cur_station, cur_load = self.veh_info[0], self.veh_info[2]
            if cur_station:  # at station
                cur_inv = self.stations[cur_station].num_self
                # inv decision
                dep = sum(
                    self.mu_s_array[int(self.t / MIN_STEP):int(self.t / MIN_STEP + GLA_HORIZON / MIN_STEP),
                    cur_station - 1])
                arr = sum(
                    self.lambda_s_array[int(self.t / MIN_STEP):int(self.t / MIN_STEP + GLA_HORIZON / MIN_STEP),
                    cur_station - 1])
                net_demand = int(dep - arr) + 1 if dep > arr else int(dep - arr)
                # print(net_demand, cur_inv, (dep, arr))
                if net_demand >= cur_inv:
                    inv_dec = min(net_demand, self.stations[cur_station].cap)
                    load_after_ins = cur_load - min(inv_dec - cur_inv, cur_load)
                else:
                    inv_dec = max(net_demand, 0)
                    load_after_ins = cur_load + min(cur_inv - inv_dec, VEH_CAP - cur_load)
                # route decision
                rate = load_after_ins / VEH_CAP
                stations = [i for i in self.stations.keys() if i != cur_station]
                random.shuffle(stations)
                if rate <= GLA_delta:  # load
                    num_self_list = [self.stations[station].num_self for station in stations]
                    route_dec = stations[num_self_list.index(max(num_self_list))]
                else:
                    net_demand_list = [
                        round(
                            sum(
                                self.mu_s_array[int(self.t / MIN_STEP):int(self.t / MIN_STEP + GLA_HORIZON / MIN_STEP),
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

                # estimate current cost
                if inv_dec > self.stations[cur_station].num_self:
                    ins = min(inv_dec - self.stations[cur_station].num_self, cur_load)
                elif inv_dec < self.stations[cur_station].num_self:
                    ins = max(inv_dec - self.stations[cur_station].num_self, cur_load - VEH_CAP)
                else:
                    ins = 0
                num_self_list = [val.num_self for val in self.stations.values()]
                num_oppo_list = [val.num_opponent for val in self.stations.values()]
                num_self_list[cur_station - 1] += ins
                on_route_t = 5 * (int((self.dist[
                                           cur_station, route_dec] - 0.2) / 5) + 1) if cur_station != route_dec else 0
                cur_step_t = CONST_OPERATION + on_route_t if cur_station != route_dec else MIN_RUN_STEP
                # cost at current step
                order_exp = self.get_estimated_order(
                    step_t=cur_step_t, num_self=num_self_list, num_oppo=num_oppo_list, start_t=self.t
                )

                post_dec_var_dict = self.get_post_decision_var_dict(inv_dec=inv_dec, route_dec=route_dec)
                self.cost_list.append(ORDER_INCOME_UNIT * order_exp - UNIT_TRAVEL_COST * on_route_t)
                self.basis_func_property.append(dict(post_dec_var_dict))


            else:  # at depot
                inv_dec = -1
                stations = [i for i in self.stations.keys() if i != cur_station]
                random.shuffle(stations)
                num_self_list = [self.stations[station].num_self for station in stations]
                route_dec = stations[num_self_list.index(max(num_self_list))]

                num_self_list = [val.num_self for val in self.stations.values()]
                num_oppo_list = [val.num_opponent for val in self.stations.values()]
                cur_step_t = 5 * (int((self.dist[
                                           cur_station, route_dec] - 0.2) / 5) + 1) if cur_station != route_dec else 0
                # cost at current step
                order_exp = self.get_estimated_order(
                    step_t=cur_step_t, num_self=num_self_list, num_oppo=num_oppo_list, start_t=self.t
                )

                post_dec_var_dict = self.get_post_decision_var_dict(inv_dec=inv_dec, route_dec=route_dec)
                self.cost_list.append(ORDER_INCOME_UNIT * order_exp - UNIT_TRAVEL_COST * cur_step_t)
                self.basis_func_property.append(dict(post_dec_var_dict))

        else:
            print('policy type error.')
            assert False

        assert route_dec is not None

        return {'inv': inv_dec, 'route': route_dec}

    def decide_time(self, route_dec: int) -> int:
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

    def apply_decision_multi_info(self, inv_dec: int, route_dec: int):
        """
        改变当前站点库存和车辆载量，时间转移

        :param inv_dec: inventory decision
        :param route_dec: route decision (station id)
        :return:
        """
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
                num_change_list, success_list, success_opponent_list, full_list, empty_list = \
                    self.generate_orders(gene_t=count_t)
                # num_change
                self.apply_num_change(num_change_list)
                # success_record
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

                self.t += operation_duration

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

    def generate_orders(self, gene_t=MIN_RUN_STEP, single=False):
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
                if round((self.t-RE_START_T)/10) < 36:
                    self.test_esd += \
                        sum([esd_arr[s-1, round((self.t-RE_START_T)/10), round((self.t+MIN_RUN_STEP-RE_START_T)/10), self.stations[s].num_self, self.stations[s].num_opponent] for s in range(1, 26)])
                elif round((self.t-RE_START_T)/10) == 36:
                    self.test_esd += \
                        sum([esd_arr[s-1, round((self.t-RE_START_T)/10)-1, -1, self.stations[s].num_self, self.stations[s].num_opponent] for s in range(1, 26)])

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
            self.t += MIN_RUN_STEP
        # before operation
        if self.veh_info[1] is not None:
            move_dist = (int(self.dist[self.veh_info[0], self.veh_info[1]] / MIN_RUN_STEP) + 1) * MIN_RUN_STEP \
                if self.veh_info[0] != self.veh_info[1] else 0
            self.veh_distance += move_dist
            self.veh_info[0] = self.veh_info[1]  # current_loc = next_loc

        if self.t >= RE_END_T and self.return_count_time < 0.8:
            assert self.return_count_time == 0
            return_dist = (int(self.dist[self.veh_info[0], self.veh_info[1]] / MIN_RUN_STEP) + 1) * MIN_RUN_STEP \
                if self.veh_info[0] != self.veh_info[1] else 0
            cur_station, cur_load = self.veh_info[0], self.veh_info[2]
            assert cur_station > 0 if self.policy is not None else True

            # put all the bikes at current station
            if self.policy is not None:
                self.stations[cur_station].num_self += cur_load
                self.veh_info[2] = 0

            self.veh_distance += return_dist
            self.return_count_time += 1
        self._log.append(self.simulation_log_format(self.stations))

    def step_single_info(self, end_t: int):
        """
        步进函数 for single information

        :return:
        """
        end_t += self.t
        while self.t < end_t and self.t < self.sim_end_time:
            num_change_list, full_list, empty_list = self.generate_orders(single=True)
            # num_change
            self.apply_num_change(num_change_list)
            # full_loss_record
            self.single_full_list.append(list(full_list))
            # empty_loss_record
            self.single_empty_list.append(list(empty_list))
            # step forward
            self.t += MIN_RUN_STEP
        # before operation
        if self.veh_info[1] is not None:
            move_dist = (int(self.dist[self.veh_info[0], self.veh_info[1]] / MIN_RUN_STEP) + 1) * MIN_RUN_STEP \
                if self.veh_info[0] != self.veh_info[1] else 0
            self.veh_distance += move_dist
            self.veh_info[0] = self.veh_info[1]  # current_loc = next_loc

    def run(self):
        """
        仿真运行主函数

        :return:
        """
        # change stage_info and simulation log
        self.stage_info.append(self.stage_info_format(0, 0, 0, 0, 0))
        self._log.append(self.simulation_log_format(self.stations))

        # start simulation
        while self.t < self.sim_end_time:

            if self.t:
                self.stage += 1
            self.stage_info.append(
                self.stage_info_format(
                    stage=self.stage, time=self.t, veh_loc=self.veh_info[0],
                    veh_next_loc=self.veh_info[1], veh_load=self.veh_info[2]))

            if RE_START_T <= self.t <= RE_END_T:
                # decisions at current stage
                dec_start = time.process_time()
                assert isinstance(self.single, bool)
                if not self.single:  # multi-info
                    dec_dict = self.decide_action_multi_info()
                else:  # single-info
                    dec_dict = self.decide_action_single_info()
                dec_end = time.process_time()
                inv_dec, route_dec = dec_dict['inv'], dec_dict['route']
                if self.print_action:
                    if self.veh_info[0] > 0:
                        print(
                            f'({int(dec_end - dec_start)}s) Decision done at {int((self.t / 60 * 100)) / 100} ' +
                            f'with inv={inv_dec}(cur_inv={self.stations[self.veh_info[0]].num_self}/{self.stations[self.veh_info[0]].num_opponent}) ' +
                            f'and route={route_dec}(from station={self.veh_info[0]}) and vehicle load={self.veh_info[2]} ' +
                            f'before operation.')
                    else:
                        print(
                            f'({int(dec_end - dec_start)}s) Decision done at {int((self.t / 60 * 100)) / 100} ' +
                            f'and route={route_dec}(from depot) at depot')
                # change next_loc and load in apply_decision
                self.apply_decision_multi_info(inv_dec=inv_dec, route_dec=route_dec)
                # self.veh_info[1] = route_dec
                t_dec = self.decide_time(route_dec=route_dec)  # 向前步进若干步，单位：min
            elif self.t > RE_END_T:
                t_dec = STAY_TIME
                num_self_list = [val.num_self for val in self.stations.values()]
                num_oppo_list = [val.num_opponent for val in self.stations.values()]
                self.cost_after_work += ORDER_INCOME_UNIT * \
                                        self.get_estimated_order(step_t=t_dec, num_self=num_self_list,
                                                                 num_oppo=num_oppo_list, start_t=self.t)
            else:
                t_dec = STAY_TIME

            self.stage_info.append(
                self.stage_info_format(
                    stage=self.stage, time=self.t, veh_loc=self.veh_info[0],
                    veh_next_loc=self.veh_info[1], veh_load=self.veh_info[2]))
            self.step_multi_info(end_t=t_dec)

    def run_replication_multi_info(self, base_info='multi', base_policy=None):
        """
        simulation in simulation, to decide best decision.

        :param base_info: base information considered in the simulation
        :param base_policy: base policy within the simulation
        :return: sum cost
        """
        # no stage_log
        sim = copy.deepcopy(self)
        sim.policy = base_policy
        if base_info == 'multi':
            sim.single = False
        elif base_info == 'single':
            sim.single = True
        # start inner simulation
        while sim.t < sim.sim_end_time:
            # decisions at current stage
            dec_dict = sim.decide_action_multi_info()
            inv_dec, route_dec = dec_dict['inv'], dec_dict['route']
            # change next_loc and load in apply_decision
            sim.apply_decision_multi_info(inv_dec=inv_dec, route_dec=route_dec)
            t_dec = sim.decide_time(route_dec)  # 向前步进若干步，单位：min
            sim.step_multi_info(end_t=t_dec)
        return sim.success

    def run_replication_single_info(self, base_policy=None):
        """
        simulation in simulation, to decide best decision.

        :param base_policy: base policy within the simulation
        :return: rep_fail(number), rep_rental(list), rep_return(list)
        """
        sim = copy.deepcopy(self)
        cur_loc = sim.veh_info[0]
        sim.policy = base_policy
        sim.single = True
        # start inner simulation
        while sim.t < sim.sim_end_time:
            sim.step_single_info(end_t=MIN_RUN_STEP)
        if cur_loc < 0.1:
            rep_fail = 0
        else:
            single_full_array, single_empty_array = np.array(sim.single_full_list), np.array(sim.single_empty_list)
            rep_fail = sum(single_full_array[:, cur_loc - 1]) + sum(single_empty_array[:, cur_loc - 1])
        rep_rental, rep_return = [], []
        for station in sim.stations.keys():
            t = self.dist[cur_loc, station]
            single_empty_array, single_full_array = np.array(sim.single_empty_list), np.array(sim.single_full_list)
            rep_rental.append(sum(single_empty_array[int(t / MIN_STEP):, station - 1]))
            rep_return.append(sum(single_full_array[int(t / MIN_STEP):, station - 1]))
        return rep_fail, rep_rental, rep_return

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
