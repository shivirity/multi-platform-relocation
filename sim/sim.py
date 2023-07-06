import copy
import time
import random
import numpy as np

from sim.consts import *

random.seed(SEED)
np.random.seed(SEED)


class Simulation:
    def __init__(self, stations: dict, dist_array: np.ndarray,
                 mu_s_array: np.ndarray, mu_c_array: np.ndarray, lambda_s_array: np.ndarray,
                 lambda_c_array: np.ndarray):
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
        self.dist = dist_array

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
        self.success_list = []
        self.success_opponent = 0  # number of successful orders from opponent platform
        self.success_opponent_list = []
        self.full_loss = 0  # number of orders that lost because of station is full
        self.full_loss_list = []
        self.empty_loss = 0  # number of orders that lost because of lack of bikes (single/both platforms)
        self.empty_loss_list = []

        # policy
        self.policy = None  # 'None', 'random', 'STR', 'rollout'
        self.single = None  # bool, False means decide with multi-information, True means decide with single-info

        # log
        self._log = []
        self.print_action = False  # print the action of relocation vehicle

        # for single rollout
        self.single_full_list = None  # record return loss for every station in single rollout
        self.single_empty_list = None  # record rental loss for every station in single rollout

    @property
    def log(self):
        return self._log

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

    def decide_action(self):
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
                            rep_sim.apply_decision(inv_dec=inv, route_dec=route)
                            t_dec = rep_sim.decide_time(route)  # 向前步进若干步, in rollout, 单位：min
                            rep_sim.step(end_t=t_dec)
                            rep_success = rep_sim.run_replication(base_policy='multi-STR')
                            rep_sim_h_success.append(rep_success)
                        route_success[(inv, route)] = np.mean(np.array(rep_sim_h_success, dtype=np.single))
                route_dec = max(route_success, key=lambda x: route_success[x])
                inv_dec, route_dec = route_dec[0], route_dec[1]
            else:
                route_success = []
                route_choose = [i for i in self.stations.keys()]
                for route in route_choose:
                    rep_sim = copy.deepcopy(self)
                    rep_sim.apply_decision(-1, route)
                    rep_sim_h_success = []
                    for _ in range(ROLLOUT_SIM_TIMES):
                        t_dec = rep_sim.decide_time(route)  # 向前步进若干步, in rollout, 单位：min
                        rep_sim.step(end_t=t_dec)
                        rep_success = rep_sim.run_replication(base_policy='multi-STR')
                        rep_sim_h_success.append(rep_success)
                    route_success.append(np.mean(np.array(rep_sim_h_success, dtype=np.single)))
                route_dec = route_success.index(max(route_success))
                inv_dec = -1

        # do nothing
        elif self.policy is None:
            cur_station = self.veh_info[0]
            inv_dec, route_dec = -1, cur_station

        # short-term relocation
        # todo 多平台STR修正
        elif self.policy == 'multi-STR':
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

        elif self.policy == 'rollout with limited horizon':
            pass

        else:
            print('policy type error.')
            assert False

        return {'inv': inv_dec, 'route': route_dec}

    def decide_action_single_station(self):
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
                        rep_sim.apply_decision_single_station(inv_dec=inv, route_dec=cur_station)
                        rep_sim.single_full_list, rep_sim.single_empty_list = [], []
                        rep_fail, rep_rental, rep_return = rep_sim.run_replication_single_station(base_policy=None)
                        rep_inv_fail.append(rep_fail)
                        station_rental = [a + b for a, b in zip(station_rental, rep_rental)]
                        station_return = [a + b for a, b in zip(station_return, rep_return)]
                    inv_fail.append(sum(rep_inv_fail))
                station_rental = [val/(len(inv_levels) * ROLLOUT_SIM_TIMES) for val in station_rental]
                station_return = [val/(len(inv_levels) * ROLLOUT_SIM_TIMES) for val in station_return]

                # print(inv_fail)
                inv_dec = inv_levels[inv_fail.index(min(inv_fail))]
                pre_rental = self.stations[cur_station].num_self + self.veh_info[2] - \
                             self.get_station_inv(self.stations[cur_station].num_self, inv_dec, self.veh_info[2])
                pre_return = VEH_CAP - pre_rental
                station_rental, station_return = \
                    [min(val, pre_rental) for val in station_rental], [min(val, pre_return) for val in station_return]
                station_max = [max(a, b) for a, b in zip(station_rental, station_return)]
                station_max[cur_station-1] = -32
                default_station = list(self.stations.keys())
                random.shuffle(default_station)
                default_max = [station_max[val-1] for val in default_station]
                route_dec = default_station[default_max.index(max(default_max))]
            else:
                station_rental, station_return = \
                    [0 for _ in range(len(self.stations.keys()))], [0 for _ in range(len(self.stations.keys()))]
                for _ in range(ROLLOUT_SIM_TIMES):
                    rep_sim = copy.deepcopy(self)
                    rep_sim.sim_end_time = self.t + SINGLE_ROLLOUT_HORIZON
                    rep_sim.apply_decision_single_station(inv_dec=-1, route_dec=cur_station)
                    rep_sim.single_full_list, rep_sim.single_empty_list = [], []
                    v, rep_rental, rep_return = rep_sim.run_replication_single_station(base_policy=None)
                    station_rental = [a + b for a, b in zip(station_rental, rep_rental)]
                    station_return = [a + b for a, b in zip(station_return, rep_return)]
                station_rental = [val/ROLLOUT_SIM_TIMES for val in station_rental]
                station_return = [val/ROLLOUT_SIM_TIMES for val in station_return]
                inv_dec = -1
                pre_rental, pre_return = 0, VEH_CAP
                station_rental, station_return = \
                    [min(val, pre_rental) for val in station_rental], [min(val, pre_return) for val in station_return]
                station_max = [max(a, b) for a, b in zip(station_rental, station_return)]
                default_station = list(self.stations.keys())
                random.shuffle(default_station)
                default_max = [station_max[val-1] for val in default_station]
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
                net_demand = round(dep - arr)
                # print((dep, arr))
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
            else:  # at depot
                inv_dec = -1
                stations = [i for i in self.stations.keys() if i != cur_station]
                random.shuffle(stations)
                num_self_list = [self.stations[station].num_self for station in stations]
                route_dec = stations[num_self_list.index(max(num_self_list))]

        else:
            print('policy type error.')
            assert False

        return {'inv': inv_dec, 'route': route_dec}

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

    def apply_decision(self, inv_dec: int, route_dec: int):
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
                num_change_list, success_list, success_opponent_list, full_list, empty_list = \
                    self.generate_orders(gene_t=count_t)
                # num_change
                self.apply_num_change(num_change_list)
                # success_record
                sum_success = sum(success_list)
                self.success += sum_success
                self.success_list.append(sum_success)
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

    def apply_decision_single_station(self, inv_dec: int, route_dec: int):
        """
        改变当前站点库存和车辆载量，用于single information时的流转量估计

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
                    (num_s - dep_s - self.stations[station].num_self, num_c - dep_c - self.stations[station].num_opponent))
            return \
                list(num_change_list), list(success_list), list(success_opponent_list), list(full_list), list(empty_list)

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

    def step(self, end_t: int):
        """
        步进函数

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
        # before operation
        if self.veh_info[1] is not None:
            self.veh_info[0] = self.veh_info[1]  # current_loc = next_loc
        self._log.append(self.simulation_log_format(self.stations))

    def step_single_station(self, end_t: int):
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
            self.t += MIN_STEP
        # before operation
        if self.veh_info[1] is not None:
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
                    dec_dict = self.decide_action()
                else:  # single-info
                    dec_dict = self.decide_action_single_station()
                dec_end = time.process_time()
                inv_dec, route_dec = dec_dict['inv'], dec_dict['route']
                if self.print_action:
                    print(
                        f'({int(dec_end - dec_start)}s) Decision done at {self.t} with inv={inv_dec} and route={route_dec} and vehicle load={self.veh_info[2]} before operation.')
                # change next_loc and load in apply_decision
                self.apply_decision(inv_dec=inv_dec, route_dec=route_dec)
                # self.veh_info[1] = route_dec
                t_dec = self.decide_time(route_dec)  # 向前步进若干步，单位：min
            else:
                t_dec = STAY_TIME

            self.stage_info.append(
                self.stage_info_format(
                    stage=self.stage, time=self.t, veh_loc=self.veh_info[0],
                    veh_next_loc=self.veh_info[1], veh_load=self.veh_info[2]))
            self.step(end_t=t_dec)

    def run_replication(self, base_info='multi', base_policy=None):
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
            dec_dict = sim.decide_action()
            inv_dec, route_dec = dec_dict['inv'], dec_dict['route']
            # change next_loc and load in apply_decision
            sim.apply_decision(inv_dec=inv_dec, route_dec=route_dec)
            t_dec = sim.decide_time(route_dec)  # 向前步进若干步，单位：min
            sim.step(end_t=t_dec)
        return sim.success

    def run_replication_single_station(self, base_policy=None):
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
            sim.step_single_station(end_t=MIN_STEP)
        if cur_loc < 0.1:
            rep_fail = 0
        else:
            single_full_array, single_empty_array = np.array(sim.single_full_list), np.array(sim.single_empty_list)
            rep_fail = sum(single_full_array[:, cur_loc - 1]) + sum(single_empty_array[:, cur_loc - 1])
        rep_rental, rep_return = [], []
        for station in sim.stations.keys():
            t = self.dist[cur_loc, station]
            single_empty_array, single_full_array = np.array(sim.single_empty_list), np.array(sim.single_full_list)
            rep_rental.append(sum(single_empty_array[int(t/MIN_STEP):, station - 1]))
            rep_return.append(sum(single_full_array[int(t/MIN_STEP):, station - 1]))
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
