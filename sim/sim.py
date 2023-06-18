import copy
import time
import random
import numpy as np

from sim.consts import *

random.seed(SEED)
np.random.seed(SEED)


class Simulation:
    def __init__(self, stations: dict, dist_array: np.ndarray,
                 mu_s_array: np.ndarray, mu_c_array: np.ndarray, lambda_s_array: np.ndarray, lambda_c_array: np.ndarray):
        """
        Simulation类.

        :param stations: dict of Station object
        :param dist_array: distance matrix
        :param mu_array: demand(departing) rate for every time_idx and station todo
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
        self.full_loss = 0  # number of orders that lost because of full station
        self.full_loss_list = []

        # policy
        self.policy = None  # 'None', 'random', 'STR', 'rollout'
        self.single = False  # bool, True means decide with multi-information, False means decide with single-info

        # log
        self._log = []
        self.print_action = False  # print the action of relocation vehicle

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
            return load + ins

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
                            rep_success = rep_sim.run_replication(base_policy='STR')
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
                        rep_success = rep_sim.run_replication(base_policy='STR')
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
        elif self.policy == 'STR':
            cur_station, cur_load = self.veh_info[0], self.veh_info[2]
            if cur_station:
                cur_inv = self.stations[cur_station].num_self
                # shortage
                if cur_inv < round(GAMMA * self.stations[cur_station].cap):
                    inv_dec = min(round(GAMMA * self.stations[cur_station].cap), cur_inv + cur_load)
                    load_after_ins = cur_load - (inv_dec - cur_inv)
                # surplus
                elif cur_inv > round((1-GAMMA) * self.stations[cur_station].cap):
                    inv_dec = max(round((1-GAMMA) * self.stations[cur_station].cap), cur_inv - (VEH_CAP - cur_load))
                    load_after_ins = cur_load + cur_inv - inv_dec
                # balanced
                else:
                    inv_dec = -1
                    load_after_ins = cur_load
                pot_stations = [i for i in self.stations.keys() if i != cur_station]
                if 0 < load_after_ins < VEH_CAP:
                    imb_stations = [i for i in pot_stations
                                    if self.stations[i].num_self > (1-GAMMA)*self.stations[i].cap or self.stations[i].num_self < GAMMA*self.stations[i].cap]
                elif load_after_ins == 0:
                    imb_stations = [i for i in pot_stations if
                                    self.stations[i].num_self > (1-GAMMA)*self.stations[i].cap]
                elif load_after_ins == VEH_CAP:
                    imb_stations = [i for i in pot_stations if
                                    self.stations[i].num_self < GAMMA * self.stations[i].cap]
                else:
                    imb_stations = []
                # 有可以前往的站点
                if imb_stations:
                    dis_list = [self.dist[cur_station, i] for i in imb_stations]
                    route_dec_idx = random.sample([i for i in range(len(imb_stations)) if dis_list[i] == min(dis_list)], 1)[0]
                    route_dec = imb_stations[route_dec_idx]
                # 没有可以前往的站点
                else:
                    route_dec = cur_station
            else:
                inv_dec = -1
                pot_stations = [i for i in self.stations.keys()]
                surplus_stations = [i for i in pot_stations if self.stations[i].num_self > (1-GAMMA)*self.stations[i].cap]
                # 有可以前往的站点
                if surplus_stations:
                    dis_list = [self.dist[0, i] for i in surplus_stations]
                    route_dec_idx = random.sample([i for i in range(len(surplus_stations)) if dis_list[i] == min(dis_list)], 1)[0]
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

        return {'inv': inv_dec, 'route': route_dec}
        # todo


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
                self.veh_info[1], self.veh_info[2] = route_dec, self.veh_info[2]-ins
                # time
                operation_duration = OPERATION_TIME * abs(ins)
                # 操作时流转数量
                count_t = min(self.sim_end_time - self.t, operation_duration)
                num_change_list, success_list, success_opponent_list, full_list = \
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

                self.t += operation_duration

    def generate_orders(self, gene_t=MIN_STEP):
        """
        生成 time min 内订单

        :param gene_t: 生成xx分钟内的订单
        :return: list(num_change_list), list(success_list), list(success_opponent_list), list(full_list)
        """
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

    def apply_num_change(self, num_change_list):
        for station in self.stations.keys():
            self.stations[station].change_num(num_change_list[station-1])

    def step(self, end_t: int):
        """
        步进函数

        :return:
        """
        end_t += self.t
        while self.t < end_t and self.t < self.sim_end_time:
            # simulation log for current time
            self._log.append(self.simulation_log_format(self.stations))
            num_change_list, success_list, success_opponent_list, full_list = self.generate_orders()
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
            # decisions at current stage
            dec_start = time.process_time()
            assert isinstance(self.single, bool)
            if not self.single:  # multi-info
                dec_dict = self.decide_action()
            else:  # single-info
                dec_dict = self.decide_action_single_station()
            dec_end = time.process_time()
            inv_dec, route_dec = dec_dict['inv'], dec_dict['route']
            print(f'({int(dec_end-dec_start)}s) Decision done at {self.t} with inv={inv_dec} and route={route_dec} and vehicle load={self.veh_info[2]} before operation.')
            # change next_loc and load in apply_decision
            self.apply_decision(inv_dec=inv_dec, route_dec=route_dec)
            # self.veh_info[1] = route_dec
            t_dec = self.decide_time(route_dec)  # 向前步进若干步，单位：min

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
