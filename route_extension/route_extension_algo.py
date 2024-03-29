import heapq
import logging
import numpy as np
from typing import Tuple, Union

from simulation.consts import (DELTA_CAP, L_REA, U_REA, BETA_L, BETA_U, RE_START_T, RE_END_T, VEH_CAP, CAP_S,
                               NUM_INIT_ROUTES, NUM_DUAL_ROUTES, BETA_L_DUAL, BETA_U_DUAL, ALPHA, ORDER_INCOME_UNIT)


# data structure for route extension tree
class Node:
    # initialize the node
    def __init__(self,
                 stations: Union[int, Tuple[int, int]],
                 ins: Union[int, Tuple[int, int]],
                 profit: float, duration: int, load_after_ins=None):
        """stations contain one or two stations,
        negative ins means load from station, positive ins means unload to station"""
        self.stations = stations
        self.ins = ins  # instructions at stations
        self.children = []
        self.parent = None
        self.route = None  # insert after the parent's route
        self.route_ins = None  # insert after the parent's instructions
        self.profit = profit  # profit of the route
        self.cum_profit = None  # cumulative profit of the route
        self.duration = duration
        self.total_time = None  # total time cost
        self.load_after_ins = load_after_ins  # load after the instructions

    def add_child(self, child):
        """child is a Node object"""
        child.parent = self
        self.children.append(child)
        # route extension
        if isinstance(child.stations, tuple):
            child.route = self.route + list(child.stations)
            child.route_ins = self.route_ins + list(child.ins)
        else:
            child.route = self.route + [child.stations]
            child.route_ins = self.route_ins + [child.ins]
        # calculate profit
        child.cum_profit = self.cum_profit + child.profit
        # calculate duration
        child.total_time = self.total_time + child.duration
        # calculate load
        child.load_after_ins = self.load_after_ins - child.ins


class ESDComputer:
    re_start_t = round(RE_START_T / 10)  # relocation开始时间 (in 10 min)
    re_end_t = round(RE_END_T / 10)  # relocation结束时间 (in 10 min)

    def __init__(self, esd_arr: np.ndarray, ei_s_arr: np.ndarray, ei_c_arr: np.ndarray,
                 t_cur: int, t_fore: int, c_mat: np.ndarray):
        """
        ESDComputer class to compute the expected satisfied demand (ESD) of the system in the forecast horizon

        :param esd_arr: array of Expected Satisfied Demand
        :param ei_s_arr: array of Expected Inventory (self)
        :param ei_c_arr: array of Expected Inventory (competitor)
        :param t_cur: current time
        :param t_fore: forecast horizon
        :param c_mat: distance matrix
        """
        self.esd_arr = esd_arr
        self.ei_s_arr = ei_s_arr
        self.ei_c_arr = ei_c_arr
        self.t_cur = round(t_cur - RE_START_T / 10)
        self.t_fore = t_fore
        self.c_mat = c_mat

    def compute_ESD_in_horizon(self, station_id: int, t_arr: int, ins: int, x_s_arr: list,
                               x_c_arr: list, mode: str, delta: bool, repo: bool = True) -> float:
        """
        compute ESD for station station_id in the forecasting horizon
        :param station_id: station id (starting from 1)
        :param t_arr: time point of arrival/operation
        :param ins: instruct quantity (positive-unload, negative-load)
        :param x_s_arr: array of bike numbers (self)
        :param x_c_arr: array of bike numbers (competitor)
        :param mode: estimation mode ('single' or 'multi')
        :param delta: boolean value to indicate whether to use delta ESD
        :param repo: boolean value to indicate whether repositioning is allowed
        :return: ESD value in forecasting horizon (from current time to end time)
        """
        # ei_s_arr[s, t_0, t_1, x_s, x_c]
        # ei_c_arr[s, t_0, t_1, x_s, x_c]
        # esd_arr[s, t_0, t_1, x_s, x_c]
        if mode == 'multi':
            if repo is True:
                before_val = self.esd_arr[
                    station_id - 1,
                    self.t_cur,
                    round(self.t_cur + t_arr) if round(self.t_cur + t_arr) < 49 else 48,
                    x_s_arr[station_id - 1],
                    x_c_arr[station_id - 1]]
                after_val = self.esd_arr[
                    station_id - 1,
                    round(self.t_cur + t_arr) if round(self.t_cur + t_arr) < 36 else 35,
                    round(self.t_cur + self.t_fore) if (self.t_cur + self.t_fore) < 49 else 48,
                    round(self.ei_s_arr[
                              station_id - 1,
                              self.t_cur,
                              (round(self.t_cur + t_arr)) if (round(self.t_cur + t_arr)) < 49 else 48,
                              x_s_arr[station_id - 1],
                              x_c_arr[station_id - 1]] + ins),
                    round(self.ei_c_arr[
                              station_id - 1,
                              self.t_cur,
                              round(self.t_cur + t_arr) if (round(self.t_cur + t_arr)) < 49 else 48,
                              x_s_arr[station_id - 1],
                              x_c_arr[station_id - 1]])
                ]
                if delta is False:
                    return before_val + after_val
                else:
                    original_val = self.esd_arr[
                        station_id - 1,
                        self.t_cur,
                        round(self.t_cur + self.t_fore) if (self.t_cur + self.t_fore) < 49 else 48,
                        x_s_arr[station_id - 1],
                        x_c_arr[station_id - 1]
                    ]
                    return before_val + after_val - original_val
            else:
                original_val = self.esd_arr[
                    station_id - 1,
                    self.t_cur,
                    round(self.t_cur + self.t_fore) if (self.t_cur + self.t_fore) < 49 else 48,
                    x_s_arr[station_id - 1],
                    x_c_arr[station_id - 1]
                ]
                return original_val
        else:
            raise ValueError('mode should be multi')

    def compute_route(self, r: list, t_left: int, init_l: int, x_s_arr: list, x_c_arr: list,
                      t_repo: int = 0, can_stay: bool = False):
        """
        calculate the cost of the route and the instructions using dynamic programming

        :param r: the given route (in list)
        :param t_left: time left to get to the start location
        :param init_l: initial load on the van
        :param x_s_arr: array of bike numbers (self)
        :param x_c_arr: array of bike numbers (competitor)
        :param t_repo: repositioning window length (in 10 min)
        :param can_stay: whether the van can stay at the station
        :return:
        """
        cap_van, cap_station = VEH_CAP, CAP_S
        minus_M = -1000  # unvisited
        false_flag = -10000  # infeasible
        route = list(r)
        station_num = len(route)
        level_num = round(cap_van + 1)  # number of levels of load on van
        reward_arr, trace_arr = (np.full((level_num, station_num), minus_M, dtype=float),
                                 np.full((level_num, station_num), minus_M))
        if can_stay is False:
            if route[0] == 0:  # starting from depot, only load bikes
                assert t_left == 0 and init_l == 0
                reward_arr[0, 0] = 0
                reward_arr[1:, 0] = false_flag
                trace_arr[0, 0] = 0
                trace_arr[1:, 0] = false_flag
            else:  # load or unload bikes
                for j in range(level_num):
                    ins = init_l - j  # 正代表在站点放下车辆，负代表在站点提取车辆
                    # x_s_0, x_c_0 = self.x_s_arr[route[0] - 1], self.x_c_arr[route[0] - 1]
                    x_s_0, x_c_0 = self.ei_s_arr[route[0] - 1, self.t_cur, self.t_cur + t_left, x_s_arr[route[0] - 1],
                    x_c_arr[route[0] - 1]], \
                        self.ei_c_arr[route[0] - 1, self.t_cur, self.t_cur + t_left, x_s_arr[route[0] - 1],
                        x_c_arr[route[0] - 1]]
                    if 0 <= round(x_s_0 + ins) <= cap_station:
                        reward_arr[j, 0] = ORDER_INCOME_UNIT * self.compute_ESD_in_horizon(station_id=route[0],
                                                                                           t_arr=t_left, ins=ins,
                                                                                           x_s_arr=x_s_arr,
                                                                                           x_c_arr=x_c_arr,
                                                                                           mode='multi',
                                                                                           delta=False)
                        trace_arr[j, 0] = init_l
                    else:
                        reward_arr[j, 0] = false_flag
                        trace_arr[j, 0] = false_flag

            t_trip, t_spend_on_route = t_left, t_left
            for i in range(1, station_num):
                # logging.warning(f'{t_trip}, {self.c_mat[route[i - 1], route[i]]}')
                t_trip += self.c_mat[route[i - 1], route[i]]  # plus travel time
                if self.re_start_t + self.t_cur + t_trip <= self.re_end_t:
                    # t_spend_on_route += self.c_mat[route[i - 1], route[i]] - 1  # minus operation time
                    for k in range(level_num):
                        for former_k in range(level_num):
                            if reward_arr[former_k, i - 1] == false_flag:  # infeasible
                                pass
                            else:  # feasible
                                ins = former_k - k
                                if 0 <= round(self.ei_s_arr[
                                                  round(route[i] - 1),
                                                  round(self.t_cur),
                                                  round(self.t_cur + t_trip),
                                                  round(x_s_arr[route[i] - 1]),
                                                  round(x_c_arr[route[i] - 1])]) + ins <= cap_station:
                                    station_esd = ORDER_INCOME_UNIT * self.compute_ESD_in_horizon(station_id=route[i],
                                                                                                  t_arr=t_trip,
                                                                                                  ins=ins,
                                                                                                  x_s_arr=x_s_arr,
                                                                                                  x_c_arr=x_c_arr,
                                                                                                  mode='multi',
                                                                                                  delta=False)
                                    if station_esd + reward_arr[former_k, i - 1] > reward_arr[k, i]:
                                        reward_arr[k, i] = station_esd + reward_arr[former_k, i - 1]
                                        trace_arr[k, i] = former_k
                        else:
                            if reward_arr[k, i] == minus_M:  # unable to reach this state
                                reward_arr[k, i] = false_flag
                                trace_arr[k, i] = false_flag
                else:
                    for k in range(level_num):
                        former_k = k
                        if reward_arr[former_k, i - 1] == false_flag:  # infeasible
                            reward_arr[k, i] = false_flag
                            trace_arr[k, i] = false_flag
                        else:  # feasible
                            station_esd = ORDER_INCOME_UNIT * self.compute_ESD_in_horizon(station_id=route[i], t_arr=0,
                                                                                          ins=0,
                                                                                          x_s_arr=x_s_arr,
                                                                                          x_c_arr=x_c_arr, mode='multi',
                                                                                          delta=False)
                            if station_esd + reward_arr[former_k, i - 1] > reward_arr[k, i]:
                                reward_arr[k, i] = station_esd + reward_arr[former_k, i - 1]
                                trace_arr[k, i] = former_k
                            else:  # unable to reach
                                reward_arr[k, i] = false_flag
                                trace_arr[k, i] = false_flag

            if max(reward_arr[:, -1]) == false_flag:
                cost = -1
                instruct = [None for _ in range(len(route))]
            else:
                profit_ind = np.argmax(reward_arr, axis=0)[-1]
                trace_init = trace_arr[profit_ind, -1]
                profit = reward_arr[profit_ind, -1]

                # trace path
                trace_list, trace = [profit_ind, trace_init], trace_init
                for i in range(station_num - 2, -1, -1):
                    # if trace < -1000:
                    #     logging.warning('here')
                    trace = trace_arr[int(trace), i]
                    trace_list.append(trace)
                assert len(trace_list) == station_num + 1
                trace_list = list(reversed(trace_list))
                instruct = [(trace_list[k] - trace_list[k + 1]) for k in range(len(trace_list) - 1)]
                # cost = profit - self.alpha * t_spend_on_route
                visited_stations = [i for i in r if i != 0]
                cost_sum = 0
                for s in visited_stations:
                    cost_sum += ORDER_INCOME_UNIT * self.esd_arr[
                        s - 1,
                        self.t_cur,
                        round(self.t_cur + self.t_fore) if (self.t_cur + self.t_fore) < 49 else 48,
                        x_s_arr[s - 1],
                        x_c_arr[s - 1]
                    ]
                cost = profit - cost_sum  # the cost represents the delta value

        else:  # can stay at the stations todo: need to adjust then t_left is not zero, 未作可以在depot停留的修改
            cap_van, cap_station = VEH_CAP, CAP_S
            route = list(r)
            num_level = round(cap_van + 1)  # number of levels of load on van
            to_visit_stations = [val for val in route if val != 0]
            num_stations = len(to_visit_stations)
            init_loc = r[0]
            reward_arr = [[[None for _ in range(num_level)] for __ in range(num_stations)] for ___ in range(t_repo + 1)]
            trace_arr = [[[None for _ in range(num_level)] for __ in range(num_stations)] for ___ in range(t_repo + 1)]
            calcu_arr = [[False for _ in range(num_stations)] for __ in range(t_repo + 1)]
            for t in range(t_repo):
                if t == 0:
                    if init_loc == 0:
                        assert t_left == 0 and init_l == 0
                        if to_visit_stations:
                            next_s = to_visit_stations[0]
                            arr_t = round(self.c_mat[init_loc, next_s])
                            calcu_arr[t + arr_t][0] = True
                            for j in range(num_level):
                                ins = 0 - j
                                if 0 <= self.ei_s_arr[next_s - 1,
                                self.t_cur,
                                self.t_cur + arr_t,
                                x_s_arr[next_s - 1],
                                x_c_arr[next_s - 1]] + ins <= cap_station:
                                    reward_arr[t + arr_t][0][j] = ORDER_INCOME_UNIT * self.compute_ESD_in_horizon(
                                        station_id=next_s,
                                        t_arr=arr_t, ins=ins,
                                        x_s_arr=x_s_arr,
                                        x_c_arr=x_c_arr,
                                        mode='multi',
                                        delta=True) - ALPHA * arr_t
                                    trace_arr[t + arr_t][0][j] = (0, -16, init_l)
                                else:
                                    pass
                        else:
                            assert False, "don't reposition at any stations, starting from the depot"
                    else:  # init_loc != 0
                        for inv in range(num_level):  # label every inventory level at initial point
                            ins = init_l - inv
                            if 0 <= x_s_arr[init_loc - 1] + ins <= cap_station:
                                reward_arr[t][0][inv] = ORDER_INCOME_UNIT * self.compute_ESD_in_horizon(
                                    station_id=init_loc, t_arr=t_left, ins=ins, x_s_arr=x_s_arr, x_c_arr=x_c_arr,
                                    mode='multi', delta=True
                                )
                                cur_reward = reward_arr[t][0][inv]
                                # trace to time step 0
                                if len(to_visit_stations) == 1:
                                    to_visit_next = list(to_visit_stations)
                                else:
                                    to_visit_next = to_visit_stations[:2]
                                for ne in to_visit_next:
                                    if ne == init_loc:  # stay at init location
                                        stay_t = 1
                                        if t + stay_t <= t_repo:
                                            ne_idx = to_visit_stations.index(ne)
                                            calcu_arr[t + stay_t][ne_idx] = True
                                            if reward_arr[t + stay_t][ne_idx][inv] is None:
                                                reward_arr[t + stay_t][ne_idx][inv] = cur_reward
                                                trace_arr[t + stay_t][ne_idx][inv] = (t, 0, inv)
                                            else:  # update with higher obj value
                                                new_record = cur_reward
                                                if new_record > reward_arr[t + stay_t][ne_idx][inv]:
                                                    reward_arr[t + stay_t][ne_idx][inv] = new_record
                                                    trace_arr[t + stay_t][ne_idx][inv] = (t, 0, inv)
                                                else:
                                                    pass
                                        else:
                                            pass
                                    else:  # travel to next station
                                        arr_t = round(self.c_mat[init_loc, ne])
                                        if t + arr_t <= t_repo:
                                            ne_idx = to_visit_stations.index(ne)
                                            calcu_arr[t + arr_t][ne_idx] = True
                                            for ne_inv in range(num_level):
                                                ins = inv - ne_inv
                                                if 0 <= self.ei_s_arr[ne - 1, self.t_cur, self.t_cur + arr_t,
                                                x_s_arr[ne - 1], x_c_arr[ne - 1]] + ins <= cap_station:
                                                    new_reward = cur_reward + ORDER_INCOME_UNIT * self.compute_ESD_in_horizon(
                                                        station_id=ne, t_arr=arr_t, ins=ins, x_s_arr=x_s_arr,
                                                        x_c_arr=x_c_arr, mode='multi', delta=True) - ALPHA * (arr_t - 1)
                                                    if reward_arr[t + arr_t][ne_idx][ne_inv] is None:
                                                        reward_arr[t + arr_t][ne_idx][ne_inv] = new_reward
                                                        trace_arr[t + arr_t][ne_idx][ne_inv] = (t, 0, inv)
                                                    else:  # update with higher obj value
                                                        if new_reward > reward_arr[t + arr_t][ne_idx][ne_inv]:
                                                            reward_arr[t + arr_t][ne_idx][ne_inv] = new_reward
                                                            trace_arr[t + arr_t][ne_idx][ne_inv] = (t, 0, inv)
                                                        else:
                                                            pass
                                                else:
                                                    pass
                                        else:
                                            pass
                            else:
                                pass

                else:  # t > 0
                    for cur_s in range(num_stations):
                        if calcu_arr[t][cur_s] is False:
                            pass
                        else:
                            for inv in range(num_level):
                                if reward_arr[t][cur_s][inv] is None:
                                    pass
                                else:
                                    cur_reward = reward_arr[t][cur_s][inv]
                                    if cur_s == num_stations - 1:
                                        to_visit_next = [to_visit_stations[cur_s]]
                                    else:
                                        to_visit_next = to_visit_stations[cur_s: cur_s + 2]
                                    for ne in to_visit_next:
                                        if t == 1 and cur_s == 0:
                                            logging.debug('here')
                                        if ne == to_visit_stations[cur_s]:  # stay at current location
                                            stay_t = 1
                                            if t + stay_t <= t_repo:
                                                calcu_arr[t + stay_t][cur_s] = True
                                                if reward_arr[t + stay_t][cur_s][inv] is None:
                                                    reward_arr[t + stay_t][cur_s][inv] = cur_reward
                                                    trace_arr[t + stay_t][cur_s][inv] = (t, cur_s, inv)
                                                else:
                                                    new_record = cur_reward
                                                    # update with higher value
                                                    if new_record > reward_arr[t + stay_t][cur_s][inv]:
                                                        reward_arr[t + stay_t][cur_s][inv] = new_record
                                                        trace_arr[t + stay_t][cur_s][inv] = (t, cur_s, inv)
                                                    else:
                                                        pass
                                            else:
                                                pass
                                        else:  # travel to next station
                                            arr_t = round(self.c_mat[to_visit_stations[cur_s], ne])
                                            if t + arr_t <= t_repo:
                                                ne_idx = to_visit_stations.index(ne)
                                                calcu_arr[t + arr_t][ne_idx] = True
                                                assert ne_idx == cur_s + 1
                                                for next_inv in range(num_level):
                                                    ins = inv - next_inv
                                                    if 0 <= self.ei_s_arr[ne - 1, self.t_cur, self.t_cur + t + arr_t,
                                                    x_s_arr[ne - 1], x_c_arr[ne - 1]] + ins <= cap_station:
                                                        new_reward = cur_reward + ORDER_INCOME_UNIT * self.compute_ESD_in_horizon(
                                                            station_id=ne, t_arr=t + arr_t, ins=ins, x_s_arr=x_s_arr,
                                                            x_c_arr=x_c_arr, mode='multi', delta=True) - ALPHA * (arr_t - 1)
                                                        if reward_arr[t + arr_t][ne_idx][next_inv] is None:
                                                            reward_arr[t + arr_t][ne_idx][next_inv] = new_reward
                                                            trace_arr[t + arr_t][ne_idx][next_inv] = (t, cur_s, inv)
                                                        else:
                                                            if new_reward > reward_arr[t + arr_t][ne_idx][next_inv]:
                                                                reward_arr[t + arr_t][ne_idx][next_inv] = new_reward
                                                                trace_arr[t + arr_t][ne_idx][next_inv] = (t, cur_s, inv)
                                                            else:
                                                                pass
                                                    else:
                                                        pass
                                            else:
                                                pass

            max_reward_list, max_label_list = [], []
            for s in range(num_stations):
                for inv in range(num_level):
                    # print(t_repo, s, inv)
                    if reward_arr[t_repo][s][inv] is not None:
                        max_reward_list.append(reward_arr[t_repo][s][inv])
                        max_label_list.append((t_repo, s, inv))
            max_reward = max(max_reward_list)
            max_label = max_label_list[max_reward_list.index(max_reward)]
            k_t_repo, k_s, k_inv = max_label
            loc_list, inv_list = [-1 for _ in range(t_repo + 1)], [-1 for _ in range(t_repo + 1)]
            while True:
                if k_t_repo == 0:
                    assert False, f'repositioning window length is {t_repo}'
                else:
                    loc_list[k_t_repo] = to_visit_stations[k_s]
                    inv_list[k_t_repo] = k_inv
                    k_t_repo, k_s, k_inv = trace_arr[k_t_repo][k_s][k_inv]
                    if k_t_repo == 0:
                        loc_list[k_t_repo] = to_visit_stations[k_s] if k_s != -16 else 0
                        inv_list[k_t_repo] = k_inv
                        break
            print(f'max reward: {max_reward}')
            print(loc_list)
            print(inv_list)
            return max_reward, loc_list, inv_list

        return cost, instruct

    def get_on_route_time(self, route: list) -> int:
        """
        路上经历时间 (in 10min)

        :param route: 路径列表
        :return:
        """
        t_trip = 0
        for i in range(1, len(route)):
            t_trip += (
                self.c_mat[route[i - 1], route[i]] - 1 if route[i - 1] != 0 else self.c_mat[route[i - 1], route[i]])
        return round(t_trip)


def get_REA_routes_test(num_of_van: int, van_location: list, van_dis_left: list, van_load: list,
                        c_s: int, c_v: int, cur_t: int, t_p: int, t_f: int, t_roll: int,
                        c_mat: np.ndarray, ei_s_arr: np.ndarray, ei_c_arr: np.ndarray,
                        esd_arr: np.ndarray, x_s_arr: list, x_c_arr: list,
                        alpha: float, est_ins: int, dual_van_vector: list = None, dual_station_vector: list = None,
                        branch: int = 2, state: str = 'init'):
    """
    测试 REA 算法效果 (without dual information)

    :return:
    """
    # todo: 多车时补充不可行站点 list
    num_of_stations = c_mat.shape[0] - 1  # exclude the depot
    reg_cur_t = round(cur_t - RE_START_T / 10)
    for van in range(num_of_van):
        root_loc = van_location[van]
        root_dis_left = van_dis_left[van]
        root_load = van_load[van]
        esd_computer = ESDComputer(
            esd_arr=esd_arr, ei_s_arr=ei_s_arr, ei_c_arr=ei_c_arr, t_cur=cur_t, t_fore=t_f, c_mat=c_mat)
        undone_nodes, done_nodes = [], []  # only destination node in done_nodes

        if root_loc == 0:
            root_profit = 0
            planned_ins = 0
        else:
            if est_ins >= 0:  # to load bikes to the station
                planned_ins = min(est_ins, root_load, c_s - x_s_arr[root_loc - 1])
            else:
                planned_ins = -min(-est_ins, c_v - root_load, x_s_arr[root_loc - 1])
            assert (0 <= planned_ins +
                    round(ei_s_arr[
                              root_loc - 1,
                              reg_cur_t,
                              round(reg_cur_t + root_dis_left),
                              x_s_arr[root_loc - 1],
                              x_c_arr[root_loc - 1]]) <= c_s)
            root_profit = esd_computer.compute_ESD_in_horizon(
                station_id=root_loc, t_arr=root_dis_left, ins=planned_ins,
                x_s_arr=x_s_arr, x_c_arr=x_c_arr, mode='multi', delta=True)
        # create root node
        root = Node(stations=root_loc, ins=planned_ins, profit=root_profit,
                    duration=root_dis_left, load_after_ins=root_load - planned_ins)
        root.route, root.total_time, root.route_ins, root.cum_profit = (
            [root_loc], root_dis_left, [planned_ins], root_profit)
        # create undone nodes list
        undone_nodes.append(root)

        while undone_nodes:
            cur_node = undone_nodes.pop(0)
            node_station = cur_node.stations if isinstance(cur_node.stations, int) else cur_node.stations[1]
            node_t = cur_node.total_time
            est_load = cur_node.load_after_ins
            if est_load / c_v < DELTA_CAP:  # relatively few bikes

                if node_station == 13 and node_t == 4 and cur_node.cum_profit > 11.73:
                    logging.debug('here')

                ant_stations = [i for i in range(1, num_of_stations + 1) if i not in cur_node.route]
                ant_loading = [0 for _ in range(num_of_stations)]
                exp_inv = [
                    round(ei_s_arr[i,
                    reg_cur_t if reg_cur_t < 36 else 35,
                    round(reg_cur_t + node_t + c_mat[node_station, i + 1])
                    if (reg_cur_t + node_t + c_mat[node_station, i + 1]) < 49 else 48,
                    x_s_arr[i],
                    x_c_arr[i]])
                    for i in range(num_of_stations)]
                for j in ant_stations:
                    ant_loading[j - 1] = max(exp_inv[j - 1] - L_REA, 0)
                max_loading = max(ant_loading)
                if state == 'init':
                    beta_l = BETA_L
                elif state == 'dual':
                    beta_l = BETA_L_DUAL
                else:
                    raise ValueError('state should be init or dual')
                ant_next = [j for j in ant_stations if ant_loading[j - 1] >= max_loading * beta_l]
                # finish set selection
                ant_inv = [exp_inv[j - 1] if ant_loading[j - 1] == 0 else max(L_REA, exp_inv[j - 1] - c_v + est_load)
                           for j in ant_next]
                ant_ins = [ant_inv[j] - exp_inv[ant_next[j] - 1] for j in range(len(ant_next))]
                if dual_station_vector is None:
                    ant_profit = [
                        esd_computer.compute_ESD_in_horizon(
                            station_id=ant_next[j], t_arr=node_t + c_mat[node_station, ant_next[j]], ins=ant_ins[j],
                            x_s_arr=x_s_arr, x_c_arr=x_c_arr, mode='multi', delta=True) for j in range(len(ant_next))]
                else:
                    ant_profit = [
                        esd_computer.compute_ESD_in_horizon(
                            station_id=ant_next[j], t_arr=node_t + c_mat[node_station, ant_next[j]], ins=ant_ins[j],
                            x_s_arr=x_s_arr, x_c_arr=x_c_arr, mode='multi', delta=True) - dual_station_vector[
                            ant_next[j] - 1]
                        for j in range(len(ant_next))]
                ant_profit_per_min = [ant_profit[j] / (t_p - node_t - c_mat[node_station, ant_next[j]] + 0.1)
                                      for j in range(len(ant_next))]
                ant_profit_per_min = [val if val <= 0 else val - 1000 for val in ant_profit_per_min]
                if len(ant_profit_per_min) > branch:
                    max_profit_val = heapq.nlargest(branch, ant_profit_per_min)
                    max_index = list(map(ant_profit_per_min.index, max_profit_val))
                    for idx in range(len(max_index)):
                        option = ant_next[max_index[idx]]
                        opt_ins = ant_ins[max_index[idx]]
                        assert (0 <= opt_ins +
                                round(ei_s_arr[
                                          option - 1,
                                          reg_cur_t,
                                          round(reg_cur_t + node_t + c_mat[node_station, option]) if round(
                                              reg_cur_t + node_t + c_mat[node_station, option]) < 49 else 48,
                                          x_s_arr[option - 1],
                                          x_c_arr[option - 1]]) <= c_s)
                        child_node = Node(stations=option, ins=opt_ins, profit=ant_profit[max_index[idx]],
                                          duration=c_mat[node_station, option], load_after_ins=est_load - opt_ins)
                        cur_node.add_child(child_node)
                        # route finish check
                        if child_node.total_time < t_p and cur_t + child_node.total_time < RE_END_T / 10:
                            undone_nodes.append(child_node)
                        else:
                            done_nodes.append(child_node)
                else:
                    for idx in range(len(ant_next)):
                        option = ant_next[idx]
                        opt_ins = ant_ins[idx]
                        assert (0 <= opt_ins +
                                round(ei_s_arr[
                                          option - 1,
                                          reg_cur_t,
                                          round(reg_cur_t + node_t + c_mat[node_station, option]) if round(
                                              reg_cur_t + node_t + c_mat[node_station, option]) < 49 else 48,
                                          x_s_arr[option - 1],
                                          x_c_arr[option - 1]]) <= c_s)
                        child_node = Node(stations=option, ins=opt_ins, profit=ant_profit[idx],
                                          duration=c_mat[node_station, option], load_after_ins=est_load - opt_ins)
                        cur_node.add_child(child_node)
                        # route finish check
                        if child_node.total_time < t_p and cur_t + child_node.total_time < RE_END_T / 10:
                            undone_nodes.append(child_node)
                        else:
                            done_nodes.append(child_node)

            else:  # relatively many bikes
                ant_stations = [i for i in range(1, num_of_stations + 1) if i not in cur_node.route]
                ant_unloading = [0 for _ in range(num_of_stations)]
                exp_inv = [
                    round(ei_s_arr[i,
                    reg_cur_t,
                    round(reg_cur_t + node_t + c_mat[node_station, i + 1]) if (reg_cur_t + node_t + c_mat[
                        node_station, i + 1]) < 49 else 48,
                    x_s_arr[i],
                    x_c_arr[i]])
                    for i in range(num_of_stations)]
                for j in ant_stations:
                    ant_unloading[j - 1] = max(U_REA - exp_inv[j - 1], 0)
                max_unloading = max(ant_unloading)
                if state == 'init':
                    beta_u = BETA_U
                elif state == 'dual':
                    beta_u = BETA_U_DUAL
                else:
                    raise ValueError('state should be init or dual')
                ant_next = [j for j in ant_stations if ant_unloading[j - 1] >= max_unloading * beta_u]
                # choose one or two stations to unload
                # case1: visit only one station
                ant_inv = [exp_inv[j - 1] if exp_inv[j - 1] >= U_REA else min(U_REA, exp_inv[j - 1] + est_load)
                           for j in ant_next]
                ant_ins = [ant_inv[j] - exp_inv[ant_next[j] - 1] for j in range(len(ant_next))]
                if dual_station_vector is None:
                    ant_profit = [
                        esd_computer.compute_ESD_in_horizon(
                            station_id=ant_next[j], t_arr=node_t + c_mat[node_station, ant_next[j]], ins=ant_ins[j],
                            x_s_arr=x_s_arr, x_c_arr=x_c_arr, mode='multi', delta=True) for j in range(len(ant_next))]
                else:
                    ant_profit = [
                        esd_computer.compute_ESD_in_horizon(
                            station_id=ant_next[j], t_arr=node_t + c_mat[node_station, ant_next[j]], ins=ant_ins[j],
                            x_s_arr=x_s_arr, x_c_arr=x_c_arr, mode='multi', delta=True) - dual_station_vector[
                            ant_next[j] - 1]
                        for j in range(len(ant_next))]
                ant_profit_per_min = [(ant_profit[j] / c_mat[node_station, ant_next[j]]) for j in range(len(ant_next))]
                # case2: visit two stations (a, b in order)
                visit_pair, visit_ins, visit_profit_per_min = [], [], []
                # compare and choose
                unload_profit_list = ant_profit_per_min + visit_profit_per_min
                if len(unload_profit_list) > branch:
                    max_profit_val = heapq.nlargest(branch, unload_profit_list)
                    max_index = list(map(unload_profit_list.index, max_profit_val))
                else:
                    max_index = list(range(len(unload_profit_list)))
                for idx in range(len(max_index)):
                    if max_index[idx] < len(ant_next):
                        option = ant_next[max_index[idx]]
                        opt_ins = ant_ins[max_index[idx]]
                        child_node = Node(stations=option, ins=opt_ins, profit=ant_profit[max_index[idx]],
                                          duration=c_mat[node_station, option], load_after_ins=est_load - opt_ins)
                        cur_node.add_child(child_node)
                        # route finish check
                        if child_node.total_time < t_p and cur_t + child_node.total_time < RE_END_T / 10:
                            undone_nodes.append(child_node)
                        else:
                            done_nodes.append(child_node)
                    else:  # 应该用不到
                        assert False
                        # option = visit_pair[max_index[idx] - len(ant_next)]
                        # opt_ins = visit_ins[max_index[idx] - len(ant_next)]
                        # arr_a, arr_b = (
                        #     node_t + c_mat[node_station, option[0]],
                        #     node_t + c_mat[node_station, option[0]] + c_mat[option[0], option[1]])
                        # profit_a = esd_computer.compute_ESD_in_horizon(
                        #     station_id=option[0], t_arr=arr_a, ins=opt_ins[0], x_s_arr=x_s_arr,
                        #     x_c_arr=x_c_arr, mode='multi', delta=True)
                        # profit_b = esd_computer.compute_ESD_in_horizon(
                        #     station_id=option[1], t_arr=arr_b, ins=opt_ins[1], x_s_arr=x_s_arr,
                        #     x_c_arr=x_c_arr, mode='multi', delta=True)
                        # child_node = Node(stations=option, ins=opt_ins, profit=profit_a + profit_b,
                        #                   duration=c_mat[node_station, option[0]] + c_mat[option[0], option[1]],
                        #                   load_after_ins=est_load - opt_ins[0] - opt_ins[1])
                        # cur_node.add_child(child_node)
                        # # route finish check
                        # if child_node.total_time < t_p and cur_t + child_node.total_time < RE_END_T / 10:
                        #     undone_nodes.append(child_node)
                        # else:
                        #     done_nodes.append(child_node)

        # get the best solution
        # fix done nodes
        route_list, route_ins_list, profit_list, duration_list, on_route_list = [], [], [], [], []
        end_node_list = []
        for node in done_nodes:
            # try deleting one station
            # assert len(node.route) > 3
            last_1, last_2 = node.route[-1], node.route[-2]
            if node.ins < 0:  # most recent action is loading
                # delete one
                cur_route = node.route[:-1]
                cur_route_ins = node.route_ins[:-1]
                if isinstance(node.stations, int):
                    cur_profit = node.cum_profit - node.profit
                else:
                    assert False
                    # elim_profit = esd_computer.compute_ESD_in_horizon(
                    #     station_id=last_1,
                    #     t_arr=node.total_time,
                    #     ins=node.ins[1],
                    #     x_s_arr=x_s_arr,
                    #     x_c_arr=x_c_arr,
                    #     mode='multi', delta=True
                    # )
                    # cur_profit = node.cum_profit - elim_profit
                end_node = node.parent
                if end_node not in end_node_list:
                    route_list.append(cur_route)
                    route_ins_list.append(cur_route_ins)
                    on_route_t = esd_computer.get_on_route_time(route=cur_route)
                    on_route_list.append(on_route_t)
                    profit_list.append(ORDER_INCOME_UNIT * cur_profit - alpha * on_route_t)
                    duration_list.append(node.total_time - c_mat[last_2, last_1])
                    end_node_list.append(end_node)
                else:
                    pass
            else:  # most recent action is unloading
                if node.total_time <= t_p and cur_t + node.total_time <= RE_END_T / 10:
                    end_node = node
                    if end_node not in end_node_list:
                        cur_route = node.route
                        cur_route_ins = node.route_ins
                        cur_profit = node.cum_profit
                        route_list.append(cur_route)
                        route_ins_list.append(cur_route_ins)
                        on_route_t = esd_computer.get_on_route_time(route=cur_route)
                        on_route_list.append(on_route_t)
                        profit_list.append(ORDER_INCOME_UNIT * cur_profit - alpha * on_route_t)
                        duration_list.append(node.total_time)
                        end_node_list.append(end_node)
                    else:
                        pass
                else:
                    # delete two
                    if len(node.route) < 3:
                        continue
                    assert len(node.route) >= 3
                    last_3 = node.route[-3]
                    end_node = node.parent.parent
                    if end_node not in end_node_list:
                        cur_route = node.route[:-2]
                        cur_route_ins = node.route_ins[:-2]
                        cur_profit = node.cum_profit - node.profit - node.parent.profit
                        route_list.append(cur_route)
                        route_ins_list.append(cur_route_ins)
                        on_route_t = esd_computer.get_on_route_time(route=cur_route)
                        on_route_list.append(on_route_t)
                        cur_profit = ORDER_INCOME_UNIT * cur_profit - alpha * on_route_t
                        duration_list.append(node.total_time - c_mat[last_2, last_1] - c_mat[last_3, last_2])
                        profit_list.append(cur_profit)
                        end_node_list.append(end_node)
                    else:
                        pass

        # improve profit with best loading/unloading sequence
        improve_list, improved_profit_list, positive_count = [], [], 0  # in 'init'
        red_cost_list, seq_profit_list = [], []  # in 'dual'
        if profit_list:
            perc = np.percentile(profit_list, 50)
            # print(f'perc: {perc}')
            for r in range(len(route_list)):
                if profit_list[r] >= perc:
                    cost, instruct = esd_computer.compute_route(
                        r=route_list[r],
                        t_left=root_dis_left,
                        init_l=van_load[van],
                        x_s_arr=x_s_arr,
                        x_c_arr=x_c_arr,
                    )
                else:
                    cost, instruct = profit_list[r] + alpha * on_route_list[r], route_ins_list[r]
                if state == 'init':
                    if profit_list[r] > 0:
                        positive_count += 1
                    if profit_list[r] > cost - alpha * on_route_list[r] + 1e-6:
                        logging.warning(f'{r}')
                        logging.warning(f'old & new: {profit_list[r]}, {cost - alpha * on_route_list[r]}')
                        logging.warning(f'old: {route_list[r]}, {route_ins_list[r]}')
                        logging.warning(f'new: {cost - alpha * on_route_list[r]}, {instruct}')
                        assert False
                    improve_list.append(cost - alpha * on_route_list[r] - profit_list[r])
                    improved_profit_list.append(profit_list[r])
                    route_ins_list[r] = instruct
                    # print(f'old & new: {profit_list[r]}, {cost - alpha * on_route_list[r]}')
                    profit_list[r] = cost - alpha * on_route_list[r]
                else:
                    assert state == 'dual'
                    # dual_cost = -(cost - route_cost)+dual_van+dual_stations
                    seq_profit_list.append(cost - alpha * on_route_list[r])  # original profit without dual vals
                    dual_cost = (- cost + alpha * on_route_list[r] + dual_van_vector[van] +
                                 sum([dual_station_vector[s - 1] for s in route_list[r] if s != 0]))
                    red_cost_list.append(dual_cost)
        else:
            pass

        # test best route
        # esd_computer.t_cur = 20
        # test_cost, test_instruct = esd_computer.compute_route(
        #     r=[9, 6, 19, 12, 24, 4, 20, 25],
        #     t_left=0,
        #     init_l=8,
        #     x_s_arr=[10, 5, 6, 4, 16, 9, 2, 12, 14, 3, 0, 29, 9, 6, 4, 3, 15, 0, 5, 0, 10, 2, 1, 13, 3],
        #     x_c_arr=[3, 16, 8, 0, 26, 15, 2, 16, 36, 31, 2, 56, 17, 8, 5, 4, 6, 3, 18, 1, 24, 5, 2, 6, 4],
        # )
        # print(f'test best route: {test_cost}, {test_instruct}')
        # print(f'test best profit: {test_cost - alpha * esd_computer.get_on_route_time(route=[9, 6, 19, 12, 24, 4, 20, 25])}')
        # print(esd_computer.get_on_route_time(route=[0, 5, 24, 10, 23, 21, 4, 8, 18]))
        # best solution route: [0, 5, 24, 10, 23, 21, 4, 8, 18]
        # best solution instructions: [0, -25, 15, -15, 20, -20, 25, -25, 25]

    if state == 'init':
        # 10 best improvements
        if positive_count == 0:
            print('no positive profits')
        else:
            print(f'positive count: {positive_count}, {positive_count / len(route_list)}')
        # best_improve = heapq.nlargest(10, improve_list)
        # best_improve_index = list(map(improve_list.index, best_improve))
        # print('ten best improves: ', best_improve)
        # print('ten best profits: ', list(map(profit_list.__getitem__, best_improve_index)))
        # print(f'mean improve: {sum(improve_list) / len(improve_list)}')

        best_profits = heapq.nlargest(10, profit_list)
        print('ten best routes: ', best_profits)
        best_idx = list(map(profit_list.index, best_profits))
        # for idx in best_idx:
        #     print('profit: ', profit_list[idx])
        #     print('route: ', route_list[idx])
        #     print('route_ins: ', route_ins_list[idx])
        sel_route_list, sel_profit_list = [], []
        best_profits = heapq.nlargest(NUM_INIT_ROUTES, profit_list)
        best_idx = list(map(profit_list.index, best_profits))
        for idx in best_idx:
            if profit_list[idx] > 0:
                sel_route_list.append(route_list[idx])
                # sel_profit_list.append(profit_list[idx] + sum([esd_computer.compute_ESD_in_horizon(
                #     station_id=k, t_arr=0, ins=0, x_s_arr=x_s_arr, x_c_arr=x_c_arr, mode='multi', delta=True, repo=False)
                #     for k in route_list[idx] if k != 0]))
                sel_profit_list.append(profit_list[idx])
        return sel_route_list, sel_profit_list

    elif state == 'dual':
        assert dual_van_vector is not None and dual_station_vector is not None
        sort_red_cost, sort_route, sort_profit = (list(t) for t in
                                                  zip(*sorted(zip(red_cost_list, route_list, seq_profit_list))))
        sel_route_list, sel_profit_list = [], []
        for r in range(len(sort_red_cost)):
            if sort_red_cost[r] < 0:
                sel_route_list.append(sort_route[r])
                # sel_profit_list.append(seq_profit_list[r] + sum([esd_computer.compute_ESD_in_horizon(
                #     station_id=k, t_arr=0, ins=0, x_s_arr=x_s_arr, x_c_arr=x_c_arr, mode='multi', delta=True, repo=False)
                #     for k in route_list[idx] if k != 0]))
                sel_profit_list.append(sort_profit[r])
            if len(sel_route_list) >= NUM_DUAL_ROUTES:
                break
        return sel_route_list, sel_profit_list  # with negative reduced cost

    # choose the best solution
    # max_profit = max(profit_list)
    # best_idx = profit_list.index(max_profit)
    # best_route = route_list[best_idx]
    # best_route_ins = route_ins_list[best_idx]
    # best_profit = max_profit
    # # return the best solution
    # result = {
    #     'objective': best_profit,
    #     'start_time': cur_t,
    #     'routes': best_route,
    # }
    #
    # 'n_r', 'loc', 'exp_inv', 'exp_target_inv'
