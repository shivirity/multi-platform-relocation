import numba
from numba.typed import List
import time
import logging
import numpy as np
from route_extension.route_extension_algo import get_REA_routes_test, ESDComputer
from simulation.consts import RE_START_T, RE_END_T, ORDER_INCOME_UNIT
from gurobipy import *


def get_node_mat(num_stations: int, route_pool: list):
    """generate node_mat from the route pool (list of lists)"""
    num_routes = len(route_pool)
    node_mat = np.zeros((num_stations, num_routes))
    for j in range(num_routes):
        for i in range(1, num_stations + 1):
            if i in route_pool[j]:
                node_mat[i - 1, j] = 1
    return node_mat


def init_LP_relaxed_RMP(num_stations: int, route_pool: list, profit_pool: list,
                        veh_mat: np.ndarray, node_mat: np.ndarray, station_esd_list: list):
    """
    solve LP-related RMP

    :param num_stations: 站点数量
    :param route_pool: 备选路径
    :param profit_pool: 备选路径收益, 存储路径收益的 delta 值
    :param veh_mat: 路线-车辆矩阵
    :param node_mat: 路线-站点矩阵
    :param station_esd_list: 站点ESD列表
    :return:
    """
    num_routes, num_veh = len(route_pool), veh_mat.shape[0]
    rmp = Model('RMP')
    # variables
    x = {}
    for j in range(num_routes):
        x[j] = rmp.addVar(vtype=GRB.BINARY, name=f'x{j}')
    # constraints
    # vehicle constr
    veh_cons = {}
    for j in range(num_veh):
        veh_cons[j] = rmp.addConstr(quicksum(veh_mat[j, k] * x[k] for k in range(num_routes)) <= 1, name=f'veh_{j}')
    # node constr
    node_cons = {}
    for j in range(num_stations):
        node_cons[j] = rmp.addConstr(quicksum(node_mat[j, k] * x[k] for k in range(num_routes)) <= 1, name=f'node_{j}')
    # objective
    rmp.setObjective(
        ORDER_INCOME_UNIT * sum(station_esd_list) + quicksum(profit_pool[j] * x[j] for j in range(num_routes)),
        GRB.MAXIMIZE)

    rmp.update()
    relax_RMP = rmp.relax()
    relax_RMP.setParam('OutputFlag', 0)
    relax_RMP.optimize()
    # get dual vector
    dual_vector = [con.Pi for con in relax_RMP.getConstrs()]
    return rmp, veh_cons, node_cons, dual_vector


def solve_RMP(model: Model, routes_pool: list) -> tuple:
    model.setParam('OutputFlag', 0)
    model.optimize()
    route_ind = [j for j in range(len(routes_pool)) if model.getVarByName(f'x{j}').x > 0.5][0]
    return routes_pool[route_ind], model.objVal


def is_dominated(label1: list, label2: list) -> bool:
    """check if label1 dominates label2, returns True if yes, False otherwise"""
    # label: [] -> [reward, visited_set]
    if label1[0] >= label2[0] and label1[1] <= label2[1]:  # set1是set2的子集
        return True
    else:
        return False


def get_dp_reduced_cost_forward(cap_v: int, cap_s: int, num_stations: int, init_loc: int, init_t_left: int,
                                init_load: int, x_s_arr: list, x_c_arr: list, ei_s_arr: np.ndarray,
                                ei_c_arr: np.ndarray,
                                esd_arr: np.ndarray, c_mat: np.ndarray, cur_t: int, t_p: int, t_f: int, t_roll: int,
                                alpha: float, dual_van_vec: list, dual_station_vec: list):
    """calculate exact reduced cost using dynamic programming"""
    com = ESDComputer(
        esd_arr=esd_arr, ei_s_arr=ei_s_arr, ei_c_arr=ei_c_arr, t_cur=cur_t, t_fore=t_f, c_mat=c_mat)
    cur_t = round(cur_t - RE_START_T / 10)
    # post decision state inventory
    # inv_dict = {i: i for i in range(0, cap_v + 1)}  # level_id: inv_level_on_veh
    # inv_id_dict = {i: i for i in range(0, cap_v + 1)}  # inv_level_on_veh: inv_id
    # inv_dict = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 6: 12, 7: 14, 8: 16, 9: 18, 10: 20, 11: 22, 12: 25}  # level_id: inv_level_on_veh
    # inv_id_dict = {25: 12, 22: 11, 20: 10, 18: 9, 16: 8, 14: 7, 12: 6, 10: 5, 8: 4, 6: 3, 4: 2, 2: 1, 0: 0}  # inv_level_on_veh: inv_id
    inv_dict = {0: 0, 1: 5, 2: 10, 3: 15, 4: 20, 5: 25}  # level_id: inv_level_on_veh
    inv_id_dict = {25: 5, 20: 4, 15: 3, 10: 2, 5: 1, 0: 0}  # inv_level_on_veh: inv_id
    # inv_dict = {0: 0, 1: 5, 2: 20, 3: 25}  # level_id: inv_level_on_veh
    # inv_id_dict = {25: 3, 20: 2, 5: 1, 0: 0}  # inv_level_on_veh: inv_id
    inv_num = len(inv_dict)
    t_repo = t_p if RE_START_T / 10 + cur_t + t_p <= RE_END_T / 10 else round(RE_END_T / 10 - cur_t - RE_START_T / 10)
    if t_repo == 1:
        return [init_loc], com.compute_ESD_in_horizon(station_id=init_loc, t_arr=0, ins=init_load, x_s_arr=x_s_arr,
                                                      x_c_arr=x_c_arr, mode='multi', delta=True, repo=True)
    elif t_repo == 0:
        assert False
    print('in get_dp_reduced_cost(), t_repo =', t_repo)
    reward_arr = [[[None for _ in range(inv_num)] for __ in range(1 + num_stations)] for ___ in range(t_repo + 1)]
    trace_arr = [[[None for _ in range(inv_num)] for __ in range(1 + num_stations)] for ___ in range(t_repo + 1)]
    calcu_arr = [[False for _ in range(num_stations)] for __ in range(t_repo + 1)]
    st = time.process_time()
    for t in range(t_repo + 1):
        if t == init_t_left:
            stept = time.process_time()
            print(f't={t}, time: {stept - st}')
            # (reduced_cost, visited_set)
            if init_loc == 0:
                assert init_load == 0
                # reward_arr[t][init_loc-1][inv_id_dict[init_load]] = [(0, {init_loc})]
                reward_arr[t][init_loc][inv_id_dict[init_load]] = [(0, {init_loc})]
                cur_reward, cur_set = (0, {init_loc})
                for ne in range(num_stations + 1):  # cannot stay at depot
                    if ne == 0:
                        stay_t = 1
                        inv = inv_id_dict[init_load]
                        if t + stay_t <= t_repo:
                            if reward_arr[t + stay_t][ne][inv] is None:
                                new_reward = cur_reward
                                reward_arr[t + stay_t][ne][inv] = [(new_reward, cur_set)]
                                trace_arr[t + stay_t][ne][inv] = [(t, init_loc, inv, 0)]
                            else:  # dominate rules applied
                                assert False
                    else:
                        arr_t = round(c_mat[init_loc, ne])
                        if t + arr_t <= t_repo:
                            calcu_arr[arr_t][ne - 1] = True
                            for inv in range(inv_num):
                                ins = 0 - inv_dict[inv]
                                if 0 <= ei_s_arr[
                                    ne - 1, cur_t, cur_t + t + arr_t, x_s_arr[ne - 1], x_c_arr[ne - 1]] + ins <= cap_s:
                                    reward_arr[arr_t][ne][inv] = [(
                                        ORDER_INCOME_UNIT * com.compute_ESD_in_horizon(
                                            station_id=ne,
                                            t_arr=t + arr_t,
                                            ins=ins,
                                            x_s_arr=x_s_arr,
                                            x_c_arr=x_c_arr,
                                            mode='multi',
                                            delta=True,
                                            repo=True
                                        ) - alpha * arr_t - dual_station_vec[ne - 1], {init_loc, ne})]
                                    trace_arr[arr_t][ne][inv] = [
                                        (init_t_left, init_loc, inv_id_dict[init_load], 0)]  # time-space index
                                else:
                                    pass
            else:  # init_loc > 0
                for inv in range(inv_num):  # label every inventory level at initial point
                    ins = init_load - inv_dict[inv]
                    if 0 <= ei_s_arr[
                        init_loc - 1, cur_t, cur_t + t, x_s_arr[init_loc - 1], x_c_arr[init_loc - 1]] + ins <= cap_s:
                        reward_arr[t][init_loc][inv] = [(
                            ORDER_INCOME_UNIT * com.compute_ESD_in_horizon(
                                station_id=init_loc,
                                t_arr=t,
                                ins=ins,
                                x_s_arr=x_s_arr,
                                x_c_arr=x_c_arr,
                                mode='multi',
                                delta=True,
                                repo=True
                            ) - dual_station_vec[init_loc - 1], {init_loc})]
                        cur_reward, cur_set = reward_arr[t][init_loc][inv][0]
                        # trace to time step 0
                        for ne in range(1, num_stations + 1):
                            if ne == init_loc:
                                stay_t = 1
                                if t + stay_t <= t_repo:
                                    calcu_arr[t + stay_t][ne - 1] = True
                                    if reward_arr[t + stay_t][ne][inv] is None:
                                        new_reward = cur_reward
                                        reward_arr[t + stay_t][ne][inv] = [(new_reward, cur_set)]
                                        trace_arr[t + stay_t][ne][inv] = [
                                            (t, init_loc, inv, 0)]
                                    else:  # dominate rules applied
                                        new_reward = cur_reward
                                        tmp_label = [new_reward, cur_set]
                                        dom_idx = []
                                        for ne_label_id in range(
                                                len(reward_arr[t + stay_t][ne][inv])):
                                            ne_label = reward_arr[t + stay_t][ne][inv][ne_label_id]
                                            if is_dominated(label1=tmp_label, label2=ne_label):
                                                dom_idx.append(ne_label_id)
                                            elif is_dominated(label1=ne_label, label2=tmp_label):
                                                assert not dom_idx  # dom_idx is empty
                                                break
                                        else:
                                            if len(dom_idx) == 0:  # no domination
                                                reward_arr[t + stay_t][ne][inv].append(tmp_label)
                                                trace_arr[t + stay_t][ne][inv].append((t, init_loc, inv, 0))
                                            else:
                                                dom_idx.sort(reverse=True)  # 从后往前删除，避免索引错位
                                                # first delete
                                                for idx in dom_idx:
                                                    reward_arr[t + stay_t][ne][inv].pop(idx)
                                                    trace_arr[t + stay_t][ne][inv].pop(idx)
                                                # then add
                                                reward_arr[t + stay_t][ne][inv].append(tmp_label)
                                                trace_arr[t + stay_t][ne][inv].append((t, init_loc, inv, 0))
                            else:
                                arr_t = round(c_mat[init_loc, ne])
                                if t + arr_t <= t_repo:
                                    calcu_arr[t + arr_t][ne - 1] = True
                                    for ne_inv in range(inv_num):
                                        ins = inv_dict[inv] - inv_dict[ne_inv]
                                        if 0 <= ei_s_arr[
                                            ne - 1, cur_t, cur_t + t + arr_t, x_s_arr[ne - 1], x_c_arr[
                                                ne - 1]] + ins <= cap_s:
                                            if reward_arr[t + arr_t][ne][ne_inv] is None:
                                                reward_arr[t + arr_t][ne][ne_inv] = [(
                                                    cur_reward + ORDER_INCOME_UNIT * com.compute_ESD_in_horizon(
                                                        station_id=ne,
                                                        t_arr=t + arr_t,
                                                        ins=ins,
                                                        x_s_arr=x_s_arr,
                                                        x_c_arr=x_c_arr,
                                                        mode='multi',
                                                        delta=True,
                                                        repo=True
                                                    ) - alpha * (arr_t - 1) - dual_station_vec[ne - 1], {init_loc, ne})]
                                                trace_arr[t + arr_t][ne][ne_inv] = [(t, init_loc, inv, 0)]
                                            else:
                                                new_reward = cur_reward + ORDER_INCOME_UNIT * com.compute_ESD_in_horizon(
                                                    station_id=ne,
                                                    t_arr=t + arr_t,
                                                    ins=ins,
                                                    x_s_arr=x_s_arr,
                                                    x_c_arr=x_c_arr,
                                                    mode='multi',
                                                    delta=True,
                                                    repo=True) - alpha * (arr_t - 1) - dual_station_vec[ne - 1]
                                                assert len(reward_arr[t + arr_t][ne][ne_inv]) == 1
                                                if new_reward > reward_arr[t + arr_t][ne][ne_inv][0][0]:
                                                    reward_arr[t + arr_t][ne][ne_inv] = [(new_reward, {init_loc, ne})]
                                                    trace_arr[t + arr_t][ne][ne_inv] = [(t, init_loc, inv, 0)]
                                                else:
                                                    pass
        elif t > init_t_left:  # t > init_t_left
            if t == t_repo - 1:
                stept = time.process_time()
                print(f't={t}, time: {stept - st}')
                break
            else:
                stept = time.process_time()
                print(f't={t}, time: {stept - st}')
                for cur_s in range(num_stations + 1):
                    if cur_s > 0 and calcu_arr[t][cur_s - 1] is False:
                        pass
                    else:
                        for inv in range(inv_num):
                            if reward_arr[t][cur_s][inv] is None:
                                pass
                            else:  # select labels to extend
                                for label_id in range(len(reward_arr[t][cur_s][inv])):
                                    label = reward_arr[t][cur_s][inv][label_id]
                                    # forward update
                                    cur_reward, cur_set = label
                                    for next_s in range(1, num_stations + 1):
                                        # stay at 本站
                                        if next_s == cur_s:
                                            stay_t = 1
                                            if t + stay_t <= t_repo:
                                                if t + stay_t < t_repo - 1 or inv_dict[
                                                    inv] == 0:  # 最后一站放空，不改变库存所以remain inv
                                                    calcu_arr[t + stay_t][next_s - 1] = True
                                                    if reward_arr[t + stay_t][next_s][inv] is None:
                                                        new_reward = cur_reward
                                                        reward_arr[t + stay_t][next_s][inv] = [(new_reward, cur_set)]
                                                        trace_arr[t + stay_t][next_s][inv] = [
                                                            (t, cur_s, inv, label_id)]
                                                    else:  # dominate rules applied
                                                        new_reward = cur_reward
                                                        tmp_label = [new_reward, cur_set]
                                                        dom_idx = []
                                                        for ne_label_id in range(
                                                                len(reward_arr[t + stay_t][next_s][inv])):
                                                            ne_label = reward_arr[t + stay_t][next_s][inv][ne_label_id]
                                                            if is_dominated(label1=tmp_label, label2=ne_label):
                                                                dom_idx.append(ne_label_id)
                                                            elif is_dominated(label1=ne_label, label2=tmp_label):
                                                                assert not dom_idx  # dom_idx is empty
                                                                break
                                                        else:
                                                            if len(dom_idx) == 0:  # no domination
                                                                reward_arr[t + stay_t][next_s][inv].append(tmp_label)
                                                                trace_arr[t + stay_t][next_s][inv].append(
                                                                    (t, cur_s, inv, label_id))
                                                            else:
                                                                dom_idx.sort(reverse=True)  # 从后往前删除，避免索引错位
                                                                # first delete
                                                                for idx in dom_idx:
                                                                    reward_arr[t + stay_t][next_s][inv].pop(idx)
                                                                    trace_arr[t + stay_t][next_s][inv].pop(idx)
                                                                # then add
                                                                reward_arr[t + stay_t][next_s][inv].append(tmp_label)
                                                                trace_arr[t + stay_t][next_s][inv].append(
                                                                    (t, cur_s, inv, label_id))
                                        elif next_s in cur_set:
                                            pass
                                        else:
                                            arr_t = round(c_mat[cur_s, next_s])
                                            if t + arr_t <= t_repo:
                                                if t + arr_t < t_repo - 1:
                                                    can_do_inv = inv_num
                                                else:
                                                    can_do_inv = 1
                                                for next_inv in range(can_do_inv):
                                                    # if t == 3 and cur_s == 3 and next_s == 12 and next_inv == 5:
                                                    #     logging.info('debug')
                                                    ins = inv_dict[inv] - inv_dict[next_inv]
                                                    if 0 <= ei_s_arr[
                                                        next_s - 1, cur_t, cur_t + t + arr_t, x_s_arr[next_s - 1],
                                                        x_c_arr[next_s - 1]] + ins <= cap_s:
                                                        calcu_arr[t + arr_t][next_s - 1] = True
                                                        if reward_arr[t + arr_t][next_s][next_inv] is None:
                                                            new_reward = cur_reward + ORDER_INCOME_UNIT * com.compute_ESD_in_horizon(
                                                                station_id=next_s,
                                                                t_arr=t + arr_t,
                                                                ins=ins,
                                                                x_s_arr=x_s_arr,
                                                                x_c_arr=x_c_arr,
                                                                mode='multi',
                                                                delta=True,
                                                                repo=True) - alpha * (arr_t - 1) - dual_station_vec[
                                                                             next_s - 1]
                                                            reward_arr[t + arr_t][next_s][next_inv] = [
                                                                (new_reward, cur_set | {next_s})]
                                                            trace_arr[t + arr_t][next_s][next_inv] = [
                                                                (t, cur_s, inv, label_id)]
                                                        else:  # dominate rules applied
                                                            new_reward = cur_reward + ORDER_INCOME_UNIT * com.compute_ESD_in_horizon(
                                                                station_id=next_s,
                                                                t_arr=t + arr_t,
                                                                ins=ins,
                                                                x_s_arr=x_s_arr,
                                                                x_c_arr=x_c_arr,
                                                                mode='multi',
                                                                delta=True,
                                                                repo=True) - alpha * (arr_t - 1) - dual_station_vec[
                                                                             next_s - 1]
                                                            tmp_label = [new_reward, cur_set | {next_s}]
                                                            dom_idx = []
                                                            for ne_label_id in range(
                                                                    len(reward_arr[t + arr_t][next_s][next_inv])):
                                                                ne_label = reward_arr[t + arr_t][next_s][next_inv][
                                                                    ne_label_id]
                                                                if is_dominated(label1=tmp_label, label2=ne_label):
                                                                    dom_idx.append(ne_label_id)
                                                                elif is_dominated(label1=ne_label, label2=tmp_label):
                                                                    assert not dom_idx  # dom_idx is empty
                                                                    break
                                                            else:
                                                                if len(dom_idx) == 0:  # no domination
                                                                    reward_arr[t + arr_t][next_s][next_inv].append(
                                                                        tmp_label)
                                                                    trace_arr[t + arr_t][next_s][next_inv].append(
                                                                        (t, cur_s, inv, label_id))
                                                                else:
                                                                    dom_idx.sort(reverse=True)  # 从后往前删除，避免索引错位
                                                                    # first delete
                                                                    for idx in dom_idx:
                                                                        reward_arr[t + arr_t][next_s][next_inv].pop(
                                                                            idx)
                                                                        trace_arr[t + arr_t][next_s][next_inv].pop(
                                                                            idx)
                                                                    # then add
                                                                    reward_arr[t + arr_t][next_s][next_inv].append(
                                                                        tmp_label)
                                                                    trace_arr[t + arr_t][next_s][next_inv].append(
                                                                        (t, cur_s, inv, label_id))

    # print(reward_arr[11][17][0], reward_arr[12][17][0])
    # for label in reward_arr[11][17][0]:
    #     if label[1].issubset({3, 19, 12, 16, 8, 18}):
    #         print(label)
    # print('compute_route: ', com.compute_route(r=[3, 19, 12, 16, 8, 18], t_left=0, init_l=13, x_s_arr=x_s_arr, x_c_arr=x_c_arr))

    max_reward_list, max_label_list = [], []

    label_length_test = []

    for s in range(num_stations + 1):
        if s == init_loc:
            pass
        else:
            for inv in range(inv_num):
                if reward_arr[t_repo][s][inv] is not None:
                    for l_id in range(len(reward_arr[t_repo][s][inv])):
                        max_reward_list.append(reward_arr[t_repo][s][inv][l_id][0])
                        max_label_list.append((t_repo, s, inv, l_id))
                        label_length_test.append(len(reward_arr[t_repo][s][inv]))
                if reward_arr[t_repo - 1][s][inv] is not None:
                    for l_id in range(len(reward_arr[t_repo - 1][s][inv])):
                        max_reward_list.append(reward_arr[t_repo - 1][s][inv][l_id][0])
                        max_label_list.append((t_repo - 1, s, inv, l_id))
                        label_length_test.append(len(reward_arr[t_repo - 1][s][inv]))
    if max_reward_list:
        max_reward = max(max_reward_list)
        print(max(max_reward_list))
        max_label = max_label_list[max_reward_list.index(max_reward)]
        print(reward_arr[max_label[0]][max_label[1]][max_label[2]][max_label[3]])
        print(f'max label length: {max(label_length_test)}')
        k_t_repo, k_s, k_inv, k_l_id = max_label
        loc_list, inv_list = [-1 for _ in range(t_repo + 1)], [-1 for _ in range(t_repo + 1)]
        while True:
            if k_t_repo == 0:
                assert False
            else:
                loc_list[k_t_repo] = k_s
                inv_list[k_t_repo] = inv_dict[k_inv]
                k_t_repo, k_s, k_inv, k_l_id = trace_arr[k_t_repo][k_s][k_inv][k_l_id]
                if k_t_repo == init_t_left:
                    loc_list[k_t_repo] = k_s
                    inv_list[k_t_repo] = inv_dict[k_inv]
                    break
        print(loc_list)
        print(inv_list)

        # delete remaining in route
        clean_route = []
        for k in loc_list:
            if k not in clean_route and k > -0.5:
                clean_route.append(k)
    else:  # time is too short
        loc_list, inv_list = [-1 for _ in range(t_repo + 1)], [-1 for _ in range(t_repo + 1)]
        for step in range(init_t_left, t_repo + 1):
            loc_list[step] = init_loc
            inv_list[step] = init_load
        clean_route = [init_loc]
        max_reward = 0  # can be fixed

    return clean_route, max_reward


@numba.jit(
    'i1[:](i8,i8,i8,i8,i8,i4[:],i4[:],f8[:,:,:,:,:],f8[:,:,:,:,:],f8[:,:,:,:,:],f8[:,:],i8,i8,i8,i8,i4[:],i4[:])',
    nopython=True, nogil=True)
def get_dp_reduced_cost_numba(cap_v: int, cap_s: int, num_stations: int, init_loc: int, init_load: int,
                              x_s_arr: np.ndarray, x_c_arr: np.ndarray, ei_s_arr: np.ndarray, ei_c_arr: np.ndarray,
                              esd_arr: np.ndarray, c_mat: np.ndarray, cur_t: int, t_p: int, t_f: int,
                              alpha: float, dual_van_vec: np.ndarray, dual_station_vec: np.ndarray):
    """calculate exact reduced cost using dynamic programming (accelerated by numba)"""
    cur_t = round(cur_t - RE_START_T / 10)
    # post decision state inventory
    # inv_dict = {i: i for i in range(0, cap_v + 1)}  # level_id: inv_level_on_veh
    # inv_id_dict = {i: i for i in range(0, cap_v + 1)}  # inv_level_on_veh: inv_id
    # inv_dict = {0: 0, 1: 5, 2: 10, 3: 15, 4: 20, 5: 25}  # level_id: inv_level_on_veh
    # inv_id_dict = {25: 5, 20: 4, 15: 3, 10: 2, 5: 1, 0: 0}  # inv_level_on_veh: inv_id
    # inv_dict = {0: 0, 1: 5, 2: 20, 3: 25}  # level_id: inv_level_on_veh
    # inv_id_dict = {25: 3, 20: 2, 5: 1, 0: 0}  # inv_level_on_veh: inv_id

    inv_dict_arr = np.array([0, 5, 10, 15, 20, 25], dtype=np.int8)
    inv_id_dict_arr = np.array([0, 0, 0, 0, 0,
                                1, 1, 1, 1, 1,
                                2, 2, 2, 2, 2,
                                3, 3, 3, 3, 3,
                                4, 4, 4, 4, 4,
                                5, 5, 5, 5, 5], dtype=np.int8)
    # inv_dict_arr = np.array([0, 5, 20, 25], dtype=np.int8)
    # inv_id_dict_arr = np.array([0, 0, 0, 0, 0,
    #                             1, 1, 1, 1, 1,
    #                             2, 2, 2, 2, 2,
    #                             3, 3, 3, 3, 3,
    #                             2, 2, 2, 2, 2,
    #                             3, 3, 3, 3, 3], dtype=np.int8)

    # inv_num = len(inv_dict)
    inv_num = inv_dict_arr.shape[0]
    t_repo = t_p if RE_START_T / 10 + cur_t + t_p <= RE_END_T / 10 else round(RE_END_T / 10 - cur_t - RE_START_T / 10)
    if t_repo == 1:
        # after_val = esd_arr[
        #     init_loc - 1,
        #     cur_t,
        #     cur_t + t_f if cur_t + t_f < 49 else 48,
        #     x_s_arr[init_loc - 1] + init_load,
        #     x_c_arr[init_loc - 1]]
        # original_val = esd_arr[
        #     init_loc - 1,
        #     cur_t,
        #     cur_t + t_f if cur_t + t_f < 49 else 48,
        #     x_s_arr[init_loc - 1],
        #     x_c_arr[init_loc - 1]
        # ]
        # return_val = after_val - original_val
        return np.array([init_loc], dtype=np.int8)

    elif t_repo == 0:
        assert False
    print('in get_dp_reduced_cost(), t_repo =', t_repo)
    # reward_arr = [[[None for _ in range(inv_num)] for __ in range(1 + num_stations)] for ___ in range(t_repo + 1)]
    # trace_arr = [[[None for _ in range(inv_num)] for __ in range(1 + num_stations)] for ___ in range(t_repo + 1)]
    # calcu_arr = [[False for _ in range(num_stations)] for __ in range(t_repo + 1)]
    max_label_num = 1200
    label_num_arr = np.zeros((t_repo + 1, num_stations + 1, inv_num), dtype=np.int16)
    reward_val_arr = np.zeros((t_repo + 1, num_stations + 1, inv_num, max_label_num), dtype=np.float64)
    reward_set_arr = np.zeros((t_repo + 1, num_stations + 1, inv_num, max_label_num, num_stations + 1), dtype=np.bool_)
    trace_t_arr = np.zeros((t_repo + 1, num_stations + 1, inv_num, max_label_num), dtype=np.int8)
    trace_s_arr = np.zeros((t_repo + 1, num_stations + 1, inv_num, max_label_num), dtype=np.int8)
    trace_inv_arr = np.zeros((t_repo + 1, num_stations + 1, inv_num, max_label_num), dtype=np.int8)
    trace_lid_arr = np.zeros((t_repo + 1, num_stations + 1, inv_num, max_label_num), dtype=np.int16)
    calcu_arr = np.zeros((t_repo + 1, num_stations), dtype=np.bool_)

    # st = time.process_time()
    for t in range(t_repo + 1):
        if t == 0:
            # stept = time.process_time()
            # print(f't={t}, time: {stept - st}')
            # (reduced_cost, visited_set)
            if init_loc == 0:
                assert init_load == 0
                label_num_arr[t, init_loc, inv_id_dict_arr[init_load]] = 1
                reward_val_arr[t, init_loc, inv_id_dict_arr[init_load], 0] = 0
                reward_set_arr[t, init_loc, inv_id_dict_arr[init_load], 0, init_loc] = True
                # cur_reward, cur_set = (0, {init_loc})
                cur_reward = 0
                for ne in range(num_stations + 1):  # can stay at depot
                    if ne == 0:
                        stay_t = 1
                        inv = inv_id_dict_arr[init_load]
                        if t + stay_t <= t_repo:
                            if label_num_arr[t + stay_t, ne, inv] == 0:
                                new_reward = cur_reward

                                label_num_arr[t + stay_t, ne, inv] = 1
                                reward_val_arr[t + stay_t, ne, inv, 0] = new_reward
                                reward_set_arr[t + stay_t, ne, inv, 0, 0] = True

                                trace_t_arr[t + stay_t, ne, inv, 0] = t
                                trace_s_arr[t + stay_t, ne, inv, 0] = init_loc
                                trace_inv_arr[t + stay_t, ne, inv, 0] = inv
                                trace_lid_arr[t + stay_t, ne, inv, 0] = 0

                            else:  # dominate rules applied
                                assert False
                    else:
                        arr_t = round(c_mat[init_loc, ne])
                        if t + arr_t <= t_repo:
                            calcu_arr[arr_t, ne - 1] = True
                            for inv in range(inv_num):
                                ins = 0 - inv_dict_arr[inv]
                                if (0 <=
                                        ei_s_arr[ne - 1, cur_t, cur_t + arr_t, x_s_arr[ne - 1], x_c_arr[ne - 1]] + ins
                                        <= cap_s):

                                    before_val = esd_arr[
                                        ne - 1,
                                        cur_t,
                                        cur_t + arr_t if cur_t + arr_t < 49 else 48,
                                        x_s_arr[ne - 1],
                                        x_c_arr[ne - 1]]
                                    after_val = esd_arr[
                                        ne - 1,
                                        cur_t + arr_t if cur_t + arr_t < 36 else 35,
                                        cur_t + t_f if cur_t + t_f < 49 else 48,
                                        round(ei_s_arr[
                                                  ne - 1,
                                                  cur_t,
                                                  cur_t + arr_t if cur_t + arr_t < 49 else 48,
                                                  x_s_arr[ne - 1],
                                                  x_c_arr[ne - 1]] + ins),
                                        round(ei_c_arr[
                                                  ne - 1,
                                                  cur_t,
                                                  cur_t + arr_t if cur_t + arr_t < 49 else 48,
                                                  x_s_arr[ne - 1],
                                                  x_c_arr[ne - 1]])
                                    ]
                                    original_val = esd_arr[
                                        ne - 1,
                                        cur_t,
                                        cur_t + t_f if cur_t + t_f < 49 else 48,
                                        x_s_arr[ne - 1],
                                        x_c_arr[ne - 1]
                                    ]
                                    computed_ESD = before_val + after_val - original_val

                                    label_num_arr[arr_t, ne, inv] = 1
                                    reward_val_arr[arr_t, ne, inv, 0] = (
                                            ORDER_INCOME_UNIT * computed_ESD - alpha * arr_t - dual_station_vec[ne - 1])
                                    reward_set_arr[arr_t, ne, inv, 0, init_loc] = True
                                    reward_set_arr[arr_t, ne, inv, 0, ne] = True

                                    trace_t_arr[arr_t, ne, inv, 0] = 0
                                    trace_s_arr[arr_t, ne, inv, 0] = init_loc
                                    trace_inv_arr[arr_t, ne, inv, 0] = inv_id_dict_arr[init_load]
                                    trace_lid_arr[arr_t, ne, inv, 0] = 0  # time-space index

                                else:
                                    pass
            else:  # init_loc > 0
                for inv in range(inv_num):  # label every inventory level at initial point
                    ins = init_load - inv_dict_arr[inv]
                    if 0 <= x_s_arr[init_loc - 1] + ins <= cap_s:

                        after_val = esd_arr[
                            init_loc - 1,
                            cur_t,
                            cur_t + t_f if cur_t + t_f < 49 else 48,
                            x_s_arr[init_loc - 1] + ins,
                            x_c_arr[init_loc - 1]
                        ]
                        original_val = esd_arr[
                            init_loc - 1,
                            cur_t,
                            cur_t + t_f if cur_t + t_f < 49 else 48,
                            x_s_arr[init_loc - 1],
                            x_c_arr[init_loc - 1]
                        ]
                        computed_ESD = after_val - original_val

                        label_num_arr[t, init_loc, inv] = 1
                        reward_val_arr[t, init_loc, inv, 0] = ORDER_INCOME_UNIT * computed_ESD - dual_station_vec[
                            init_loc - 1]
                        reward_set_arr[t, init_loc, inv, 0, init_loc] = True

                        cur_reward = ORDER_INCOME_UNIT * computed_ESD - dual_station_vec[init_loc - 1]
                        # current set == {init_loc}
                        # trace to time step 0
                        for ne in range(1, num_stations + 1):
                            if ne == init_loc:
                                stay_t = 1
                                if t + stay_t <= t_repo:
                                    calcu_arr[t + stay_t, ne - 1] = True
                                    if label_num_arr[t + stay_t, ne, inv] == 0:
                                        new_reward = cur_reward

                                        reward_val_arr[t + stay_t, ne, inv, 0] = new_reward
                                        reward_set_arr[t + stay_t, ne, inv, 0, init_loc] = True

                                        trace_t_arr[t + stay_t, ne, inv, 0] = t
                                        trace_s_arr[t + stay_t, ne, inv, 0] = init_loc
                                        trace_inv_arr[t + stay_t, ne, inv, 0] = inv
                                        trace_lid_arr[t + stay_t, ne, inv, 0] = 0

                                    else:  # dominate rules applied
                                        # new_reward = cur_reward
                                        # tmp_label = [new_reward, cur_set]
                                        tmp_val = cur_reward
                                        tmp_set = reward_set_arr[t, init_loc, inv, 0, :].copy()
                                        dom_idx = List()
                                        for ne_label_id in range(label_num_arr[t + stay_t, ne, inv]):
                                            ne_val = reward_val_arr[t + stay_t, ne, inv, ne_label_id]
                                            ne_set = reward_set_arr[t + stay_t, ne, inv, ne_label_id, :].copy()

                                            if tmp_val >= ne_val and not np.any(tmp_set > ne_set):  # set1是set2的子集
                                                dom_idx.append(ne_label_id)
                                            elif ne_val >= tmp_val and not np.any(ne_set > tmp_set):
                                                assert len(dom_idx) == 0  # dom_idx is empty
                                                break
                                        else:
                                            if len(dom_idx) == 0:  # no domination
                                                cur_label_num = label_num_arr[t + stay_t, ne, inv]
                                                label_num_arr[t + stay_t, ne, inv] += 1
                                                reward_val_arr[t + stay_t, ne, inv, cur_label_num] = tmp_val
                                                reward_set_arr[t + stay_t, ne, inv, cur_label_num, :] = tmp_set
                                                trace_t_arr[t + stay_t, ne, inv, cur_label_num] = t
                                                trace_s_arr[t + stay_t, ne, inv, cur_label_num] = init_loc
                                                trace_inv_arr[t + stay_t, ne, inv, cur_label_num] = inv
                                                trace_lid_arr[t + stay_t, ne, inv, cur_label_num] = 0
                                            elif len(dom_idx) == 1:
                                                change_idx = dom_idx[0]
                                                reward_val_arr[t + stay_t, ne, inv, change_idx] = tmp_val
                                                reward_set_arr[t + stay_t, ne, inv, change_idx, :] = tmp_set
                                                trace_t_arr[t + stay_t, ne, inv, change_idx] = t
                                                trace_s_arr[t + stay_t, ne, inv, change_idx] = init_loc
                                                trace_inv_arr[t + stay_t, ne, inv, change_idx] = inv
                                                trace_lid_arr[t + stay_t, ne, inv, change_idx] = 0
                                            else:
                                                idx_arr = np.empty(len(dom_idx), dtype=dom_idx._dtype)
                                                for i, v in enumerate(dom_idx):
                                                    idx_arr[i] = v
                                                # idx_arr = np.array(dom_idx)
                                                idx_arr.sort()
                                                idx_arr = idx_arr[::-1]
                                                # first delete
                                                for del_idx in idx_arr:
                                                    if del_idx == label_num_arr[t + stay_t, ne, inv] - 1:
                                                        label_num_arr[t + stay_t, ne, inv] -= 1
                                                    else:
                                                        # exchange del_idx and label_num-1
                                                        total_num = label_num_arr[t + stay_t, ne, inv]
                                                        reward_val_arr[t + stay_t, ne, inv, del_idx] = reward_val_arr[
                                                            t + stay_t, ne, inv, total_num - 1]
                                                        reward_set_arr[t + stay_t, ne, inv, del_idx,
                                                        :] = reward_set_arr[t + stay_t, ne, inv, total_num - 1, :]
                                                        trace_t_arr[t + stay_t, ne, inv, del_idx] = trace_t_arr[
                                                            t + stay_t, ne, inv, total_num - 1]
                                                        trace_s_arr[t + stay_t, ne, inv, del_idx] = trace_s_arr[
                                                            t + stay_t, ne, inv, total_num - 1]
                                                        trace_inv_arr[t + stay_t, ne, inv, del_idx] = trace_inv_arr[
                                                            t + stay_t, ne, inv, total_num - 1]
                                                        trace_lid_arr[t + stay_t, ne, inv, del_idx] = trace_lid_arr[
                                                            t + stay_t, ne, inv, total_num - 1]
                                                        label_num_arr[t + stay_t, ne, inv] -= 1
                                                # then add
                                                cur_label_num = label_num_arr[t + stay_t, ne, inv]
                                                label_num_arr[t + stay_t, ne, inv] += 1
                                                reward_val_arr[t + stay_t, ne, inv, cur_label_num] = tmp_val
                                                reward_set_arr[t + stay_t, ne, inv, cur_label_num, :] = tmp_set
                                                trace_t_arr[t + stay_t, ne, inv, cur_label_num] = t
                                                trace_s_arr[t + stay_t, ne, inv, cur_label_num] = init_loc
                                                trace_inv_arr[t + stay_t, ne, inv, cur_label_num] = inv
                                                trace_lid_arr[t + stay_t, ne, inv, cur_label_num] = 0

                            else:
                                arr_t = round(c_mat[init_loc, ne])
                                if t + arr_t <= t_repo:
                                    calcu_arr[arr_t, ne - 1] = True
                                    for ne_inv in range(inv_num):
                                        ins = inv_dict_arr[inv] - inv_dict_arr[ne_inv]
                                        if (0 <=
                                                ei_s_arr[ne - 1, cur_t, cur_t + arr_t, x_s_arr[ne - 1], x_c_arr[
                                                    ne - 1]] + ins
                                                <= cap_s):

                                            if label_num_arr[arr_t, ne, ne_inv] == 0:
                                                before_val = esd_arr[
                                                    ne - 1,
                                                    cur_t,
                                                    cur_t + arr_t if cur_t + arr_t < 49 else 48,
                                                    x_s_arr[ne - 1],
                                                    x_c_arr[ne - 1]]
                                                after_val = esd_arr[
                                                    ne - 1,
                                                    cur_t + arr_t if cur_t + arr_t < 36 else 35,
                                                    cur_t + t_f if cur_t + t_f < 49 else 48,
                                                    round(ei_s_arr[
                                                              ne - 1,
                                                              cur_t,
                                                              cur_t + arr_t if cur_t + arr_t < 49 else 48,
                                                              x_s_arr[ne - 1],
                                                              x_c_arr[ne - 1]] + ins),
                                                    round(ei_c_arr[
                                                              ne - 1,
                                                              cur_t,
                                                              cur_t + arr_t if cur_t + arr_t < 49 else 48,
                                                              x_s_arr[ne - 1],
                                                              x_c_arr[ne - 1]])
                                                ]
                                                original_val = esd_arr[
                                                    ne - 1,
                                                    cur_t,
                                                    cur_t + t_f if cur_t + t_f < 49 else 48,
                                                    x_s_arr[ne - 1],
                                                    x_c_arr[ne - 1]
                                                ]
                                                computed_ESD = before_val + after_val - original_val

                                                label_num_arr[arr_t, ne, ne_inv] = 1
                                                reward_val_arr[arr_t, ne, ne_inv, 0] = (
                                                        cur_reward +
                                                        ORDER_INCOME_UNIT * computed_ESD -
                                                        alpha * (arr_t - 1) - dual_station_vec[ne - 1])
                                                reward_set_arr[arr_t, ne, ne_inv, 0, init_loc] = True
                                                reward_set_arr[arr_t, ne, ne_inv, 0, ne] = True
                                                trace_t_arr[arr_t, ne, ne_inv, 0] = 0
                                                trace_s_arr[arr_t, ne, ne_inv, 0] = init_loc
                                                trace_inv_arr[arr_t, ne, ne_inv, 0] = inv
                                                trace_lid_arr[arr_t, ne, ne_inv, 0] = 0

                                            else:
                                                before_val = esd_arr[
                                                    ne - 1,
                                                    cur_t,
                                                    cur_t + arr_t if cur_t + arr_t < 49 else 48,
                                                    x_s_arr[ne - 1],
                                                    x_c_arr[ne - 1]]
                                                after_val = esd_arr[
                                                    ne - 1,
                                                    cur_t + arr_t if cur_t + arr_t < 36 else 35,
                                                    cur_t + t_f if cur_t + t_f < 49 else 48,
                                                    round(ei_s_arr[
                                                              ne - 1,
                                                              cur_t,
                                                              cur_t + arr_t if cur_t + arr_t < 49 else 48,
                                                              x_s_arr[ne - 1],
                                                              x_c_arr[ne - 1]] + ins),
                                                    round(ei_c_arr[
                                                              ne - 1,
                                                              cur_t,
                                                              cur_t + arr_t if cur_t + arr_t < 49 else 48,
                                                              x_s_arr[ne - 1],
                                                              x_c_arr[ne - 1]])
                                                ]
                                                original_val = esd_arr[
                                                    ne - 1,
                                                    cur_t,
                                                    cur_t + t_f if cur_t + t_f < 49 else 48,
                                                    x_s_arr[ne - 1],
                                                    x_c_arr[ne - 1]
                                                ]
                                                computed_ESD = before_val + after_val - original_val
                                                new_reward = (cur_reward +
                                                              ORDER_INCOME_UNIT * computed_ESD -
                                                              alpha * (arr_t - 1) - dual_station_vec[ne - 1])
                                                assert label_num_arr[arr_t, ne, ne_inv] == 1
                                                if new_reward > reward_val_arr[arr_t, ne, ne_inv, 0]:
                                                    reward_val_arr[arr_t, ne, ne_inv, 0] = new_reward
                                                    reward_set_arr[arr_t, ne, ne_inv, 0, init_loc] = True
                                                    reward_set_arr[arr_t, ne, ne_inv, 0, ne] = True
                                                    trace_t_arr[arr_t, ne, ne_inv, 0] = 0
                                                    trace_s_arr[arr_t, ne, ne_inv, 0] = init_loc
                                                    trace_inv_arr[arr_t, ne, ne_inv, 0] = inv
                                                    trace_lid_arr[arr_t, ne, ne_inv, 0] = 0
                                                else:
                                                    pass

        else:  # t > 0
            if t == t_repo - 1:
                # stept = time.process_time()
                # print(f't={t}, time: {stept - st}')
                break
            else:
                # stept = time.process_time()
                # print(f't={t}, time: {stept - st}')
                for cur_s in range(num_stations + 1):
                    if cur_s > 0 and calcu_arr[t, cur_s - 1] is False:
                        pass
                    else:
                        for inv in range(inv_num):
                            if label_num_arr[t, cur_s, inv] == 0:
                                pass
                            else:  # select labels to extend
                                for label_id in range(label_num_arr[t, cur_s, inv]):
                                    cur_reward = reward_val_arr[t, cur_s, inv, label_id]
                                    cur_set = reward_set_arr[t, cur_s, inv, label_id, :].copy()

                                    # forward update
                                    for next_s in range(1, num_stations + 1):
                                        # stay at 本站
                                        if next_s == cur_s:
                                            stay_t = 1
                                            if t + stay_t <= t_repo:

                                                if t + stay_t < t_repo - 1 or inv_dict_arr[inv] == 0:

                                                    calcu_arr[t + stay_t, next_s - 1] = True

                                                    if label_num_arr[t + stay_t, next_s, inv] == 0:
                                                        new_reward = cur_reward
                                                        label_num_arr[t + stay_t, next_s, inv] = 1
                                                        reward_val_arr[t + stay_t, next_s, inv, 0] = new_reward
                                                        reward_set_arr[t + stay_t, next_s, inv, 0, :] = cur_set
                                                        trace_t_arr[t + stay_t, next_s, inv, 0] = t
                                                        trace_s_arr[t + stay_t, next_s, inv, 0] = cur_s
                                                        trace_inv_arr[t + stay_t, next_s, inv, 0] = inv
                                                        trace_lid_arr[t + stay_t, next_s, inv, 0] = label_id

                                                    else:  # dominate rules applied
                                                        tmp_val = cur_reward
                                                        tmp_set = reward_set_arr[t, cur_s, inv, label_id, :].copy()
                                                        dom_idx = List()
                                                        for ne_label_id in range(
                                                                label_num_arr[t + stay_t, next_s, inv]):
                                                            ne_val = reward_val_arr[
                                                                t + stay_t, next_s, inv, ne_label_id]
                                                            ne_set = reward_set_arr[t + stay_t, next_s, inv,
                                                                     ne_label_id, :].copy()

                                                            if tmp_val >= ne_val and not np.any(
                                                                    tmp_set > ne_set):  # set1是set2的子集
                                                                dom_idx.append(ne_label_id)
                                                            elif ne_val >= tmp_val and not np.any(ne_set > tmp_set):
                                                                assert len(dom_idx) == 0  # dom_idx is empty
                                                                break
                                                        else:
                                                            if len(dom_idx) == 0:  # no domination
                                                                cur_label_num = label_num_arr[t + stay_t, next_s, inv]
                                                                label_num_arr[t + stay_t, next_s, inv] += 1
                                                                reward_val_arr[
                                                                    t + stay_t, next_s, inv, cur_label_num] = tmp_val
                                                                reward_set_arr[t + stay_t, next_s, inv, cur_label_num,
                                                                :] = tmp_set
                                                                trace_t_arr[t + stay_t, next_s, inv, cur_label_num] = t
                                                                trace_s_arr[
                                                                    t + stay_t, next_s, inv, cur_label_num] = cur_s
                                                                trace_inv_arr[
                                                                    t + stay_t, next_s, inv, cur_label_num] = inv
                                                                trace_lid_arr[
                                                                    t + stay_t, next_s, inv, cur_label_num] = label_id
                                                            elif len(dom_idx) == 1:
                                                                change_idx = dom_idx[0]
                                                                reward_val_arr[
                                                                    t + stay_t, next_s, inv, change_idx] = tmp_val
                                                                reward_set_arr[t + stay_t, next_s, inv, change_idx,
                                                                :] = tmp_set
                                                                trace_t_arr[t + stay_t, next_s, inv, change_idx] = t
                                                                trace_s_arr[t + stay_t, next_s, inv, change_idx] = cur_s
                                                                trace_inv_arr[t + stay_t, next_s, inv, change_idx] = inv
                                                                trace_lid_arr[
                                                                    t + stay_t, next_s, inv, change_idx] = label_id
                                                            else:
                                                                idx_arr = np.empty(len(dom_idx), dtype=dom_idx._dtype)
                                                                for i, v in enumerate(dom_idx):
                                                                    idx_arr[i] = v
                                                                # idx_arr = np.array(dom_idx)
                                                                idx_arr.sort()
                                                                idx_arr = idx_arr[::-1]
                                                                # first delete
                                                                for del_idx in idx_arr:
                                                                    if del_idx == label_num_arr[
                                                                        t + stay_t, next_s, inv] - 1:
                                                                        label_num_arr[t + stay_t, next_s, inv] -= 1
                                                                    else:
                                                                        # exchange del_idx and label_num-1
                                                                        total_num = label_num_arr[
                                                                            t + stay_t, next_s, inv]
                                                                        reward_val_arr[
                                                                            t + stay_t, next_s, inv, del_idx] = \
                                                                            reward_val_arr[
                                                                                t + stay_t, next_s, inv, total_num - 1]
                                                                        reward_set_arr[t + stay_t, next_s, inv, del_idx,
                                                                        :] \
                                                                            = reward_set_arr[t + stay_t, next_s, inv,
                                                                              total_num - 1, :]
                                                                        trace_t_arr[t + stay_t, next_s, inv, del_idx] = \
                                                                            trace_t_arr[
                                                                                t + stay_t, next_s, inv, total_num - 1]
                                                                        trace_s_arr[t + stay_t, next_s, inv, del_idx] = \
                                                                            trace_s_arr[
                                                                                t + stay_t, next_s, inv, total_num - 1]
                                                                        trace_inv_arr[
                                                                            t + stay_t, next_s, inv, del_idx] = \
                                                                            trace_inv_arr[
                                                                                t + stay_t, next_s, inv, total_num - 1]
                                                                        trace_lid_arr[
                                                                            t + stay_t, next_s, inv, del_idx] = \
                                                                            trace_lid_arr[
                                                                                t + stay_t, next_s, inv, total_num - 1]
                                                                        label_num_arr[t + stay_t, next_s, inv] -= 1
                                                                # then add
                                                                cur_label_num = label_num_arr[t + stay_t, next_s, inv]
                                                                label_num_arr[t + stay_t, next_s, inv] += 1
                                                                reward_val_arr[
                                                                    t + stay_t, next_s, inv, cur_label_num] = tmp_val
                                                                reward_set_arr[t + stay_t, next_s, inv, cur_label_num,
                                                                :] = tmp_set
                                                                trace_t_arr[t + stay_t, next_s, inv, cur_label_num] = t
                                                                trace_s_arr[
                                                                    t + stay_t, next_s, inv, cur_label_num] = cur_s
                                                                trace_inv_arr[
                                                                    t + stay_t, next_s, inv, cur_label_num] = inv
                                                                trace_lid_arr[
                                                                    t + stay_t, next_s, inv, cur_label_num] = label_id
                                        elif cur_set[next_s]:
                                            pass
                                        else:
                                            # assert cur_set[next_s] is False, f'{cur_set[next_s]}'
                                            arr_t = round(c_mat[cur_s, next_s])
                                            if t + arr_t <= t_repo:
                                                if t + arr_t < t_repo - 1:
                                                    can_do_inv = inv_num
                                                else:
                                                    can_do_inv = 1
                                                for next_inv in range(can_do_inv):
                                                    # if t == 3 and t + arr_t == 5 and next_s == 5 and next_inv == 5:
                                                    #     logging.info('debug')
                                                    ins = inv_dict_arr[inv] - inv_dict_arr[next_inv]
                                                    if 0 <= ei_s_arr[
                                                        next_s - 1, cur_t, cur_t + t + arr_t, x_s_arr[next_s - 1],
                                                        x_c_arr[next_s - 1]] + ins <= cap_s:
                                                        calcu_arr[t + arr_t, next_s - 1] = True

                                                        before_val = esd_arr[
                                                            next_s - 1,
                                                            cur_t,
                                                            cur_t + t + arr_t if cur_t + t + arr_t < 49 else 48,
                                                            x_s_arr[next_s - 1],
                                                            x_c_arr[next_s - 1]]
                                                        after_val = esd_arr[
                                                            next_s - 1,
                                                            cur_t + t + arr_t if cur_t + t + arr_t < 36 else 35,
                                                            cur_t + t_f if cur_t + t_f < 49 else 48,
                                                            round(ei_s_arr[
                                                                      next_s - 1,
                                                                      cur_t,
                                                                      cur_t + t + arr_t if cur_t + t + arr_t < 49 else 48,
                                                                      x_s_arr[next_s - 1],
                                                                      x_c_arr[next_s - 1]] + ins),
                                                            round(ei_c_arr[
                                                                      next_s - 1,
                                                                      cur_t,
                                                                      cur_t + t + arr_t if cur_t + t + arr_t < 49 else 48,
                                                                      x_s_arr[next_s - 1],
                                                                      x_c_arr[next_s - 1]])
                                                        ]
                                                        original_val = esd_arr[
                                                            next_s - 1,
                                                            cur_t,
                                                            cur_t + t_f if cur_t + t_f < 49 else 48,
                                                            x_s_arr[next_s - 1],
                                                            x_c_arr[next_s - 1]
                                                        ]
                                                        computed_ESD = before_val + after_val - original_val
                                                        new_reward = (cur_reward +
                                                                      ORDER_INCOME_UNIT * computed_ESD -
                                                                      alpha * (arr_t - 1) -
                                                                      dual_station_vec[next_s - 1])

                                                        if label_num_arr[t + arr_t, next_s, next_inv] == 0:

                                                            label_num_arr[t + arr_t, next_s, next_inv] = 1
                                                            reward_val_arr[t + arr_t, next_s, next_inv, 0] = new_reward
                                                            for sta in range(num_stations + 1):
                                                                reward_set_arr[t + arr_t, next_s, next_inv, 0, sta] = \
                                                                    cur_set[sta]
                                                            reward_set_arr[
                                                                t + arr_t, next_s, next_inv, 0, next_s] = True
                                                            trace_t_arr[t + arr_t, next_s, next_inv, 0] = t
                                                            trace_s_arr[t + arr_t, next_s, next_inv, 0] = cur_s
                                                            trace_inv_arr[t + arr_t, next_s, next_inv, 0] = inv
                                                            trace_lid_arr[t + arr_t, next_s, next_inv, 0] = label_id

                                                        else:  # dominate rules applied
                                                            tmp_val = new_reward
                                                            tmp_set = reward_set_arr[t, cur_s, inv, label_id, :].copy()
                                                            tmp_set[next_s] = True

                                                            dom_idx = List()
                                                            for ne_label_id in range(
                                                                    label_num_arr[t + arr_t, next_s, next_inv]):
                                                                ne_val = reward_val_arr[
                                                                    t + arr_t, next_s, next_inv, ne_label_id]
                                                                ne_set = reward_set_arr[t + arr_t, next_s, next_inv,
                                                                         ne_label_id, :].copy()

                                                                if tmp_val >= ne_val and not np.any(
                                                                        tmp_set > ne_set):  # set1是set2的子集
                                                                    dom_idx.append(ne_label_id)
                                                                elif ne_val >= tmp_val and not np.any(ne_set > tmp_set):
                                                                    # if len(dom_idx) > 0:
                                                                    #     logging.debug('dom_idx is not empty')
                                                                    assert len(
                                                                        dom_idx) == 0, f'{len(dom_idx), t + arr_t, next_s, next_inv}'  # dom_idx is empty
                                                                    break
                                                            else:
                                                                if len(dom_idx) == 0:  # no domination
                                                                    cur_label_num = label_num_arr[
                                                                        t + arr_t, next_s, next_inv]
                                                                    label_num_arr[t + arr_t, next_s, next_inv] += 1
                                                                    reward_val_arr[
                                                                        t + arr_t, next_s, next_inv, cur_label_num] = tmp_val
                                                                    reward_set_arr[t + arr_t, next_s, next_inv,
                                                                    cur_label_num, :] = tmp_set
                                                                    trace_t_arr[
                                                                        t + arr_t, next_s, next_inv, cur_label_num] = t
                                                                    trace_s_arr[
                                                                        t + arr_t, next_s, next_inv, cur_label_num] = cur_s
                                                                    trace_inv_arr[
                                                                        t + arr_t, next_s, next_inv, cur_label_num] = inv
                                                                    trace_lid_arr[
                                                                        t + arr_t, next_s, next_inv, cur_label_num] = label_id
                                                                elif len(dom_idx) == 1:
                                                                    change_idx = dom_idx[0]
                                                                    reward_val_arr[
                                                                        t + arr_t, next_s, next_inv, change_idx] = tmp_val
                                                                    reward_set_arr[t + arr_t, next_s, next_inv,
                                                                    change_idx, :] = tmp_set
                                                                    trace_t_arr[
                                                                        t + arr_t, next_s, next_inv, change_idx] = t
                                                                    trace_s_arr[
                                                                        t + arr_t, next_s, next_inv, change_idx] = cur_s
                                                                    trace_inv_arr[
                                                                        t + arr_t, next_s, next_inv, change_idx] = inv
                                                                    trace_lid_arr[
                                                                        t + arr_t, next_s, next_inv, change_idx] = label_id
                                                                else:
                                                                    idx_arr = np.empty(len(dom_idx),
                                                                                       dtype=dom_idx._dtype)
                                                                    for i, v in enumerate(dom_idx):
                                                                        idx_arr[i] = v
                                                                    # idx_arr = np.array(dom_idx)
                                                                    idx_arr.sort()
                                                                    idx_arr = idx_arr[::-1]
                                                                    # first delete
                                                                    for del_idx in idx_arr:
                                                                        if del_idx == label_num_arr[
                                                                            t + arr_t, next_s, next_inv] - 1:
                                                                            label_num_arr[
                                                                                t + arr_t, next_s, next_inv] -= 1
                                                                        else:
                                                                            # exchange del_idx and label_num-1
                                                                            total_num = label_num_arr[
                                                                                t + arr_t, next_s, next_inv]
                                                                            reward_val_arr[
                                                                                t + arr_t, next_s, next_inv, del_idx] = \
                                                                                reward_val_arr[
                                                                                    t + arr_t, next_s, next_inv, total_num - 1]
                                                                            reward_set_arr[t + arr_t, next_s, next_inv,
                                                                            del_idx, :] \
                                                                                = reward_set_arr[t + arr_t, next_s,
                                                                                  next_inv, total_num - 1, :]
                                                                            trace_t_arr[
                                                                                t + arr_t, next_s, next_inv, del_idx] = \
                                                                                trace_t_arr[
                                                                                    t + arr_t, next_s, next_inv, total_num - 1]
                                                                            trace_s_arr[
                                                                                t + arr_t, next_s, next_inv, del_idx] = \
                                                                                trace_s_arr[
                                                                                    t + arr_t, next_s, next_inv, total_num - 1]
                                                                            trace_inv_arr[
                                                                                t + arr_t, next_s, next_inv, del_idx] = \
                                                                                trace_inv_arr[
                                                                                    t + arr_t, next_s, next_inv, total_num - 1]
                                                                            trace_lid_arr[
                                                                                t + arr_t, next_s, next_inv, del_idx] = \
                                                                                trace_lid_arr[
                                                                                    t + arr_t, next_s, next_inv, total_num - 1]
                                                                            label_num_arr[
                                                                                t + arr_t, next_s, next_inv] -= 1
                                                                    # then add
                                                                    cur_label_num = label_num_arr[
                                                                        t + arr_t, next_s, next_inv]
                                                                    label_num_arr[t + arr_t, next_s, next_inv] += 1
                                                                    reward_val_arr[
                                                                        t + arr_t, next_s, next_inv, cur_label_num] = tmp_val
                                                                    reward_set_arr[t + arr_t, next_s, next_inv,
                                                                    cur_label_num, :] = tmp_set
                                                                    trace_t_arr[
                                                                        t + arr_t, next_s, next_inv, cur_label_num] = t
                                                                    trace_s_arr[
                                                                        t + arr_t, next_s, next_inv, cur_label_num] = cur_s
                                                                    trace_inv_arr[
                                                                        t + arr_t, next_s, next_inv, cur_label_num] = inv
                                                                    trace_lid_arr[
                                                                        t + arr_t, next_s, next_inv, cur_label_num] = label_id

    # print(reward_arr[11][17][0], reward_arr[12][17][0])
    # for label in reward_arr[11][17][0]:
    #     if label[1].issubset({3, 19, 12, 16, 8, 18}):
    #         print(label)
    # print('compute_route: ', com.compute_route(r=[3, 19, 12, 16, 8, 18], t_left=0, init_l=13, x_s_arr=x_s_arr, x_c_arr=x_c_arr))

    max_reward = -np.inf
    max_t_repo, max_s, max_inv, max_l_id = -1, -1, -1, -1
    # max_reward_list, max_label_list = [], []

    for s in range(num_stations + 1):
        if s == init_loc:
            pass
        else:
            for inv in range(inv_num):
                if label_num_arr[t_repo, s, inv] > 0:
                    for l_id in range(label_num_arr[t_repo, s, inv]):
                        if reward_val_arr[t_repo, s, inv, l_id] > max_reward:
                            max_reward = reward_val_arr[t_repo, s, inv, l_id]
                            max_t_repo, max_s, max_inv, max_l_id = t_repo, s, inv, l_id
                if label_num_arr[t_repo - 1, s, inv] > 0:
                    for l_id in range(label_num_arr[t_repo - 1, s, inv]):
                        if reward_val_arr[t_repo - 1, s, inv, l_id] > max_reward:
                            max_reward = reward_val_arr[t_repo - 1, s, inv, l_id]
                            max_t_repo, max_s, max_inv, max_l_id = t_repo - 1, s, inv, l_id
    print(max_reward)
    # max_label = max_label_list[max_reward_list.index(max_reward)]
    # print(reward_val_arr[max_label[0]][max_label[1]][max_label[2]][max_label[3]])
    k_t_repo, k_s, k_inv, k_l_id = max_t_repo, max_s, max_inv, max_l_id
    loc_list, inv_list = np.array([-1 for _ in range(t_repo + 1)]), np.array([-1 for _ in range(t_repo + 1)])
    while True:
        if k_t_repo == 0:
            assert False
        else:
            loc_list[k_t_repo] = k_s
            inv_list[k_t_repo] = inv_dict_arr[k_inv]
            k_t_repo, k_s, k_inv, k_l_id = trace_t_arr[k_t_repo][k_s][k_inv][k_l_id], \
                trace_s_arr[k_t_repo][k_s][k_inv][k_l_id], \
                trace_inv_arr[k_t_repo][k_s][k_inv][k_l_id], trace_lid_arr[k_t_repo][k_s][k_inv][k_l_id]

            if k_t_repo == 0:
                loc_list[k_t_repo] = k_s
                inv_list[k_t_repo] = inv_dict_arr[k_inv]
                break
    print(loc_list)
    print(inv_list)

    # delete remaining in route
    clean_route = List()
    for k in loc_list:
        for tmp_k in clean_route:
            if k == tmp_k:
                break
        else:
            if k > -0.5:
                clean_route.append(k)

    clearn_route_arr = np.empty(len(clean_route), dtype=np.int8)
    for i, v in enumerate(clean_route):
        clearn_route_arr[i] = v
    print(clearn_route_arr)
    # clearn_route_arr[-1] = max_reward

    return clearn_route_arr


# def get_dp_reduced_cost_bidirectional(cap_s: int, num_stations: int, init_loc: int, init_t_left: int,
#                                       init_load: int, x_s_arr: list, x_c_arr: list, ei_s_arr: np.ndarray,
#                                       ei_c_arr: np.ndarray, esd_arr: np.ndarray, c_mat: np.ndarray,
#                                       cur_t: int, t_p: int, t_f: int, alpha: float,
#                                       dual_van_vec: list, dual_station_vec: list, inventory_dict: dict = None,
#                                       inventory_id_dict: dict = None):
#     """calculate heuristic or exact reduced cost using bidirectional labeling algorithm"""
#     com = ESDComputer(
#         esd_arr=esd_arr, ei_s_arr=ei_s_arr, ei_c_arr=ei_c_arr, t_cur=cur_t, t_fore=t_f, c_mat=c_mat)
#     cur_t = round(cur_t - RE_START_T / 10)
#     t_repo = t_p if RE_START_T / 10 + cur_t + t_p <= RE_END_T / 10 else round(RE_END_T / 10 - cur_t - RE_START_T / 10)
#     half_way_t = int(t_repo / 2)  # forward to: h(actually h-1?), backward to: h + 2(min travel distance)
#     if t_repo == 1:
#         return [init_loc], com.compute_ESD_in_horizon(station_id=init_loc, t_arr=0, ins=init_load, x_s_arr=x_s_arr,
#                                                       x_c_arr=x_c_arr, mode='multi', delta=True, repo=True)
#     elif t_repo == 0:
#         assert False
#     # decision inventory state
#     inv_dict = inventory_dict if inventory_dict is not None else {0: 0, 1: 5, 2: 10, 3: 15, 4: 20, 5: 25}
#     inv_id_dict = inventory_id_dict if inventory_id_dict is not None else {25: 5, 20: 4, 15: 3, 10: 2, 5: 1, 0: 0}
#     inv_num = len(inv_dict)
#     print(f'in get_dp_reduced_cost_bidirec(), t_repo = {t_repo}')
#     reward_arr = [[[None for _ in range(inv_num)] for __ in range(1 + num_stations)] for ___ in range(t_repo + 1)]
#     trace_arr = [[[None for _ in range(inv_num)] for __ in range(1 + num_stations)] for ___ in range(t_repo + 1)]
#     calcu_arr = [[False for _ in range(num_stations)] for __ in range(t_repo + 1)]
#     st = time.process_time()
#     for t in range(t_repo + 1):
#         if



def get_exact_cost(cap_v: int, cap_s: int, num_stations: int, t_left: list, init_loc: list, init_load: list,
                   x_s_arr: list, x_c_arr: list, ei_s_arr: np.ndarray, ei_c_arr: np.ndarray,
                   esd_arr: np.ndarray, c_mat: np.ndarray, cur_t: int, t_p: int, t_f: int, t_roll: int, alpha: float):
    """calculate exact cost using Gurobi"""
    esd_computer = ESDComputer(
        esd_arr=esd_arr, ei_s_arr=ei_s_arr, ei_c_arr=ei_c_arr, t_cur=cur_t, t_fore=t_f, c_mat=c_mat)
    station_esd_list = [
        esd_computer.compute_ESD_in_horizon(
            station_id=i,
            t_arr=0,
            ins=0,
            x_s_arr=x_s_arr,
            x_c_arr=x_c_arr,
            mode='multi',
            delta=True,
            repo=False
        ) for i in range(1, num_stations + 1)
    ]
    reg_t = round(cur_t - RE_START_T / 10)
    t_repo = t_p if cur_t + t_p <= RE_END_T / 10 else round(RE_END_T / 10 - cur_t)
    num_veh = len(t_left)
    ei_s_arr_ij = np.zeros((num_stations, t_repo + 1))
    esd_arr_ij = np.zeros((num_stations, t_repo + 1))
    esd_arr_ijk = np.zeros((num_stations, t_repo + 1, 2 * cap_v + 1))
    for i in range(1, num_stations + 1):
        for j in range(t_repo + 1):
            ei_s_arr_ij[i - 1, j] = ei_s_arr[
                i - 1,
                reg_t,
                round(reg_t + j) if round(reg_t + j) < 49 else 48,
                x_s_arr[i - 1],
                x_c_arr[i - 1]
            ]
            esd_arr_ij[i - 1, j] = esd_arr[
                i - 1,
                reg_t,
                round(reg_t + j) if round(reg_t + j) < 49 else 48,
                x_s_arr[i - 1],
                x_c_arr[i - 1]
            ]
            for k in range(2 * cap_v + 1):
                est_s_inv = round(ei_s_arr[
                                      i - 1,
                                      reg_t,
                                      round(reg_t + j) if round(reg_t + j) < 49 else 48,
                                      x_s_arr[i - 1],
                                      x_c_arr[i - 1]]) - (k - cap_v)
                if 0 <= est_s_inv <= cap_s:
                    esd_arr_ijk[i - 1, j, k] = esd_arr[
                        i - 1,
                        round(reg_t + j) if round(reg_t + j) < 36 else 35,
                        round(reg_t + t_f) if round(reg_t + t_f) < 49 else 48,
                        est_s_inv,
                        round(ei_c_arr[
                                  i - 1,
                                  reg_t,
                                  round(reg_t + j) if round(reg_t + j) < 49 else 48,
                                  x_s_arr[i - 1],
                                  x_c_arr[i - 1]])
                    ]
                else:
                    esd_arr_ijk[i - 1, j, k] = -10000

    model = Model('Model')
    # Sets
    Start_loc = list(init_loc)  # start location
    Dummy_end = [i for i in range(num_veh)]  # N_-
    assert len(Start_loc) == len(Dummy_end)
    Stations = [i for i in range(1, num_stations + 1)]  # S
    N_stations = [val for val in Stations if val not in Start_loc]  # N_S
    All_nodes = [i for i in range(num_stations + 1)]  # S and {0}(depot)
    Veh = [i for i in range(num_veh)]  # V
    i_length, j_length = len(All_nodes), len(Dummy_end) + num_stations  # |num_stations|个站点+|V|个dummy终点
    t_j = [i for i in range(t_repo + 1)]  # T
    n_k = [i for i in range(-cap_v, cap_v + 1)]  # N

    # const
    cost_mat = np.zeros((i_length, j_length))
    for i in range(i_length):
        for j in range(j_length):
            if i == 0:
                if j < num_stations:
                    cost_mat[i, j] = round(c_mat[0, j + 1])
                else:
                    cost_mat[i, j] = 0
            else:  # i > 0
                if j < num_stations:
                    if i != j + 1:
                        cost_mat[i, j] = round(c_mat[i, j + 1] - 1)
                    else:
                        cost_mat[i, j] = 0
                else:
                    cost_mat[i, j] = 0

    d_mat = np.zeros((i_length, j_length))
    for i in range(i_length):
        for j in range(j_length):
            if j < num_stations:
                d_mat[i, j] = c_mat[i, j + 1]
            else:
                d_mat[i, j] = 0

    # Variables
    x_ijv = model.addVars(i_length, j_length, num_veh, vtype=GRB.BINARY, name='x_ijv')
    q_ijv = model.addVars(i_length, j_length, num_veh, vtype=GRB.INTEGER, name='q_ijv')
    z_i = model.addVars(num_stations, vtype=GRB.BINARY, name='z_i')
    t_i = model.addVars(1 + num_stations, vtype=GRB.INTEGER, name='t_i')
    n_i = model.addVars(num_stations, lb=-GRB.INFINITY, vtype=GRB.INTEGER, name='n_i')

    # Linearization
    p_ij = model.addVars(i_length, t_repo + 1, vtype=GRB.BINARY, name='p_ij')
    r_ijk = model.addVars(i_length, t_repo + 1, 2 * cap_v + 1, vtype=GRB.BINARY, name='r_ijk')

    # Constraints
    # (1)
    for v in Veh:
        expr = LinExpr()
        for j in range(j_length):
            if j != Start_loc[v] - 1:
                expr.addTerms(1, x_ijv[Start_loc[v], j, v])
        model.addConstr(expr == 1, f'constr1_{v}')
    # (2)
    for j in range(len(Dummy_end)):
        expr = LinExpr()
        for v in Veh:
            for i in range(i_length):
                expr.addTerms(1, x_ijv[i, j + num_stations, v])
        model.addConstr(expr == 1, name=f'constr2_{j}')
    # (3)
    for i in N_stations:
        expr = LinExpr()
        for v in Veh:
            for j in range(j_length):
                if i != j + 1:
                    expr.addTerms(1, x_ijv[i, j, v])
        model.addConstr(expr <= 1, name=f'constr3_{i}')
    # (4)  Constr(4) covers Constr(3)
    for i in Stations:
        expr = LinExpr()
        for v in Veh:
            for j in range(j_length):
                if i != j + 1:
                    expr.addTerms(1, x_ijv[i, j, v])
        model.addConstr(expr == z_i[i - 1], name=f'constr4_{i}')
    # (5)
    for i in N_stations:
        for v in Veh:
            expr1, expr2 = LinExpr(), LinExpr()
            for j in All_nodes:
                if i != j:
                    expr1.addTerms(1, x_ijv[j, i - 1, v])
            for j in range(j_length):
                if i != j + 1:
                    expr2.addTerms(1, x_ijv[i, j, v])
            model.addConstr(expr1 - expr2 == 0, name=f'constr5_{i}_{v}')
    # (6)
    for i in Stations:
        for j in range(j_length):
            for v in Veh:
                if i != j + 1:
                    model.addConstr(0 <= q_ijv[i, j, v], name=f'constr6_l_{i}_{j}_{v}')
                    model.addConstr(q_ijv[i, j, v] <= cap_v * x_ijv[i, j, v], name=f'constr6_u_{i}_{j}_{v}')
    # (7a)
    for i in Stations:
        expr1, expr2 = LinExpr(), LinExpr()
        for v in Veh:
            for j in range(j_length):
                if i != j + 1:
                    expr1.addTerms(1, q_ijv[i, j, v])
            for j in Stations:
                if i != j:
                    expr2.addTerms(1, q_ijv[j, i - 1, v])
        model.addConstr(n_i[i - 1] == expr1 - expr2, name=f'constr7a_{i}')
    # (7b)
    for v in Veh:
        if Start_loc[v] != 0:
            expr = LinExpr()
            for j in range(j_length):
                if Start_loc[v] != j + 1:
                    expr.addTerms(1, q_ijv[Start_loc[v], j, v])
            model.addConstr(n_i[Start_loc[v] - 1] == expr - init_load[v], name=f'constr7b_{v}')
    # (7c): set initial load to 0 if at depot
    for v in Veh:
        for j in range(j_length):
            model.addConstr(q_ijv[0, j, v] == 0, name=f'constr7c_{v}_{j}')
    # (8)  neglected: looser than Constr (10)
    # (9)
    for i in Stations:
        expr = LinExpr()
        for t in t_j:
            expr.addTerms(ei_s_arr_ij[i - 1, t], p_ij[i, t])
        model.addConstr(0 <= expr - n_i[i - 1], name=f'constr9_l_{i}')
        model.addConstr(expr - n_i[i - 1] <= cap_s, name=f'constr9_u_{i}')
    # (10)
    for i in Stations:
        for v in Veh:
            model.addConstr(-cap_v <= n_i[i - 1], name=f'constr10_l_{i}_{v}')
            model.addConstr(n_i[i - 1] <= cap_v, name=f'constr10_u_{i}_{v}')
    # (11)
    for i in All_nodes:
        for j in range(j_length - 1):
            if i != j + 1:
                for v in Veh:
                    model.addConstr(
                        (t_i[i] + d_mat[i, j] * x_ijv[i, j, v] - t_repo * (1 - x_ijv[i, j, v]) <= t_i[j + 1]),
                        name=f'constr11_{i}_{j}_{v}')
    # (12)
    for i in Stations:
        model.addConstr(0 <= t_i[i], name=f'constr12_l_{i}')
        model.addConstr(t_i[i] <= z_i[i - 1] * t_repo, name=f'constr12_u_{i}')
    # (13): set initial arriving time as t_left
    for v in Veh:
        model.addConstr(t_i[Start_loc[v]] == t_left[v], name=f'constr13_{v}')
    # (14)&(15): sum of p_ij
    for i in Stations:
        expr1, expr2 = LinExpr(), LinExpr()
        for j in t_j:
            expr1.addTerms(1, p_ij[i, j])
            expr2.addTerms(j, p_ij[i, j])
        model.addConstr(expr1 == 1, name=f'constr14_{i}')
        model.addConstr(expr2 == t_i[i], name=f'constr15_{i}')
    # (16)&(17)&(18): sum of r_ijk
    for i in Stations:
        expr1, expr2, expr3 = LinExpr(), LinExpr(), LinExpr()
        for j in t_j:
            for k in n_k:
                expr1.addTerms(1, r_ijk[i, j, k + cap_v])
                expr2.addTerms(j, r_ijk[i, j, k + cap_v])
                expr3.addTerms(k, r_ijk[i, j, k + cap_v])
        model.addConstr(expr1 == 1, name=f'constr16_{i}')
        model.addConstr(expr2 == t_i[i], name=f'constr17_{i}')
        model.addConstr(expr3 == n_i[i - 1], name=f'constr18_{i}')

    # test case
    # [0, -1, 12, 12, -1, 13, -1]
    # [0, -1, 25, 25, -1, 0, -1]
    # model.addConstr(x_ijv[0, 11, 0] == 1, name=f'test_constr_1')
    # model.addConstr(x_ijv[12, 12, 0] == 1, name=f'test_constr_2')
    # model.addConstr(x_ijv[13, 25, 0] == 1, name=f'test_constr_3')
    # model.addConstr(n_i[11] == 25, name=f'test_constr_4')
    # model.addConstr(n_i[12] == -25, name=f'test_constr_4')

    # objective - order profit
    expr1, expr2, expr3, expr4 = QuadExpr(), QuadExpr(), LinExpr(), LinExpr()
    for i in Stations:
        for j in t_j:
            expr1.addTerms(esd_arr_ij[i - 1, j], p_ij[i, j], z_i[i - 1])
            for k in n_k:
                expr2.addTerms(esd_arr_ijk[i - 1, j, k + cap_v], r_ijk[i, j, k + cap_v], z_i[i - 1])
        expr3.addTerms(station_esd_list[i - 1], z_i[i - 1])
    # objective - route cost
    for v in Veh:
        for i in range(i_length):
            for j in range(j_length):
                expr4.addTerms(cost_mat[i, j], x_ijv[i, j, v])
    # objective
    model.setObjective(ORDER_INCOME_UNIT * (expr1 + expr2 - expr3) - alpha * expr4, GRB.MAXIMIZE)

    model.optimize()
    # model.computeIIS()
    # model.write("model_file.ilp")
    if model.status == GRB.OPTIMAL:
        print(f'station ESD sum: {sum(station_esd_list)}')
        print(f'i={0}, t_i[i]={t_i[0].x}')
        for i in range(num_stations):
            print(f'i={i}, z_i[i]={z_i[i].x}')
            print(f'i={i}, t_i[i]={t_i[i + 1].x}')
            print(f'i={i}, n_i[i]={n_i[i].x}')
        for j in range(j_length):
            print(f'j={j}, x_0j0[j]={x_ijv[0, j, 0].x}')

    return model, None, None


def get_CG_REA_routes(num_of_van: int, van_location: list, van_dis_left: list, van_load: list, c_s: int, c_v: int,
                      cur_t: int, t_p: int, t_f: int, t_roll: int, c_mat: np.ndarray, ei_s_arr: np.ndarray,
                      ei_c_arr: np.ndarray, esd_arr: np.ndarray, x_s_arr: list, x_c_arr: list, alpha: float,
                      est_ins: int, branch: int) -> dict:
    num_stations = c_mat.shape[0] - 1  # exclude the depot
    # calculate station_esd_list
    esd_computer = ESDComputer(
        esd_arr=esd_arr, ei_s_arr=ei_s_arr, ei_c_arr=ei_c_arr, t_cur=cur_t, t_fore=t_f, c_mat=c_mat)
    station_esd_list = [
        ORDER_INCOME_UNIT * esd_computer.compute_ESD_in_horizon(
            station_id=i,
            t_arr=0,
            ins=0,
            x_s_arr=x_s_arr,
            x_c_arr=x_c_arr,
            mode='multi',
            delta=True,
            repo=False
        ) for i in range(1, num_stations + 1)
    ]

    # generate initial routes
    st = time.process_time()
    init_routes, init_profit = get_REA_routes_test(
        num_of_van=num_of_van,
        van_location=van_location,
        van_dis_left=van_dis_left,
        van_load=van_load,
        c_s=c_s,
        c_v=c_v,
        cur_t=cur_t,
        t_p=t_p,
        t_f=t_f,
        t_roll=t_roll,
        c_mat=c_mat,
        ei_s_arr=ei_s_arr,
        ei_c_arr=ei_c_arr,
        esd_arr=esd_arr,
        x_s_arr=x_s_arr,
        x_c_arr=x_c_arr,
        alpha=alpha,
        est_ins=est_ins,
        branch=branch,
        state='init'
    )
    ed = time.process_time()
    print(f'init routes time: {ed - st}')
    routes_pool, profit_pool = [list(val) for val in init_routes], list(init_profit)
    negative_flag = True  # any route with negative reduced cost
    rmp, veh_constr, node_constr, dual_vector = init_LP_relaxed_RMP(
        num_stations=num_stations,
        route_pool=routes_pool,
        profit_pool=profit_pool,
        veh_mat=np.ones((num_of_van, len(routes_pool))),
        node_mat=get_node_mat(num_stations=num_stations, route_pool=routes_pool),
        station_esd_list=station_esd_list
    )
    # split the dual vector
    dual_van_vec, dual_station_vec = dual_vector[:num_of_van], dual_vector[num_of_van:]
    while negative_flag:
        # solve LP-relaxed RMP
        # add new columns using dual-REA (heuristic pricing)
        # st = time.process_time()
        # print(f'initial load: {van_load[0]}')
        # new_test_route, new_test_profit = get_dp_reduced_cost(
        #     cap_v=c_v,
        #     cap_s=c_s,
        #     num_stations=25,
        #     init_loc=van_location[0],
        #     init_load=van_load[0],
        #     x_s_arr=x_s_arr,
        #     x_c_arr=x_c_arr,
        #     ei_s_arr=ei_s_arr,
        #     ei_c_arr=ei_c_arr,
        #     esd_arr=esd_arr,
        #     c_mat=c_mat,
        #     cur_t=cur_t,
        #     t_p=t_p,
        #     t_f=t_f,
        #     t_roll=t_roll,
        #     alpha=alpha,
        #     dual_van_vec=[0],
        #     dual_station_vec=[0 for _ in range(25)],
        # )
        # # new_test_route = [0, 21, 3, 12, 13, 8, 18]
        # t_repo = t_p if cur_t + t_p <= RE_END_T / 10 else round(RE_END_T / 10 - cur_t)
        # max_reward, loc_list, inv_list = esd_computer.compute_route(r=new_test_route, t_left=van_dis_left[0],
        #                                                             init_l=van_load[0], x_s_arr=x_s_arr,
        #                                                             x_c_arr=x_c_arr, t_repo=t_repo, can_stay=True)
        # print(f'outside print: {max_reward}')
        # print(f'outside print: {loc_list}')
        # print(f'outside print: {inv_list}')
        # # new_routes, new_profit = get_REA_routes_test(
        # #     num_of_van=num_of_van,
        # #     van_location=van_location,
        # #     van_dis_left=van_dis_left,
        # #     van_load=van_load,
        # #     c_s=c_s,
        # #     c_v=c_v,
        # #     cur_t=cur_t,
        # #     t_p=t_p,
        # #     t_f=t_f,
        # #     t_roll=t_roll,
        # #     c_mat=c_mat,
        # #     ei_s_arr=ei_s_arr,
        # #     ei_c_arr=ei_c_arr,
        # #     esd_arr=esd_arr,
        # #     x_s_arr=x_s_arr,
        # #     x_c_arr=x_c_arr,
        # #     alpha=alpha,
        # #     est_ins=est_ins,
        # #     dual_van_vector=dual_van_vec,
        # #     dual_station_vector=dual_station_vec,
        # #     branch=branch,
        # #     state='dual'
        # # )
        # ed = time.process_time()
        # print(f'new routes time: {ed - st}')
        new_routes, new_profit = None, None
        if new_routes:
            negative_flag = True
            # update the master problem
            ex_routes_num = len(routes_pool)
            for k in range(len(new_routes)):
                col = Column()
                # add vehicle constr
                for v in range(num_of_van):
                    col.addTerms(1, veh_constr[v])
                # add node constr
                visit_arr = get_node_mat(num_stations=num_stations, route_pool=[new_routes[k]])
                for i in range(num_stations):
                    col.addTerms(visit_arr[i], node_constr[i])
                # add variable
                rmp.addVar(obj=new_profit[k], vtype=GRB.BINARY, name=f'x{ex_routes_num + k}', column=col)
                rmp.update()
                # update the route pool
                routes_pool.append(new_routes[k])
                profit_pool.append(new_profit[k])
            # solve RMP with new columns
            relax_RMP = rmp.relax()
            relax_RMP.setParam('OutputFlag', 0)
            relax_RMP.optimize()
            # get dual vector
            dual_vector = [con.Pi for con in relax_RMP.getConstrs()]
            dual_van_vec, dual_station_vec = dual_vector[:num_of_van], dual_vector[num_of_van:]
        else:
            negative_flag = False

    # solve RMP using columns generated
    if routes_pool:
        best_route, best_obj = solve_RMP(model=rmp, routes_pool=routes_pool)
        _, best_ins = esd_computer.compute_route(
            r=best_route, t_left=van_dis_left[0], init_l=van_load[0], x_s_arr=x_s_arr, x_c_arr=x_c_arr)
    else:
        best_route, best_obj, best_ins = [van_location[0]], 0, [0]
    # print(f'best route: {best_route}, best ins: {best_ins}')
    # generate future decision dict
    step_loc_list, step_n_list, step_exp_inv_list, step_target_inv_list, step, cumu_step, s_ind = \
        ([0 for _ in range(t_p)], [0 for _ in range(t_p)], [0 for _ in range(t_p)], [0 for _ in range(t_p)],
         0, van_dis_left[0], 0)
    van_dis_flag = False
    while step < t_p:
        if step == cumu_step:
            step_loc_list[int(step)] = best_route[s_ind]
            step_n_list[int(step)] = best_ins[s_ind]
            if best_route[s_ind] > 0:
                step_exp_inv_list[int(step)] = ei_s_arr[
                    best_route[s_ind] - 1,
                    round(cur_t - RE_START_T / 10),
                    round(cur_t - RE_START_T / 10 + step),
                    round(x_s_arr[best_route[s_ind] - 1]),
                    round(x_c_arr[best_route[s_ind] - 1])
                ]
                step_target_inv_list[int(step)] = \
                    round(step_exp_inv_list[int(step)]) + best_ins[s_ind]
            else:
                step_exp_inv_list[int(step)] = 0
                step_target_inv_list[int(step)] = 0
            if s_ind < len(best_route) - 1:
                cumu_step += c_mat[best_route[s_ind], best_route[s_ind + 1]]
            else:
                cumu_step += t_p
            if cumu_step >= t_p and van_dis_flag is False:
                van_dis_flag = True
            else:
                s_ind += 1
            step += 1
        else:
            step_loc_list[int(step)], step_n_list[int(step)], step_exp_inv_list[int(step)], step_target_inv_list[
                int(step)] = \
                None, None, None, None
            step += 1

    return {
        'objective': best_obj,
        'start_time': cur_t,
        'routes': [best_route],
        'exp_inv': [step_exp_inv_list],
        'exp_target_inv': [step_target_inv_list],
        'loc': [step_loc_list],
        'n_r': [step_n_list],
    }


def get_DP_routes_greedy(num_of_van: int, van_location: list, van_dis_left: list, van_load: list, c_s: int, c_v: int,
                         cur_t: int, t_p: int, t_f: int, t_roll: int, c_mat: np.ndarray, ei_s_arr: np.ndarray,
                         ei_c_arr: np.ndarray, esd_arr: np.ndarray, x_s_arr: list, x_c_arr: list, alpha: float
                         ) -> dict:
    """used as greedy method in multi-vehicle case"""
    num_stations = c_mat.shape[0] - 1  # exclude the depot
    # calculate station_esd_list
    esd_computer = ESDComputer(
        esd_arr=esd_arr, ei_s_arr=ei_s_arr, ei_c_arr=ei_c_arr, t_cur=cur_t, t_fore=t_f, c_mat=c_mat)
    station_esd_list = [
        ORDER_INCOME_UNIT * esd_computer.compute_ESD_in_horizon(
            station_id=i,
            t_arr=0,
            ins=0,
            x_s_arr=x_s_arr,
            x_c_arr=x_c_arr,
            mode='multi',
            delta=True,
            repo=False
        ) for i in range(1, num_stations + 1)
    ]
    # generate routes (only once for each vehicle)
    st = time.process_time()
    visited_stations = []
    aj_loc_list, aj_inv_list = [], []
    total_max_reward = 0
    print(f'van dis left: {van_dis_left}')
    for veh in range(num_of_van):
        dual_station_vec = [0 for _ in range(num_stations)]
        for i in range(num_of_van):
            if i != veh:
                dual_station_vec[van_location[i] - 1] = 1000  # avoid visiting the same station
        for node in visited_stations:
            dual_station_vec[node - 1] = 1000  # avoid visiting the same station
        route, profit = get_dp_reduced_cost_forward(
            cap_v=c_v,
            cap_s=c_s,
            num_stations=num_stations,
            init_loc=van_location[veh],
            init_t_left=van_dis_left[veh],
            init_load=van_load[veh],
            x_s_arr=x_s_arr,
            x_c_arr=x_c_arr,
            ei_s_arr=ei_s_arr,
            ei_c_arr=ei_c_arr,
            esd_arr=esd_arr,
            c_mat=c_mat,
            cur_t=cur_t,
            t_p=t_p,
            t_f=t_f,
            t_roll=t_roll,
            alpha=alpha,
            dual_van_vec=[0],
            dual_station_vec=dual_station_vec,
        )
        for i in route:
            if i != van_location[veh]:
                visited_stations.append(i)
        t_repo = t_p if cur_t + t_p <= RE_END_T / 10 else round(RE_END_T / 10 - cur_t)
        max_reward, loc_list, inv_list = esd_computer.compute_route(r=route, t_left=van_dis_left[veh],
                                                                    init_l=van_load[veh], x_s_arr=x_s_arr,
                                                                    x_c_arr=x_c_arr, t_repo=t_repo, can_stay=True)
        total_max_reward += max_reward
        aj_loc_list.append(loc_list)
        aj_inv_list.append(inv_list)
    ed = time.process_time()
    print(f'route calculation time: {ed - st}')

    re_clean_routes, re_step_exp_inv_list, re_step_target_inv_list = [], [], []
    re_step_loc_list, re_step_n_list = [], []

    for veh in range(num_of_van):
        best_route = aj_loc_list[veh]
        best_inv = aj_inv_list[veh]

        clean_best_route = []
        for k in best_route:
            if k not in clean_best_route and k > -0.5:
                clean_best_route.append(k)

        step_loc_list, step_n_list, step_exp_inv_list, step_target_inv_list, step, cumu_step, s_ind = \
            ([0 for _ in range(t_p + 1)], [0 for _ in range(t_p + 1)], [0 for _ in range(t_p + 1)],
             [0 for _ in range(t_p + 1)],
             0, van_dis_left[veh], 0)
        step_load = van_load[veh]
        while step <= t_p:
            if step == cumu_step:
                step_loc_list[step] = best_route[step]
                if step > van_dis_left[veh] and best_route[step] == best_route[step - 1]:  # stay
                    step_n_list[step] = -100
                else:
                    step_n_list[step] = step_load - best_inv[step]
                step_load = best_inv[step]
                if best_route[step] > 0:
                    step_exp_inv_list[step] = ei_s_arr[
                        best_route[step] - 1,
                        round(cur_t - RE_START_T / 10),
                        round(cur_t - RE_START_T / 10 + step),
                        round(x_s_arr[best_route[step] - 1]),
                        round(x_c_arr[best_route[step] - 1])
                    ]
                    step_target_inv_list[step] = round(step_exp_inv_list[step]) + step_n_list[step]
                else:
                    step_exp_inv_list[step] = 0
                    step_target_inv_list[step] = 0
                if step < len(best_route) - 1:
                    if best_route[step + 1] == best_route[step]:  # stay
                        cumu_step += 1
                    else:
                        assert best_route[step + 1] == -1 or best_route[step] == 0
                        visit_next = clean_best_route[clean_best_route.index(best_route[step]) + 1]
                        cumu_step += c_mat[best_route[step], visit_next]
                else:
                    cumu_step += t_p
                step += 1
            else:
                step_loc_list[step], step_n_list[step], step_exp_inv_list[step], step_target_inv_list[step] = \
                    None, None, None, None
                step += 1

        re_clean_routes.append(clean_best_route)
        re_step_loc_list.append(step_loc_list)
        re_step_n_list.append(step_n_list)
        re_step_exp_inv_list.append(step_exp_inv_list)
        re_step_target_inv_list.append(step_target_inv_list)

    return {
        'objective': total_max_reward + ORDER_INCOME_UNIT * sum(station_esd_list),
        'start_time': cur_t,
        'routes': re_clean_routes,
        'exp_inv': re_step_exp_inv_list,
        'exp_target_inv': re_step_target_inv_list,
        'loc': re_step_loc_list,
        'n_r': re_step_n_list,
    }


def get_DP_routes(num_of_van: int, van_location: list, van_dis_left: list, van_load: list, c_s: int, c_v: int,
                  cur_t: int, t_p: int, t_f: int, t_roll: int, c_mat: np.ndarray, ei_s_arr: np.ndarray,
                  ei_c_arr: np.ndarray, esd_arr: np.ndarray, x_s_arr: list, x_c_arr: list, alpha: float
                  ) -> dict:
    """deposit"""
    num_stations = c_mat.shape[0] - 1  # exclude the depot
    # calculate station_esd_list
    esd_computer = ESDComputer(
        esd_arr=esd_arr, ei_s_arr=ei_s_arr, ei_c_arr=ei_c_arr, t_cur=cur_t, t_fore=t_f, c_mat=c_mat)
    station_esd_list = [
        ORDER_INCOME_UNIT * esd_computer.compute_ESD_in_horizon(
            station_id=i,
            t_arr=0,
            ins=0,
            x_s_arr=x_s_arr,
            x_c_arr=x_c_arr,
            mode='multi',
            delta=True,
            repo=False
        ) for i in range(1, num_stations + 1)
    ]

    # generate initial routes
    st = time.process_time()
    init_route, init_profit = get_dp_reduced_cost_forward(
        cap_v=c_v,
        cap_s=c_s,
        num_stations=25,
        init_loc=van_location[0],
        init_load=van_load[0],
        x_s_arr=x_s_arr,
        x_c_arr=x_c_arr,
        ei_s_arr=ei_s_arr,
        ei_c_arr=ei_c_arr,
        esd_arr=esd_arr,
        c_mat=c_mat,
        cur_t=cur_t,
        t_p=t_p,
        t_f=t_f,
        t_roll=t_roll,
        alpha=alpha,
        dual_van_vec=[0],
        dual_station_vec=[0 for _ in range(25)],
    )
    # init_route = get_dp_reduced_cost_numba(
    #     cap_v=round(c_v),
    #     cap_s=round(c_s),
    #     num_stations=25,
    #     init_loc=round(van_location[0]),
    #     init_load=round(van_load[0]),
    #     x_s_arr=np.array(x_s_arr, dtype=np.int32),
    #     x_c_arr=np.array(x_c_arr, dtype=np.int32),
    #     ei_s_arr=ei_s_arr,
    #     ei_c_arr=ei_c_arr,
    #     esd_arr=esd_arr,
    #     c_mat=c_mat,
    #     cur_t=cur_t,
    #     t_p=t_p,
    #     t_f=t_f,
    #     alpha=alpha,
    #     dual_van_vec=np.array([0], dtype=np.int32),
    #     dual_station_vec=np.array([0 for _ in range(25)], dtype=np.int32),
    # )
    t_repo = t_p if cur_t + t_p <= RE_END_T / 10 else round(RE_END_T / 10 - cur_t)
    max_reward, loc_list, inv_list = esd_computer.compute_route(r=init_route, t_left=van_dis_left[0],
                                                                init_l=van_load[0], x_s_arr=x_s_arr,
                                                                x_c_arr=x_c_arr, t_repo=t_repo, can_stay=True)
    # init_routes, init_profit = get_REA_routes_test(
    #     num_of_van=num_of_van,
    #     van_location=van_location,
    #     van_dis_left=van_dis_left,
    #     van_load=van_load,
    #     c_s=c_s,
    #     c_v=c_v,
    #     cur_t=cur_t,
    #     t_p=t_p,
    #     t_f=t_f,
    #     t_roll=t_roll,
    #     c_mat=c_mat,
    #     ei_s_arr=ei_s_arr,
    #     ei_c_arr=ei_c_arr,
    #     esd_arr=esd_arr,
    #     x_s_arr=x_s_arr,
    #     x_c_arr=x_c_arr,
    #     alpha=alpha,
    #     est_ins=est_ins,
    #     branch=branch,
    #     state='init'
    # )
    ed = time.process_time()
    print(f'init routes time: {ed - st}')
    routes_pool, profit_pool = [init_route], [max_reward]
    negative_flag = False  # any route with negative reduced cost
    # rmp, veh_constr, node_constr, dual_vector = init_LP_relaxed_RMP(
    #     num_stations=num_stations,
    #     route_pool=routes_pool,
    #     profit_pool=profit_pool,
    #     veh_mat=np.ones((num_of_van, len(routes_pool))),
    #     node_mat=get_node_mat(num_stations=num_stations, route_pool=routes_pool),
    #     station_esd_list=station_esd_list
    # )
    # split the dual vector
    # dual_van_vec, dual_station_vec = dual_vector[:num_of_van], dual_vector[num_of_van:]
    while negative_flag:
        # solve LP-relaxed RMP
        # add new columns using dual-REA (heuristic pricing)
        # st = time.process_time()
        # print(f'initial load: {van_load[0]}')
        # new_test_route, new_test_profit = get_dp_reduced_cost(
        #     cap_v=c_v,
        #     cap_s=c_s,
        #     num_stations=25,
        #     init_loc=van_location[0],
        #     init_load=van_load[0],
        #     x_s_arr=x_s_arr,
        #     x_c_arr=x_c_arr,
        #     ei_s_arr=ei_s_arr,
        #     ei_c_arr=ei_c_arr,
        #     esd_arr=esd_arr,
        #     c_mat=c_mat,
        #     cur_t=cur_t,
        #     t_p=t_p,
        #     t_f=t_f,
        #     t_roll=t_roll,
        #     alpha=alpha,
        #     dual_van_vec=[0],
        #     dual_station_vec=[0 for _ in range(25)],
        # )
        # # new_test_route = [0, 21, 3, 12, 13, 8, 18]
        # t_repo = t_p if cur_t + t_p <= RE_END_T / 10 else round(RE_END_T / 10 - cur_t)
        # max_reward, loc_list, inv_list = esd_computer.compute_route(r=new_test_route, t_left=van_dis_left[0],
        #                                                             init_l=van_load[0], x_s_arr=x_s_arr,
        #                                                             x_c_arr=x_c_arr, t_repo=t_repo, can_stay=True)
        # print(f'outside print: {max_reward}')
        # print(f'outside print: {loc_list}')
        # print(f'outside print: {inv_list}')
        # # new_routes, new_profit = get_REA_routes_test(
        # #     num_of_van=num_of_van,
        # #     van_location=van_location,
        # #     van_dis_left=van_dis_left,
        # #     van_load=van_load,
        # #     c_s=c_s,
        # #     c_v=c_v,
        # #     cur_t=cur_t,
        # #     t_p=t_p,
        # #     t_f=t_f,
        # #     t_roll=t_roll,
        # #     c_mat=c_mat,
        # #     ei_s_arr=ei_s_arr,
        # #     ei_c_arr=ei_c_arr,
        # #     esd_arr=esd_arr,
        # #     x_s_arr=x_s_arr,
        # #     x_c_arr=x_c_arr,
        # #     alpha=alpha,
        # #     est_ins=est_ins,
        # #     dual_van_vector=dual_van_vec,
        # #     dual_station_vector=dual_station_vec,
        # #     branch=branch,
        # #     state='dual'
        # # )
        # ed = time.process_time()
        # print(f'new routes time: {ed - st}')
        new_routes, new_profit = None, None
        if new_routes:
            negative_flag = True
            # update the master problem
            ex_routes_num = len(routes_pool)
            for k in range(len(new_routes)):
                col = Column()
                # add vehicle constr
                for v in range(num_of_van):
                    col.addTerms(1, veh_constr[v])
                # add node constr
                visit_arr = get_node_mat(num_stations=num_stations, route_pool=[new_routes[k]])
                for i in range(num_stations):
                    col.addTerms(visit_arr[i], node_constr[i])
                # add variable
                rmp.addVar(obj=new_profit[k], vtype=GRB.BINARY, name=f'x{ex_routes_num + k}', column=col)
                rmp.update()
                # update the route pool
                routes_pool.append(new_routes[k])
                profit_pool.append(new_profit[k])
            # solve RMP with new columns
            relax_RMP = rmp.relax()
            relax_RMP.setParam('OutputFlag', 0)
            relax_RMP.optimize()
            # get dual vector
            dual_vector = [con.Pi for con in relax_RMP.getConstrs()]
            dual_van_vec, dual_station_vec = dual_vector[:num_of_van], dual_vector[num_of_van:]
        else:
            negative_flag = False

    # solve RMP using columns generated
    # if routes_pool:
    #     best_route, best_obj = solve_RMP(model=rmp, routes_pool=routes_pool)
    #     _, best_ins = esd_computer.compute_route(
    #         r=best_route, t_left=van_dis_left[0], init_l=van_load[0], x_s_arr=x_s_arr, x_c_arr=x_c_arr)
    # else:
    #     best_route, best_obj, best_ins = [van_location[0]], 0, [0]
    best_route, best_obj, best_inv = loc_list, max_reward + ORDER_INCOME_UNIT * sum(station_esd_list), inv_list
    print(f'best route: {best_route}, best inv: {best_inv}')
    # generate future decision dict

    clean_best_route = []
    for k in best_route:
        if k not in clean_best_route and k > -0.5:
            clean_best_route.append(k)

    step_loc_list, step_n_list, step_exp_inv_list, step_target_inv_list, step, cumu_step, s_ind = \
        ([0 for _ in range(t_p + 1)], [0 for _ in range(t_p + 1)], [0 for _ in range(t_p + 1)],
         [0 for _ in range(t_p + 1)],
         0, van_dis_left[0], 0)
    step_load = van_load[0]
    while step <= t_p:
        if step == cumu_step:
            step_loc_list[step] = best_route[step]
            if step > 0 and best_route[step] == best_route[step - 1]:  # stay
                step_n_list[step] = -100
            else:
                step_n_list[step] = step_load - best_inv[step]
            step_load = best_inv[step]
            if best_route[step] > 0:
                step_exp_inv_list[step] = ei_s_arr[
                    best_route[step] - 1,
                    round(cur_t - RE_START_T / 10),
                    round(cur_t - RE_START_T / 10 + step),
                    round(x_s_arr[best_route[step] - 1]),
                    round(x_c_arr[best_route[step] - 1])
                ]
                step_target_inv_list[step] = round(step_exp_inv_list[step]) + step_n_list[step]
            else:
                step_exp_inv_list[step] = 0
                step_target_inv_list[step] = 0
            if step < len(best_route) - 1:
                if best_route[step + 1] == best_route[step]:  # stay
                    cumu_step += 1
                else:
                    assert best_route[step + 1] == -1 or best_route[step] == 0
                    visit_next = clean_best_route[clean_best_route.index(best_route[step]) + 1]
                    cumu_step += c_mat[best_route[step], visit_next]
            else:
                cumu_step += t_p
            step += 1
        else:
            step_loc_list[step], step_n_list[step], step_exp_inv_list[step], step_target_inv_list[step] = \
                None, None, None, None
            step += 1
    return {
        'objective': best_obj,
        'start_time': cur_t,
        'routes': [clean_best_route],
        'exp_inv': [step_exp_inv_list],
        'exp_target_inv': [step_target_inv_list],
        'loc': [step_loc_list],
        'n_r': [step_n_list],
    }


def get_exact_routes(van_location: list, van_dis_left: list, van_load: list, c_s: int, c_v: int,
                     cur_t: int, t_p: int, t_f: int, t_roll: int, c_mat: np.ndarray, ei_s_arr: np.ndarray,
                     ei_c_arr: np.ndarray, esd_arr: np.ndarray, x_s_arr: list, x_c_arr: list, alpha: float
                     ) -> dict:
    num_stations = c_mat.shape[0] - 1  # exclude the depot
    # calculate station_esd_list
    esd_computer = ESDComputer(
        esd_arr=esd_arr, ei_s_arr=ei_s_arr, ei_c_arr=ei_c_arr, t_cur=cur_t, t_fore=t_f, c_mat=c_mat)
    station_esd_list = [
        ORDER_INCOME_UNIT * esd_computer.compute_ESD_in_horizon(
            station_id=i,
            t_arr=0,
            ins=0,
            x_s_arr=x_s_arr,
            x_c_arr=x_c_arr,
            mode='multi',
            delta=True,
            repo=False
        ) for i in range(1, num_stations + 1)
    ]

    # generate initial routes
    st = time.process_time()
    # compare with the DP version
    init_route, init_profit = get_dp_reduced_cost_forward(
        cap_v=c_v,
        cap_s=c_s,
        num_stations=25,
        init_loc=van_location[0],
        init_load=van_load[0],
        x_s_arr=x_s_arr,
        x_c_arr=x_c_arr,
        ei_s_arr=ei_s_arr,
        ei_c_arr=ei_c_arr,
        esd_arr=esd_arr,
        c_mat=c_mat,
        cur_t=cur_t,
        t_p=t_p,
        t_f=t_f,
        t_roll=t_roll,
        alpha=alpha,
        dual_van_vec=[0],
        dual_station_vec=[0 for _ in range(25)],
    )
    t_repo = t_p if cur_t + t_p <= RE_END_T / 10 else round(RE_END_T / 10 - cur_t)
    max_reward, loc_list, inv_list = esd_computer.compute_route(r=init_route, t_left=van_dis_left[0],
                                                                init_l=van_load[0], x_s_arr=x_s_arr,
                                                                x_c_arr=x_c_arr, t_repo=t_repo, can_stay=True)
    max_reward, loc_list, inv_list = get_exact_cost(
        cap_v=c_v,
        cap_s=c_s,
        num_stations=25,
        t_left=van_dis_left,
        init_loc=van_location,
        init_load=van_load,
        x_s_arr=x_s_arr,
        x_c_arr=x_c_arr,
        ei_s_arr=ei_s_arr,
        ei_c_arr=ei_c_arr,
        esd_arr=esd_arr,
        c_mat=c_mat,
        cur_t=cur_t,
        t_p=t_p,
        t_f=t_f,
        t_roll=t_roll,
        alpha=alpha,
    )
    return {'model': max_reward}
    t_repo = t_p if cur_t + t_p <= RE_END_T / 10 else round(RE_END_T / 10 - cur_t)
    ed = time.process_time()
    print(f'exact routes and cost calculation time: {ed - st}')

    # route: [loc every time step], [inv every time step]
    best_route, best_obj, best_inv = loc_list, max_reward + ORDER_INCOME_UNIT * sum(station_esd_list), inv_list
    print(f'best route: {best_route}, best inv: {best_inv}')

    # generate future decision dict
    clean_best_route = []
    for k in best_route:
        if k not in clean_best_route and k > -0.5:
            clean_best_route.append(k)

    step_loc_list, step_n_list, step_exp_inv_list, step_target_inv_list, step, cumu_step, s_ind = \
        ([0 for _ in range(t_p + 1)], [0 for _ in range(t_p + 1)], [0 for _ in range(t_p + 1)],
         [0 for _ in range(t_p + 1)],
         0, van_dis_left[0], 0)
    step_load = van_load[0]
    while step <= t_p:
        if step == cumu_step:
            step_loc_list[step] = best_route[step]
            if step > 0 and best_route[step] == best_route[step - 1]:  # stay
                step_n_list[step] = -100
            else:
                step_n_list[step] = step_load - best_inv[step]
            step_load = best_inv[step]
            if best_route[step] > 0:
                step_exp_inv_list[step] = ei_s_arr[
                    best_route[step] - 1,
                    round(cur_t - RE_START_T / 10),
                    round(cur_t - RE_START_T / 10 + step),
                    round(x_s_arr[best_route[step] - 1]),
                    round(x_c_arr[best_route[step] - 1])
                ]
                step_target_inv_list[step] = round(step_exp_inv_list[step]) + step_n_list[step]
            else:
                step_exp_inv_list[step] = 0
                step_target_inv_list[step] = 0
            if step < len(best_route) - 1:
                if best_route[step + 1] == best_route[step]:  # stay
                    cumu_step += 1
                else:
                    assert best_route[step + 1] == -1 or best_route[step] == 0
                    visit_next = clean_best_route[clean_best_route.index(best_route[step]) + 1]
                    cumu_step += c_mat[best_route[step], visit_next]
            else:
                cumu_step += t_p
            step += 1
        else:
            step_loc_list[step], step_n_list[step], step_exp_inv_list[step], step_target_inv_list[step] = \
                None, None, None, None
            step += 1
    return {
        'objective': best_obj,
        'start_time': cur_t,
        'routes': [clean_best_route],
        'exp_inv': [step_exp_inv_list],
        'exp_target_inv': [step_target_inv_list],
        'loc': [step_loc_list],
        'n_r': [step_n_list],
    }
