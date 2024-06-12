import time
import random
import numpy as np
import gurobipy as gp

from route_extension.bph.utils import Node, Stack
from route_extension.bph.models import MasterProblem, HeuristicProblem
from route_extension.route_extension_algo import ESDComputer
from route_extension.cg_re_algo import get_dp_reduced_cost_bidirectional_numba, get_dp_reduced_cost_bidirectional
from simulation.consts import NODE_LIMIT, RE_START_T, RE_END_T, CG_STOP_EPSILON, ORDER_INCOME_UNIT, SEED, CAP_C

random.seed(SEED)


def to_stop(pool_length: int, lb: float, ub: float, num_nodes: int, gap: float = 1e-6):
    """stop condition for branch and price"""
    if pool_length == 0:
        return True
    else:
        if num_nodes >= NODE_LIMIT:
            return True
        else:
            if ub - lb < abs(gap * ub):
                return True
            else:
                return False


def is_integer_sol(sol: list):
    """check if the solution is integer"""
    return all(float(round(s, 4)).is_integer() for s in sol)


def is_integer(number: float):
    return float(round(number, 4)).is_integer()


def branch_and_price(c_s: int, c_v: int, cur_t: int, t_p: int, t_f: int, t_roll: int, c_mat: np.ndarray,
                     ei_s_arr: np.ndarray, ei_c_arr: np.ndarray, esd_arr: np.ndarray, computer: ESDComputer,
                     alpha: float, mode: str, master_prob: MasterProblem):
    """main process of branch and price"""

    # Step1: Initialization
    stack = Stack()
    global_upper_bound = np.inf
    global_lower_bound = -np.inf
    num_of_van = master_prob.num_veh
    global_cg_column_pool, global_bp_column_pool = [[] for _ in range(num_of_van)], [[] for _ in range(num_of_van)]
    global_cg_profit_pool, global_bp_profit_pool = [[] for _ in range(num_of_van)], [[] for _ in range(num_of_van)]
    num_explored_nodes = 0

    # ----------------- Step2: Solve root node
    # ----------------- Step2.1: Update UB
    root_st = time.process_time()
    num_explored_nodes += 1  # explore root node
    solve_relax_problem(
        computer=computer, num_of_van=master_prob.num_veh, van_location=master_prob.van_location,
        van_dis_left=master_prob.van_dis_left, van_load=master_prob.van_load, c_s=c_s, c_v=c_v, cur_t=cur_t,
        t_p=t_p, t_f=t_f, t_roll=t_roll, c_mat=c_mat, ei_s_arr=ei_s_arr, ei_c_arr=ei_c_arr, esd_arr=esd_arr,
        x_s_arr=master_prob.x_s_arr, x_c_arr=master_prob.x_c_arr, mode=mode, alpha=alpha, problem=master_prob
    )
    root_ed = time.process_time()
    print(f"Root node running time: {root_ed - root_st} seconds")
    root_relax_sol = master_prob.get_relax_solution()
    # print('relax_route_vars: ', root_relax_sol)
    # print('relax_station_vars: ', master_prob.get_relax_station_vars())
    if is_integer_sol(sol=root_relax_sol):
        return master_prob
    else:
        root_lp_val = master_prob.relax_model.objVal
        global_upper_bound = root_lp_val
        print(f'Root global upper bound: {global_upper_bound}')
        root_node = Node(node_id='1', lp_obj=root_lp_val, mp=master_prob)
        stack.push(root_node)
        # ----------------- Step2.2: Update LB
        master_prob.integer_optimize()
        global_lower_bound = master_prob.model.objVal
        print(f'Updated global lower bound: {global_lower_bound}')
        non_zero_routes_dict = master_prob.get_non_zero_routes(model='both')
        for veh in range(len(non_zero_routes_dict['route'])):
            veh_route = non_zero_routes_dict['route'][veh]
            for route in veh_route:
                global_cg_column_pool[veh].append(list(route))
            for profit in non_zero_routes_dict['profit'][veh]:
                global_cg_profit_pool[veh].append(profit)
        # print(f'global_cg_column_pool: ', global_cg_column_pool)
        # print(f'global_cg_profit_pool: ', global_cg_profit_pool)
        # ----------------- Step4: Branch and price
        while not to_stop(pool_length=len(stack), lb=global_lower_bound,
                          ub=global_upper_bound, num_nodes=num_explored_nodes):
            # ------------- Step3: Node exploration
            # ------------- Step3.1: Select incumbent node
            cur_node = stack.pop()
            # ------------- Step3.2: Node branching
            left_push_flag, right_push_flag = False, False
            left_node, right_node = branch(node=cur_node)
            # ------------- Step3.3: Solve child node
            left_prob = left_node.mp
            num_explored_nodes += 1  # explore left node
            solve_relax_problem(computer=computer, num_of_van=left_prob.num_veh, van_location=left_prob.van_location,
                                van_dis_left=left_prob.van_dis_left, van_load=left_prob.van_load, c_s=c_s, c_v=c_v,
                                cur_t=cur_t, t_p=t_p, t_f=t_f, t_roll=t_roll, c_mat=c_mat, ei_s_arr=ei_s_arr,
                                ei_c_arr=ei_c_arr, esd_arr=esd_arr, x_s_arr=left_prob.x_s_arr,
                                x_c_arr=left_prob.x_c_arr, alpha=alpha, mode=mode, problem=left_prob)
            if left_prob.relax_model.status != gp.GRB.Status.INFEASIBLE:
                left_node.lp_obj = left_prob.relax_model.objVal
                left_lp_sol = left_prob.get_relax_solution()
                non_zero_routes_dict = left_prob.get_non_zero_routes(model='relax')
                for veh in range(len(non_zero_routes_dict['route'])):
                    veh_route = non_zero_routes_dict['route'][veh]
                    for route in veh_route:
                        global_bp_column_pool[veh].append(list(route))
                    for profit in non_zero_routes_dict['profit'][veh]:
                        global_bp_profit_pool[veh].append(profit)
                # ------------- Step3.4: Check integrality and update UB
                if is_integer_sol(sol=left_lp_sol):
                    global_lower_bound = max(global_lower_bound, left_prob.relax_model.objVal)
                    print(f'Updated global lower bound: {global_lower_bound}')
                else:
                    if left_prob.relax_model.objVal > global_lower_bound:
                        # ----- Step3.5: Update LB by enhancement tech
                        # print(global_cg_column_pool)
                        # print(global_cg_profit_pool)
                        heuristic_prob = HeuristicProblem(num_of_van=left_prob.num_veh, route_pool=left_prob.route_pool,
                                                          profit_pool=left_prob.profit_pool, veh_mat=left_prob.veh_mat,
                                                          node_mat=left_prob.node_mat, model=left_prob.model,
                                                          cg_column_pool=global_cg_column_pool,
                                                          cg_profit_pool=global_cg_profit_pool)
                        # print(heuristic_prob.model.getConstrs())
                        heuristic_lower_bound = heuristic_prob.solve()
                        global_lower_bound = max(global_lower_bound, heuristic_lower_bound)
                        heuristic_non_zero_routes = heuristic_prob.obtain_non_zero_routes()
                        for veh in range(len(heuristic_non_zero_routes['route'])):
                            veh_route = heuristic_non_zero_routes['route'][veh]
                            for route in veh_route:
                                global_bp_column_pool[veh].append(list(route))
                            for profit in heuristic_non_zero_routes['profit'][veh]:
                                global_bp_profit_pool[veh].append(profit)
                        print(f'Updated global lower bound: {global_lower_bound}')
                        left_push_flag = True
                    else:
                        if to_stop(pool_length=len(stack), lb=global_lower_bound,
                                   ub=global_upper_bound, num_nodes=num_explored_nodes):
                            break  # todo: may need to adjust
            else:  # model is infeasible
                pass
            right_prob = right_node.mp
            num_explored_nodes += 1
            solve_relax_problem(computer=computer, num_of_van=num_of_van, van_location=right_prob.van_location,
                                van_dis_left=right_prob.van_dis_left, van_load=right_prob.van_load, c_s=c_s, c_v=c_v,
                                cur_t=cur_t, t_p=t_p, t_f=t_f, t_roll=t_roll, c_mat=c_mat, ei_s_arr=ei_s_arr,
                                ei_c_arr=ei_c_arr, esd_arr=esd_arr, x_s_arr=right_prob.x_s_arr,
                                x_c_arr=right_prob.x_c_arr, mode=mode, alpha=alpha, problem=right_prob)
            if right_prob.relax_model.status != gp.GRB.Status.INFEASIBLE:
                right_node.lp_obj = right_prob.relax_model.objVal
                right_lp_sol = right_prob.get_relax_solution()
                non_zero_routes_dict = right_prob.get_non_zero_routes(model='relax')
                for veh in range(len(non_zero_routes_dict['route'])):
                    veh_route = non_zero_routes_dict['route'][veh]
                    for route in veh_route:
                        global_bp_column_pool[veh].append(list(route))
                    for profit in non_zero_routes_dict['profit'][veh]:
                        global_bp_profit_pool[veh].append(profit)
                if is_integer_sol(sol=right_lp_sol):
                    global_lower_bound = max(global_lower_bound, right_prob.relax_model.objVal)
                    print(f'Updated global lower bound: {global_lower_bound}')
                else:
                    if right_prob.relax_model.objVal > global_lower_bound:
                        heuristic_prob = HeuristicProblem(num_of_van=right_prob.num_veh, route_pool=right_prob.route_pool,
                                                          profit_pool=right_prob.profit_pool, veh_mat=right_prob.veh_mat,
                                                          node_mat=right_prob.node_mat, model=right_prob.model,
                                                          cg_column_pool=global_cg_column_pool,
                                                          cg_profit_pool=global_cg_profit_pool)
                        heuristic_lower_bound = heuristic_prob.solve()
                        global_lower_bound = max(global_lower_bound, heuristic_lower_bound)
                        heuristic_non_zero_routes = heuristic_prob.obtain_non_zero_routes()
                        for veh in range(len(heuristic_non_zero_routes['route'])):
                            veh_route = heuristic_non_zero_routes['route'][veh]
                            for route in veh_route:
                                global_bp_column_pool[veh].append(list(route))
                            for profit in heuristic_non_zero_routes['profit'][veh]:
                                global_bp_profit_pool[veh].append(profit)
                        print(f'Updated global lower bound: {global_lower_bound}')
                        right_push_flag = True
                    else:
                        if to_stop(pool_length=len(stack), lb=global_lower_bound,
                                   ub=global_upper_bound, num_nodes=num_explored_nodes):
                            break  # todo: may need to adjust
            else:
                pass
            if right_push_flag:
                stack.push(right_node)
            if left_push_flag:
                stack.push(left_node)

        master_prob.add_columns(column_pool=global_bp_column_pool, column_profit=global_bp_profit_pool)

        return master_prob


def solve_relax_problem(computer: ESDComputer, num_of_van: int, van_location: list, van_dis_left: list, van_load: list,
                        c_s: int, c_v: int, cur_t: int, t_p: int, t_f: int, t_roll: int, c_mat: np.ndarray,
                        ei_s_arr: np.ndarray, ei_c_arr: np.ndarray, esd_arr: np.ndarray, x_s_arr: list,
                        x_c_arr: list, alpha: float, mode: str, problem: MasterProblem) -> None:
    """solve the problem using column generation"""
    problem.relax_optimize()
    if problem.relax_model.status == gp.GRB.Status.OPTIMAL:
        print(f"Relax model objVal: {problem.relax_model.objVal}, "
              f"delta: {problem.relax_model.objVal - ORDER_INCOME_UNIT * problem.no_repo_esd}")
        # sol = problem.get_relax_solution()
        # dual = problem.get_dual_vector()
        node_reach_optimal = False
        cg_times = 1
        while not node_reach_optimal:
            st = time.process_time()
            cg_column_pool, cg_profit_pool, early_stop_flag = column_generation(
                computer=computer, num_of_van=num_of_van, van_location=van_location, van_dis_left=van_dis_left,
                van_load=van_load, c_s=c_s, c_v=c_v, cur_t=cur_t, t_p=t_p, t_f=t_f, t_roll=t_roll, c_mat=c_mat,
                ei_s_arr=ei_s_arr, ei_c_arr=ei_c_arr, esd_arr=esd_arr, x_s_arr=x_s_arr, x_c_arr=x_c_arr, alpha=alpha,
                mode=mode, problem=problem)
            ed = time.process_time()
            print(f'the {cg_times}th column generation time: {ed - st} seconds')
            cg_times += 1
            if early_stop_flag is False and cg_column_pool:
                problem.add_columns(column_pool=cg_column_pool, column_profit=cg_profit_pool)
                problem.relax_optimize()
                print(f"Relax model objVal: {problem.relax_model.objVal}, "
                      f"delta: {problem.relax_model.objVal - ORDER_INCOME_UNIT * problem.no_repo_esd}")
            else:
                node_reach_optimal = True
    else:
        print('Model is infeasible.')


def column_generation(computer: ESDComputer, num_of_van: int, van_location: list, van_dis_left: list, van_load: list,
                      c_s: int, c_v: int, cur_t: int, t_p: int, t_f: int, t_roll: int, c_mat: np.ndarray,
                      ei_s_arr: np.ndarray, ei_c_arr: np.ndarray, esd_arr: np.ndarray, x_s_arr: list,
                      x_c_arr: list, alpha: float, mode: str, problem: MasterProblem) -> tuple:
    """column generation process, return generated columns and early stop flag"""
    num_stations = c_mat.shape[0] - 1  # exclude the depot
    t_repo = t_p if cur_t + t_p <= RE_END_T / 10 else round(RE_END_T / 10 - cur_t)
    lp_obj = problem.relax_model.objVal
    dual_vector, veh_num_labels, visit_con_keys = problem.get_dual_vector()
    dual_van_vec, dual_station_vec = dual_vector[:num_of_van], dual_vector[num_of_van:num_stations + num_of_van]
    # adjust with new dual vector (in case there's any new constraints)
    for idx in range(len(veh_num_labels)):
        if veh_num_labels[idx]:
            dual_van_vec = [val - dual_vector[num_stations + num_of_van + idx] for val in dual_van_vec]
        else:
            dual_van_vec = [val + dual_vector[num_stations + num_of_van + idx] for val in dual_van_vec]
    for idx in range(len(visit_con_keys)):
        dual_station_vec[visit_con_keys[idx]] -= dual_vector[num_stations + num_of_van + len(veh_num_labels) + idx]
    if all(x == van_location[0] for x in van_location) and \
            all(x == van_dis_left[0] for x in van_dis_left) and \
            all(x == van_load[0] for x in van_load):
        same_state = True
    else:
        same_state = False
    if same_state:
        default_inv_id_arr = np.array([0, 0, 0, 0, 0,
                                       1, 1, 1, 1, 1,
                                       2, 2, 2, 2, 2,
                                       3, 3, 3, 3, 3,
                                       4, 4, 4, 4, 4,
                                       5, 5, 5, 5, 5], dtype=np.int8)
        default_inv_arr = np.array([0, 5, 10, 15, 20, 25], dtype=np.int8)
        if mode == 'multi':
            com_ei_s_arr, com_esd_arr = ei_s_arr, esd_arr
        else:
            assert mode == 'single'
            new_ei_shape, new_esd_shape = (*ei_s_arr.shape, CAP_C + 1), (*esd_arr.shape, CAP_C + 1)
            com_ei_s_arr = np.broadcast_to(np.expand_dims(ei_s_arr, axis=-1), shape=new_ei_shape)
            com_ei_s_arr = np.ascontiguousarray(com_ei_s_arr)
            com_esd_arr = np.broadcast_to(np.expand_dims(esd_arr, axis=-1), shape=new_esd_shape)
            com_esd_arr = np.ascontiguousarray(com_esd_arr)
        new_route = get_dp_reduced_cost_bidirectional_numba(
            cap_s=c_s, num_stations=num_stations, init_loc=van_location[0], init_t_left=van_dis_left[0],
            init_load=van_load[0], x_s_arr=np.array(x_s_arr, dtype=np.int32), x_c_arr=np.array(x_c_arr, dtype=np.int32),
            ei_s_arr=com_ei_s_arr, ei_c_arr=ei_c_arr, esd_arr=com_esd_arr, c_mat=c_mat, cur_t=cur_t, t_p=t_p,
            t_f=t_f, alpha=alpha, dual_station_vec=np.array(dual_station_vec, dtype=np.float64),
            inventory_dict=default_inv_arr, inventory_id_dict=default_inv_id_arr
        )
        route = list(new_route)
        max_reward, loc_list, inv_list = computer.compute_route(r=route, t_left=van_dis_left[0],
                                                                init_l=van_load[0], x_s_arr=x_s_arr, x_c_arr=x_c_arr,
                                                                mode=mode, t_repo=t_repo, can_stay=True)
        clean_route = []
        for k in loc_list:
            if k not in clean_route and k > -0.5:
                clean_route.append(k)
        # assert len(clean_route) == len(route), f'{clean_route}, {route}'
        station_reduced_cost = sum([dual_station_vec[j-1] for j in route if j > 0])
        # print(f'minimum reduced cost: {- max_reward + min(dual_van_vec) + station_reduced_cost}')
        if max_reward - min(dual_van_vec) - station_reduced_cost > 1e-5:  # found route with negative reduced cost
            route_pool, profit_pool = [[] for _ in range(num_of_van)], [[] for _ in range(num_of_van)]
            for van in range(num_of_van):
                route_pool[van].append(route)
                profit_pool[van].append(max_reward)
            return route_pool, profit_pool, False
        else:  # use exact algorithm to find the route
            default_inv_id_arr = np.array(list(range(c_v + 1)), dtype=np.int8)
            default_inv_arr = np.array(list(range(c_v + 1)), dtype=np.int8)
            if mode == 'multi':
                com_ei_s_arr, com_esd_arr = ei_s_arr, esd_arr
            else:
                assert mode == 'single'
                new_ei_shape, new_esd_shape = (*ei_s_arr.shape, CAP_C + 1), (*esd_arr.shape, CAP_C + 1)
                com_ei_s_arr = np.broadcast_to(np.expand_dims(ei_s_arr, axis=-1), shape=new_ei_shape)
                com_ei_s_arr = np.ascontiguousarray(com_ei_s_arr)
                com_esd_arr = np.broadcast_to(np.expand_dims(esd_arr, axis=-1), shape=new_esd_shape)
                com_esd_arr = np.ascontiguousarray(com_esd_arr)
            new_route = get_dp_reduced_cost_bidirectional_numba(
                cap_s=c_s, num_stations=num_stations, init_loc=van_location[0], init_t_left=van_dis_left[0],
                init_load=van_load[0], x_s_arr=np.array(x_s_arr, dtype=np.int32),
                x_c_arr=np.array(x_c_arr, dtype=np.int32),
                ei_s_arr=com_ei_s_arr, ei_c_arr=ei_c_arr, esd_arr=com_esd_arr, c_mat=c_mat, cur_t=cur_t, t_p=t_p,
                t_f=t_f, alpha=alpha, dual_station_vec=np.array(dual_station_vec, dtype=np.float64),
                inventory_dict=default_inv_arr, inventory_id_dict=default_inv_id_arr
            )
            route = list(new_route)
            max_reward, loc_list, inv_list = computer.compute_route(r=route, t_left=van_dis_left[0],
                                                                    init_l=van_load[0], x_s_arr=x_s_arr, x_c_arr=x_c_arr,
                                                                    mode=mode, t_repo=t_repo, can_stay=True,
                                                                    to_print=False)
            clean_route = []
            for k in loc_list:
                if k not in clean_route and k > -0.5:
                    clean_route.append(k)
            assert len(clean_route) == len(route), f'{clean_route}, {route}'
            station_reduced_cost = sum([dual_station_vec[j - 1] for j in route if j > 0])
            # print(f'minimum reduced cost: {- max_reward + min(dual_van_vec) + station_reduced_cost}')
            if max_reward - min(dual_van_vec) - station_reduced_cost < lp_obj * CG_STOP_EPSILON / num_of_van:
                early_stop_flag = True
                return None, None, early_stop_flag
            else:
                # print(f'minimum reduced cost: {- max_reward + min(dual_van_vec) + station_reduced_cost}')
                if max_reward - min(
                        dual_van_vec) - station_reduced_cost > 1e-5:  # found route with negative reduced cost
                    route_pool, profit_pool = [[] for _ in range(num_of_van)], [[] for _ in range(num_of_van)]
                    for van in range(num_of_van):
                        route_pool[van].append(route)
                        profit_pool[van].append(max_reward)
                    return route_pool, profit_pool, False
                else:  # cannot find route with negative reduced cost
                    return None, None, True

    else:  # different states
        get_new_route = False
        early_stop_flag = [False for _ in range(num_of_van)]
        route_pool, profit_pool = [[] for _ in range(num_of_van)], [[] for _ in range(num_of_van)]

        for van in range(num_of_van):
            default_inv_id_arr = np.array([0, 0, 0, 0, 0,
                                           1, 1, 1, 1, 1,
                                           2, 2, 2, 2, 2,
                                           3, 3, 3, 3, 3,
                                           4, 4, 4, 4, 4,
                                           5], dtype=np.int8)
            default_inv_arr = np.array([0, 5, 10, 15, 20, 25], dtype=np.int8)
            if mode == 'multi':
                com_ei_s_arr, com_esd_arr = ei_s_arr, esd_arr
            else:
                assert mode == 'single'
                new_ei_shape, new_esd_shape = (*ei_s_arr.shape, CAP_C + 1), (*esd_arr.shape, CAP_C + 1)
                com_ei_s_arr = np.broadcast_to(np.expand_dims(ei_s_arr, axis=-1), shape=new_ei_shape)
                com_ei_s_arr = np.ascontiguousarray(com_ei_s_arr)
                com_esd_arr = np.broadcast_to(np.expand_dims(esd_arr, axis=-1), shape=new_esd_shape)
                com_esd_arr = np.ascontiguousarray(com_esd_arr)
            new_route = get_dp_reduced_cost_bidirectional_numba(
                cap_s=c_s, num_stations=num_stations, init_loc=van_location[van], init_t_left=van_dis_left[van],
                init_load=van_load[van], x_s_arr=np.array(x_s_arr, dtype=np.int32),
                x_c_arr=np.array(x_c_arr, dtype=np.int32),
                ei_s_arr=com_ei_s_arr, ei_c_arr=ei_c_arr, esd_arr=com_esd_arr, c_mat=c_mat, cur_t=cur_t, t_p=t_p,
                t_f=t_f, alpha=alpha, dual_station_vec=np.array(dual_station_vec, dtype=np.float64),
                inventory_dict=default_inv_arr, inventory_id_dict=default_inv_id_arr
            )
            route = list(new_route)
            # if route == [18, 13, 6, 19, 3]:
            #     print(f'route: {route}')
            #     test_route = get_dp_reduced_cost_bidirectional(
            #         cap_s=c_s, num_stations=num_stations, init_loc=van_location[van], init_t_left=van_dis_left[van],
            #         init_load=van_load[van], x_s_arr=x_s_arr, x_c_arr=x_c_arr, ei_s_arr=com_ei_s_arr, ei_c_arr=ei_c_arr,
            #         esd_arr=com_esd_arr, c_mat=c_mat, cur_t=cur_t, t_p=t_p, t_f=t_f, alpha=alpha, dual_van=0,
            #         dual_station_vec=list(dual_station_vec), inventory_dict={
            #             0: 0, 1: 5, 2: 10, 3: 15, 4: 20, 5: 25
            #         },
            #         inventory_id_dict={
            #             25: 5, 20: 4, 15: 3, 10: 2, 5: 1, 0: 0
            #         }
            #     )
            max_reward, loc_list, inv_list = computer.compute_route(r=route, t_left=van_dis_left[van],
                                                                    init_l=van_load[van], x_s_arr=x_s_arr,
                                                                    x_c_arr=x_c_arr, mode=mode, t_repo=t_repo,
                                                                    can_stay=True, to_print=False)
            clean_route = []
            for k in loc_list:
                if k not in clean_route and k > -0.5:
                    clean_route.append(k)
            # assert len(clean_route) == len(route), f'{clean_route}, {route}'
            station_reduced_cost = sum([dual_station_vec[j - 1] for j in route if j > 0])
            # print(f'minimum reduced cost: {- max_reward + dual_van_vec[van] + station_reduced_cost}')
            if max_reward - dual_van_vec[van] - station_reduced_cost > 1e-5:  # found route with negative reduced cost
                route_pool[van].append(route)
                profit_pool[van].append(max_reward)
                get_new_route = True
            else:
                pass
        if not get_new_route:  # cannot find route with negative reduced cost, use exact algorithm
            print(f'finding new routes with exact algo...')
            for van in range(num_of_van):
                default_inv_id_arr = np.array(list(range(c_v + 1)), dtype=np.int8)
                default_inv_arr = np.array(list(range(c_v + 1)), dtype=np.int8)
                if mode == 'multi':
                    com_ei_s_arr, com_esd_arr = ei_s_arr, esd_arr
                else:
                    assert mode == 'single'
                    new_ei_shape, new_esd_shape = (*ei_s_arr.shape, CAP_C + 1), (*esd_arr.shape, CAP_C + 1)
                    com_ei_s_arr = np.broadcast_to(np.expand_dims(ei_s_arr, axis=-1), shape=new_ei_shape)
                    com_ei_s_arr = np.ascontiguousarray(com_ei_s_arr)
                    com_esd_arr = np.broadcast_to(np.expand_dims(esd_arr, axis=-1), shape=new_esd_shape)
                    com_esd_arr = np.ascontiguousarray(com_esd_arr)
                new_route = get_dp_reduced_cost_bidirectional_numba(
                    cap_s=c_s, num_stations=num_stations, init_loc=van_location[van], init_t_left=van_dis_left[van],
                    init_load=van_load[van], x_s_arr=np.array(x_s_arr, dtype=np.int32),
                    x_c_arr=np.array(x_c_arr, dtype=np.int32),
                    ei_s_arr=com_ei_s_arr, ei_c_arr=ei_c_arr, esd_arr=com_esd_arr, c_mat=c_mat, cur_t=cur_t, t_p=t_p,
                    t_f=t_f, alpha=alpha, dual_station_vec=np.array(dual_station_vec, dtype=np.float64),
                    inventory_dict=default_inv_arr, inventory_id_dict=default_inv_id_arr
                )
                route = list(new_route)
                max_reward, loc_list, inv_list = computer.compute_route(r=route, t_left=van_dis_left[van],
                                                                        init_l=van_load[van], x_s_arr=x_s_arr,
                                                                        x_c_arr=x_c_arr, mode=mode, t_repo=t_repo,
                                                                        can_stay=True, to_print=False)
                clean_route = []
                for k in loc_list:
                    if k not in clean_route and k > -0.5:
                        clean_route.append(k)
                assert len(clean_route) == len(route), f'{clean_route}, {route}'
                station_reduced_cost = sum([dual_station_vec[j - 1] for j in route if j > 0])
                # print(f'minimum reduced cost: {- max_reward + dual_van_vec[van] + station_reduced_cost}')
                if max_reward - dual_van_vec[van] - station_reduced_cost < lp_obj * CG_STOP_EPSILON / num_of_van:
                    early_stop_flag[van] = True
                if max_reward - dual_van_vec[van] - station_reduced_cost > 1e-5:
                    # found route with negative reduced cost
                    route_pool[van].append(route)
                    profit_pool[van].append(max_reward)
                    get_new_route = True
                else:  # cannot find route with negative reduced cost
                    pass
            if all(early_stop_flag):
                return None, None, True
            else:
                if get_new_route:
                    return route_pool, profit_pool, False
                else:
                    return None, None, True
        else:
            return route_pool, profit_pool, False


def branch(node: Node):
    """branch the node"""
    print(f'Branching, branching node: {node.node_id}')
    problem = node.mp
    node_id = node.node_id
    # 1. branch on number of vehicles
    sum_veh_vars = sum(problem.get_relax_solution())
    if not is_integer(sum_veh_vars):  # not is_integer(sum_veh_vars)
        thd_lb = int(sum_veh_vars)
        thd_ub = thd_lb + 1
        left_node = branch_on_vehicle(node_id=node_id, problem=problem, threshold=thd_ub, branch_on=1)
        right_node = branch_on_vehicle(node_id=node_id, problem=problem, threshold=thd_lb, branch_on=0)
        return left_node, right_node
    else:
        # 2. branch on whether to visit station i
        station_vars = problem.get_relax_station_vars()
        if not is_integer_sol(sol=station_vars):
            non_int_stations = [i for i in range(len(station_vars)) if not is_integer(station_vars[i])]
            # random shuffle and sort by distance to 0.5
            random.shuffle(non_int_stations)
            non_int_stations = sorted(non_int_stations, key=lambda x: abs(station_vars[x] - 0.5))
            station_to_branch = non_int_stations[0] + 1
            left_node = branch_on_station(node_id=node_id, problem=problem, station=station_to_branch, branch_on=1)
            right_node = branch_on_station(node_id=node_id, problem=problem, station=station_to_branch, branch_on=0)
            return left_node, right_node
        else:
            relax_sol = problem.get_relax_solution()
            if not is_integer_sol(sol=relax_sol):
                assert False, f'need to branch on routes: {relax_sol}'
            else:
                pass


def branch_on_station(node_id: str, problem: MasterProblem, station: int, branch_on: int) -> Node:
    """branch on whether to visit station i"""
    if branch_on == 1:
        new_problem = problem.__deepcopy__()
        # add constraints
        new_problem.add_station_visit_constr(station=station, visit=1)
        # add belongings
        new_problem.must_visit_station.append(station)
        new_node = Node(node_id=f'{node_id}@0', lp_obj=0, mp=new_problem)
        return new_node
    elif branch_on == 0:
        new_problem = problem.__deepcopy__()
        # add constraints
        new_problem.add_station_visit_constr(station=station, visit=0)
        # add belongings
        new_problem.must_not_visit_station.append(station)
        new_node = Node(node_id=f'{node_id}@1', lp_obj=0, mp=new_problem)
        return new_node
    else:
        assert False, f'infeasible branch_on: {branch_on}'


def branch_on_vehicle(node_id: str, problem: MasterProblem, threshold: int, branch_on: int) -> Node:
    """
    branch on number of vehicles

    :param node_id:
    :param problem:
    :param threshold: given threshold
    :param branch_on: 1 means be greater than, 0 means be less than
    :return:
    """
    if branch_on == 1:
        new_problem = problem.__deepcopy__()
        # add constraints
        new_problem.add_vehicle_num_constr(threshold=threshold, greater=True)
        # create node
        new_node = Node(node_id=f'{node_id}@0', lp_obj=0, mp=new_problem)
        return new_node
    elif branch_on == 0:
        new_problem = problem.__deepcopy__()
        # add constraints
        new_problem.add_vehicle_num_constr(threshold=threshold, greater=False)
        # create node
        new_node = Node(node_id=f'{node_id}@1', lp_obj=0, mp=new_problem)
        return new_node
