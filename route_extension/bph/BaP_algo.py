import time
import numpy as np

from route_extension.bph.utils import GreedySolution
from route_extension.bph.models import MasterProblem
from route_extension.bph.BaP import branch_and_price
from simulation.consts import RE_START_T, RE_END_T, ORDER_INCOME_UNIT
from route_extension.route_extension_algo import ESDComputer


def get_routes_branch_and_price(num_of_van: int, van_location: list, van_dis_left: list, van_load: list, c_s: int,
                                c_v: int, cur_t: int, t_p: int, t_f: int, t_roll: int, c_mat: np.ndarray,
                                ei_s_arr: np.ndarray, ei_c_arr: np.ndarray, esd_arr: np.ndarray, x_s_arr: list,
                                x_c_arr: list, alpha: float, est_ins: list, mode: str) -> dict:
    """use branch and price to get the routes"""
    num_stations = c_mat.shape[0] - 1  # exclude the depot
    t_repo = t_p if cur_t + t_p <= RE_END_T / 10 else round(RE_END_T / 10 - cur_t)
    esd_computer = ESDComputer(
        esd_arr=esd_arr, ei_s_arr=ei_s_arr, ei_c_arr=ei_c_arr, t_cur=cur_t, t_fore=t_f, c_mat=c_mat)
    station_esd_list = [
        esd_computer.compute_ESD_in_horizon(
            station_id=i,
            t_arr=0,
            ins=0,
            x_s_arr=x_s_arr,
            x_c_arr=x_c_arr,
            mode=mode,
            delta=True,
            repo=False
        ) for i in range(1, num_stations + 1)
    ]
    st = time.process_time()
    # ------------------ generate routes (only once for each vehicle) --------------------------------
    greedy_problem = GreedySolution(num_of_van=num_of_van, van_location=van_location, van_dis_left=van_dis_left,
                                    van_load=van_load, c_s=c_s, c_v=c_v, cur_t=cur_t, t_p=t_p, t_f=t_f, t_roll=t_roll,
                                    c_mat=c_mat, ei_s_arr=ei_s_arr, ei_c_arr=ei_c_arr, esd_arr=esd_arr, x_s_arr=x_s_arr,
                                    x_c_arr=x_c_arr, alpha=alpha, est_ins=est_ins, mode=mode)
    init_routes, init_profits = greedy_problem.generate_init_solution(computer=esd_computer)
    # ---------------------------- construct master problem ------------------------------------------
    mp = MasterProblem(num_veh=num_of_van, num_stations=num_stations, van_location=van_location,
                       van_dis_left=van_dis_left, van_load=van_load, x_s_arr=x_s_arr, x_c_arr=x_c_arr,
                       init_routes=init_routes, init_profits=init_profits, esd_list=station_esd_list)
    mp.build_model()
    # -------------------------------- run branch and price ------------------------------------------
    result_mp = branch_and_price(c_s=c_s, c_v=c_v, cur_t=cur_t, t_p=t_p, t_f=t_f, t_roll=t_roll, c_mat=c_mat,
                                 ei_s_arr=ei_s_arr, ei_c_arr=ei_c_arr, esd_arr=esd_arr, computer=esd_computer,
                                 alpha=alpha, mode=mode, master_prob=mp)

    # ----------------------------------- get the routes ---------------------------------------------
    result_mp.integer_optimize()
    result = result_mp.get_non_zero_routes(model='integer')
    re_clean_routes, re_step_exp_inv_list, re_step_target_inv_list = [], [], []
    re_step_loc_list, re_step_n_list = [], []
    for veh in range(num_of_van):
        if len(result['route'][veh]) > 0:
            veh_route = result['route'][veh][0]
            max_reward, loc_list, inv_list = esd_computer.compute_route(r=veh_route, t_left=van_dis_left[veh],
                                                                        init_l=van_load[veh], x_s_arr=x_s_arr,
                                                                        x_c_arr=x_c_arr, mode=mode, t_repo=t_repo,
                                                                        can_stay=True, to_print=False)
            best_route, best_inv = loc_list, inv_list
            print(f'best route: {best_route}, best inv: {best_inv}')
        else:
            best_route, best_inv = [van_location[veh] for _ in range(t_repo)], [van_load[veh] for _ in range(t_repo)]
            # need to scan other vehicles in case they visit the same station todo: this is myopic
            for v in range(num_of_van):
                if v != veh and len(result['route'][v]) > 0 and van_location[veh] in result['route'][v][0]:
                    result['route'][v][0].remove(van_location[veh])
                    assert van_location[veh] not in result['route'][v][0]

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
                    ] if mode == 'multi' else ei_s_arr[
                        best_route[step] - 1,
                        round(cur_t - RE_START_T / 10),
                        round(cur_t - RE_START_T / 10 + step),
                        round(x_s_arr[best_route[step] - 1])
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

    ed = time.process_time()
    print(f"Running time for branch & price: {ed - st} seconds")

    return {
        'objective': result_mp.model.ObjVal,
        'start_time': cur_t,
        'routes': re_clean_routes,
        'exp_inv': re_step_exp_inv_list,
        'exp_target_inv': re_step_target_inv_list,
        'loc': re_step_loc_list,
        'n_r': re_step_n_list,
    }
