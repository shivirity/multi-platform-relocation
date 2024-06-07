import numpy as np
from route_extension.bph.models import MasterProblem
from route_extension.route_extension_algo import get_REA_initial_routes, ESDComputer
from route_extension.cg_re_algo import get_dp_reduced_cost_bidirectional_numba
from simulation.consts import RE_END_T, CAP_C


class GreedySolution:
    """
    Model to find initial routes using REA
    """

    def __init__(self, num_of_van: int, van_location: list, van_dis_left: list,
                 van_load: list, c_s: int, c_v: int, cur_t: int, t_p: int, t_f: int,
                 t_roll: int, c_mat: np.ndarray, ei_s_arr: np.ndarray, ei_c_arr: np.ndarray,
                 esd_arr: np.ndarray, x_s_arr: list, x_c_arr: list, alpha: float, est_ins: list, mode: str):
        self.num_of_van = num_of_van
        self.van_location = van_location
        self.van_dis_left = van_dis_left
        self.van_load = van_load
        self.c_s = c_s
        self.c_v = c_v
        self.cur_t = cur_t
        self.t_p = t_p
        self.t_f = t_f
        self.t_roll = t_roll
        self.c_mat = c_mat
        self.ei_s_arr = ei_s_arr
        self.ei_c_arr = ei_c_arr
        self.esd_arr = esd_arr
        self.x_s_arr = x_s_arr
        self.x_c_arr = x_c_arr
        self.alpha = alpha
        self.est_ins = est_ins
        self.mode = mode

    def generate_init_solution(self, computer: ESDComputer):

        init_routes, init_profits = get_REA_initial_routes(
            num_of_van=self.num_of_van,
            van_location=self.van_location,
            van_dis_left=self.van_dis_left,
            van_load=self.van_load,
            c_s=self.c_s,
            c_v=self.c_v,
            cur_t=self.cur_t,
            t_p=self.t_p,
            t_f=self.t_f,
            t_roll=self.t_roll,
            c_mat=self.c_mat,
            ei_s_arr=self.ei_s_arr,
            ei_c_arr=self.ei_c_arr,
            esd_arr=self.esd_arr,
            x_s_arr=self.x_s_arr,
            x_c_arr=self.x_c_arr,
            alpha=self.alpha,
            est_ins=self.est_ins,
            branch=2,
            state='init',
            mode=self.mode
        )
        num_stations = self.c_mat.shape[0] - 1
        t_repo = self.t_p if self.cur_t + self.t_p <= RE_END_T / 10 else round(RE_END_T / 10 - self.cur_t)
        visited_stations = []
        generated_routes, generated_profits = [[] for _ in range(self.num_of_van)], [[] for _ in range(self.num_of_van)]
        for veh in range(self.num_of_van):
            dual_station_vec = [0 for _ in range(num_stations)]
            for i in range(self.num_of_van):
                if i != veh and self.van_location[i] != 0:
                    dual_station_vec[self.van_location[i] - 1] = 1000000  # avoid visiting same stations
            for node in visited_stations:
                if node != 0:
                    dual_station_vec[node - 1] = 1000000  # avoid visiting visited stations
            default_inv_id_arr = np.array([0, 0, 0, 0, 0,
                                           1, 1, 1, 1, 1,
                                           2, 2, 2, 2, 2,
                                           3, 3, 3, 3, 3,
                                           4, 4, 4, 4, 4,
                                           5], dtype=np.int8)
            default_inv_arr = np.array([0, 5, 10, 15, 20, 25], dtype=np.int8)
            if self.mode == 'multi':
                com_ei_s_arr, com_esd_arr = self.ei_s_arr, self.esd_arr
            else:
                assert self.mode == 'single'
                new_ei_shape, new_esd_shape = (*self.ei_s_arr.shape, CAP_C + 1), (*self.esd_arr.shape, CAP_C + 1)
                com_ei_s_arr = np.broadcast_to(np.expand_dims(self.ei_s_arr, axis=-1), shape=new_ei_shape)
                com_ei_s_arr = np.ascontiguousarray(com_ei_s_arr)
                com_esd_arr = np.broadcast_to(np.expand_dims(self.esd_arr, axis=-1), shape=new_esd_shape)
                com_esd_arr = np.ascontiguousarray(com_esd_arr)
            route = get_dp_reduced_cost_bidirectional_numba(
                cap_s=self.c_s,
                num_stations=num_stations,
                init_loc=self.van_location[veh],
                init_t_left=self.van_dis_left[veh],
                init_load=self.van_load[veh],
                x_s_arr=np.array(self.x_s_arr, dtype=np.int32),
                x_c_arr=np.array(self.x_c_arr, dtype=np.int32),
                ei_s_arr=com_ei_s_arr,
                ei_c_arr=self.ei_c_arr,
                esd_arr=com_esd_arr,
                c_mat=self.c_mat,
                cur_t=self.cur_t,
                t_p=self.t_p,
                t_f=self.t_f,
                alpha=self.alpha,
                dual_station_vec=np.array(dual_station_vec, dtype=np.float64),
                inventory_dict=default_inv_arr,
                inventory_id_dict=default_inv_id_arr
            )
            for i in route:
                if i != self.van_location[veh]:
                    visited_stations.append(i)

            max_reward, _, __ = computer.compute_route(r=route, t_left=self.van_dis_left[veh],
                                                       init_l=self.van_load[veh], x_s_arr=self.x_s_arr,
                                                       x_c_arr=self.x_c_arr, t_repo=t_repo,
                                                       can_stay=True, to_print=False, mode=self.mode)
            generated_routes[veh].append(list(route))
            generated_profits[veh].append(max_reward)

        if all(x == self.van_location[0] for x in self.van_location) and \
                all(x == self.van_dis_left[0] for x in self.van_dis_left) and \
                all(x == self.van_load[0] for x in self.van_load):
            same_state = True
        else:
            same_state = False

        if same_state:
            for veh in range(self.num_of_van):
                for van in range(self.num_of_van):
                    for route_ind in range(len(generated_routes[van])):
                        init_routes[veh].append(generated_routes[van][route_ind])
                        init_profits[veh].append(generated_profits[van][route_ind])
        else:
            for veh in range(self.num_of_van):
                for route_ind in range(len(generated_routes[veh])):
                    init_routes[veh].append(generated_routes[veh][route_ind])
                    init_profits[veh].append(generated_profits[veh][route_ind])

        return init_routes, init_profits


class Node:
    """Node class for the branch and price algorithm"""

    def __init__(self, node_id: str, lp_obj: float, mp: MasterProblem):
        self.node_id = node_id
        self.lp_obj = lp_obj
        self.mp = mp


class Stack(object):
    """Stack class for the branch and price algorithm"""

    def __init__(self):
        self.stack = []

    def __str__(self):
        return ' '.join([str(i) for i in self.stack])

    def __len__(self):
        return len(self.stack)

    def is_empty(self):
        return len(self.stack) == 0

    def push(self, node: Node):
        self.stack.append(node)

    def pop(self):
        try:
            return self.stack.pop()
        except IndexError:
            print()
            exit()
