import logging
import time
import numpy as np
import pandas as pd

from simulation.sim import Simulation
from simulation.consts import LAMBDA, RE_START_T, RE_END_T, POLICY_DURATION, SMALL_CONST_IN_INIT
from test_file.test_case import test_case_25


class VFATrainer:

    test_case = test_case_25

    def __init__(self, trainer_name: str, num_of_funcs: int, func_names: list, train_rep: int, method: str = 'RLS'):
        """传入变量个数，变量名，建立变量字典"""
        self.name = trainer_name
        self.num_of_funcs = num_of_funcs
        self.func_names = func_names
        self.train_rep = train_rep
        self.method = method
        self.func_dict = {}
        self.init_func_dict()
        self.rep = 0

        # 自变量和因变量
        self.t = []  # 时间索引
        self.x = []
        self.y = []

        # record
        self.recent_10_avg_list = []

        # RLS params
        self.B = None
        self.init_dim = 0
        self.gamma = {key: 0 for key in range(int(RE_START_T/POLICY_DURATION), int(RE_END_T/POLICY_DURATION) + 1)}

        # SGA params

        if self.method == 'RLS':
            self.init_B()

    def init_B(self):
        """初始化B_0矩阵"""
        X = np.empty(shape=(0, self.num_of_funcs))
        y_list = []
        try_times = 0
        while self.init_dim < self.num_of_funcs and try_times <= 20:
            sim = Simulation(**self.test_case, func_dict=self.func_dict)
            sim.single = False
            sim.policy = 'offline_VFA_train'
            sim.random_choice_to_init_B = True
            sim.run()

            new_cost = list(np.cumsum(sim.cost_list[::-1]))
            new_cost = list([val+sim.cost_after_work for val in new_cost])[:-2]
            new_property = list(sim.basis_func_property[-2::-1])[:-1]
            zipped = zip(new_property, new_cost)
            for pair in zipped:
                new_x_T, new_y = self.convert_x_to_vector(pair[0]), pair[1]
                test = np.append(X, new_x_T, axis=0)
                if test.shape[0] >= self.num_of_funcs and np.linalg.matrix_rank(test) == self.num_of_funcs:
                    X = np.append(X, new_x_T, axis=0)
                    y_list.append(new_y)
                    break
                else:
                    if X.shape[0] == 0:
                        X = np.append(X, new_x_T, axis=0)
                        y_list.append(new_y)
                    elif np.linalg.matrix_rank(test) > np.linalg.matrix_rank(X):
                        X = np.append(X, new_x_T, axis=0)
                        y_list.append(new_y)
            self.init_dim = X.shape[0]
            print(f'current X size: {X.shape}, rank={np.linalg.matrix_rank(X)}, {np.linalg.matrix_rank(X.T @ X)}')
            try_times += 1

        if self.init_dim < self.num_of_funcs:
            print('X matrix rank is not full, use small const to initialize B matrix.')
            theta_array = np.array([0 for _ in range(self.num_of_funcs)]).reshape((-1, 1))
            self.convert_vector_to_theta(theta_array=theta_array, index=-1)
            B_0 = SMALL_CONST_IN_INIT * np.eye(self.num_of_funcs)
            print(f'B matrix successfully initialized, shape: {B_0.shape}')
            self.B = \
                {key: np.array(B_0) for key in
                 range(int(RE_START_T / POLICY_DURATION), int(RE_END_T / POLICY_DURATION) + 1)}

        else:
            Y = np.array(y_list).reshape((-1, 1))
            theta = np.linalg.inv(X.T @ X) @ X.T @ Y
            self.convert_vector_to_theta(theta_array=theta, index=-1)
            B_0 = np.linalg.inv(np.dot(X.T, X))
            print(f'B matrix successfully initialized, shape: {B_0.shape}')
            self.B = \
                {key: np.array(B_0) for key in range(int(RE_START_T/POLICY_DURATION), int(RE_END_T/POLICY_DURATION) + 1)}

    def init_func_dict(self):
        """初始化参数表"""
        for k in range(int(RE_START_T/POLICY_DURATION), int(RE_END_T/POLICY_DURATION) + 1):
            self.func_dict[k] = {key: 0 for key in self.func_names}

    def train(self, **kwargs):
        """返回训练后的参数表"""

        print_list = kwargs['print_list'] if 'print_list' in kwargs.keys() else []

        self.init_func_dict()
        recent_10 = []
        while self.rep < self.train_rep:
            sim = Simulation(**self.test_case, func_dict=self.func_dict)
            sim.single = False
            sim.policy = 'offline_VFA_train'
            # sim.print_action = True
            sim.run()
            recent_10.append(sim.success_work)
            self.update_func_dict(
                time_list=sim.dec_time_list, cost_list=sim.cost_list,
                property_list=sim.basis_func_property, cost_after_work=sim.cost_after_work)
            if self.rep % 10 == 0 and self.rep > 0:
                cost_sum = sum(sim.cost_list)
                real_sum = sim.success_work_till_done
                real_sum_done = sim.success_work
                recent_10_avg = sum(recent_10)/len(recent_10)
                self.recent_10_avg_list.append(recent_10_avg)
                print('Training process: {}/{}'.format(self.rep, self.train_rep))
                print(f'Recent 10 simulation average:{recent_10_avg}')
                print(
                    f'cost_sum: {cost_sum}, real_sum: {real_sum}, cost_sum_till_sim_end: {cost_sum + sim.cost_after_work}, real_sum_till_sim_end: {real_sum_done}')
                recent_10 = []

            self.rep += 1
            if self.rep in print_list:
                self.to_csv()

    def update_func_dict(self, time_list: list, cost_list: list, property_list: list, cost_after_work: float = 0):
        """
        更新参数表

        :param time_list: 每次仿真各阶段决策时点列表
        :param cost_list: 每次仿真各阶段成本列表
        :param property_list: 每次仿真各阶段特征值列表
        :param cost_after_work: relocation结束后的成本
        :return:
        """
        if self.method == 'RLS':
            new_cost = list(np.cumsum(cost_list[::-1]))
            new_cost = [val+cost_after_work for val in new_cost]
            zipped = zip(time_list[-2::-1], property_list[-2::-1], new_cost[:-1])
            for pair in zipped:
                new_t_idx, new_x, new_y = int(pair[0]/POLICY_DURATION), self.convert_x_to_vector(pair[1]).T, pair[2]
                assert new_y > 0, f'{new_t_idx, new_x, new_y}'
                theta = self.convert_theta_to_vector(index=new_t_idx)
                self.t.append(new_t_idx)
                self.x.append(new_x)
                self.y.append(new_y)
                self.gamma[new_t_idx] = LAMBDA + float(new_x.T @ self.B[new_t_idx] @ new_x)
                assert self.gamma[new_t_idx] != 0, f'{new_x, float(new_x.T @ self.B[new_t_idx] @ new_x)}'
                # assert abs(new_x.T @ theta - new_y) < 1e10, f'{new_x, new_y, theta}'
                theta = theta - self.B[new_t_idx] @ new_x * (float(new_x.T @ theta) - new_y) / self.gamma[new_t_idx]
                self.B[new_t_idx] = \
                    1 / LAMBDA * (self.B[new_t_idx] - self.B[new_t_idx] @ new_x @ new_x.T @ self.B[new_t_idx] / self.gamma[new_t_idx])
                self.convert_vector_to_theta(index=new_t_idx, theta_array=theta)

        elif self.method == 'SGA':
            pass
        else:
            assert False, 'Invalid method!'

    def convert_theta_to_vector(self, index: int):
        """将参数表转换为向量"""
        theta_array = np.array([self.func_dict[index][key] for key in self.func_names])
        return theta_array.reshape((-1, 1))

    def convert_vector_to_theta(self, index: int, theta_array: np.ndarray):
        """将向量转换为参数表, index为时间索引, -1代表全部更新"""
        if index >= 0:
            for k in range(self.num_of_funcs):
                self.func_dict[index][self.func_names[k]] = float(theta_array[k])
        else:
            for idx in self.func_dict.keys():
                for k in range(self.num_of_funcs):
                    self.func_dict[idx][self.func_names[k]] = float(theta_array[k])

    def convert_x_to_vector(self, property_dict: dict):
        """将特征值字典转换为向量"""
        property_array = np.array([property_dict[key] for key in self.func_names])
        return property_array.reshape((1, -1))

    def to_csv(self):
        """输出结果"""
        out_dict = {}
        for name in self.func_names:
            out_dict[name] = [val[name] for val in self.func_dict.values()]
        df = pd.DataFrame(out_dict)
        df.index = list(self.func_dict.keys())
        # format: params_{trainer_name}_{num_of_stations}_{policy_duration}_{train_rep}.csv
        df.to_csv(f'params/params_{self.name}_25_{POLICY_DURATION}_{self.rep}.csv')


if __name__ == '__main__':

    case_dict = {
        'phi_1': {'trainer_name': 'phi_1', 'num_of_funcs': 1, 'func_names': ['const']},
        'phi_2': {'trainer_name': 'phi_2', 'num_of_funcs': 2, 'func_names': ['const', 'veh_load']},
        'phi_3': {'trainer_name': 'phi_3', 'num_of_funcs': 2 + 25, 'func_names': ['const', 'veh_load']},
        'phi_4': {'trainer_name': 'phi_4', 'num_of_funcs': 2 + 25 + 25, 'func_names': ['const', 'veh_load']},
        'phi_5': {'trainer_name': 'phi_5', 'num_of_funcs': 2 + 25 + 25 + 25, 'func_names': ['const', 'veh_load']},
        'phi_6': {'trainer_name': 'phi_6', 'num_of_funcs': 2 + 25 + 25, 'func_names': ['const', 'veh_load']},
        'phi_7': {'trainer_name': 'phi_7', 'num_of_funcs': 4 + 25 + 25 + 25, 'func_names': ['const', 'veh_load', 'step_t', 'des_inv']},
        'phi_8': {'trainer_name': 'phi_8', 'num_of_funcs': 4 + 25 + 25 + 25 + 25, 'func_names': ['const', 'veh_load', 'step_t', 'des_inv']},
        'phi_9': {'trainer_name': 'phi_9', 'num_of_funcs': 4 + 25 + 25 + 25 + 25 + 25 + 25, 'func_names': ['const', 'time', 'veh_load', 'des_inv']},
    }
    # fix case phi_3
    for i in range(1, 26):
        case_dict['phi_3']['func_names'].append(f'veh_des_{i}')
    # fix case phi_4
    for i in range(1, 26):
        case_dict['phi_4']['func_names'].append(f'veh_des_{i}')
        case_dict['phi_4']['func_names'].append(f'orders_till_sim_end_{i}')
    # fix case phi_5
    for i in range(1, 26):
        case_dict['phi_5']['func_names'].append(f'veh_des_{i}')
        case_dict['phi_5']['func_names'].append(f'orders_till_sim_end_{i}')
        case_dict['phi_5']['func_names'].append(f'num_self_{i}')
    # fix case phi_6
    for i in range(1, 26):
        case_dict['phi_6']['func_names'].append(f'veh_des_{i}')
    for i in range(1, 26):
        case_dict['phi_6']['func_names'].append(f'self_order_proportion_till_ends_{i}')
    # fix case phi_7
    for i in range(1, 26):
        case_dict['phi_7']['func_names'].append(f'veh_loc_{i}')
    for i in range(1, 26):
        case_dict['phi_7']['func_names'].append(f'num_self_{i}')
    for i in range(1, 26):
        case_dict['phi_7']['func_names'].append(f'num_oppo_{i}')
    # fix case phi_8
    for i in range(1, 26):
        case_dict['phi_8']['func_names'].append(f'veh_loc_{i}')
    for i in range(1, 26):
        case_dict['phi_8']['func_names'].append(f'num_self_{i}')
    for i in range(1, 26):
        case_dict['phi_8']['func_names'].append(f'num_oppo_{i}')
    for i in range(1, 26):
        case_dict['phi_8']['func_names'].append(f'net_demand_in_2_hours_{i}')
    # fix case phi_9
    for i in range(1, 26):
        case_dict['phi_9']['func_names'].append(f'veh_loc_{i}')
    for i in range(1, 26):
        case_dict['phi_9']['func_names'].append(f'num_self_{i}')
    for i in range(1, 26):
        case_dict['phi_9']['func_names'].append(f'num_oppo_{i}')
    for i in range(1, 26):
        case_dict['phi_9']['func_names'].append(f'orders_till_sim_end_{i}')
    for i in range(1, 26):
        case_dict['phi_9']['func_names'].append(f'bikes_s_arr_till_sim_end{i}')
    for i in range(1, 26):
        case_dict['phi_9']['func_names'].append(f'bikes_c_arr_till_sim_end{i}')

    # train process
    train_case = 'phi_9'
    train_rep_list = [100, 250, 500, 750, 1000]  # 递增训练次数

    start = time.process_time()
    trainer = VFATrainer(
        **case_dict[train_case],
        train_rep=train_rep_list[-1],
        method='RLS'
    )
    trainer.train(print_list=train_rep_list)
    end = time.process_time()
    print('Running time: %s Seconds' % (end - start))

    # test process output
    main_dict = {'t': trainer.t}
    for i in range(len(trainer.func_names)):
        main_dict[trainer.func_names[i]] = [val[i][0] for val in trainer.x]
    main_dict['y'] = list(trainer.y)
    main_df = pd.DataFrame(main_dict)
    # format: linear_regression_test_{trainer_name}_{num_of_stations}_{policy_duration}_{train_rep}.csv
    main_df.to_csv(rf"linear_regression_test\data\linear_regression_test_{trainer.name}_25_{POLICY_DURATION}_{train_rep_list[-1]}.csv")
