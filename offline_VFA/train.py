import time
import numpy as np
import pandas as pd

from simulation.sim import Simulation
from simulation.consts import LAMBDA
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
        self.rep = 0
        for _ in range(num_of_funcs):
            self.func_dict[func_names[_]] = 0

        # RLS params
        self.B = None
        self.init_dim = 0
        self.gamma = 0

        # SGA params

        if self.method == 'RLS':
            self.B = self.init_B()

    def init_B(self):
        """初始化B_0矩阵"""
        B = np.empty(shape=(0, self.num_of_funcs))
        while self.init_dim <= self.num_of_funcs:
            sim = Simulation(**self.test_case, func_dict=self.func_dict)
            sim.single = False
            sim.policy = 'offline_VFA_train'
            sim.run()
            for _ in range(0, len(sim.basis_func_property)-1):
                B = np.append(B, self.convert_x_to_vector(sim.basis_func_property[_]), axis=0)
            self.init_dim = B.shape[0]
        B_0 = np.dot(B.T, B)
        return B_0

    def init_func_dict(self):
        for key, value in self.func_dict.items():
            self.func_dict[key] = 0

    def train(self, **kwargs):
        """返回训练后的参数表"""

        print_list = kwargs['print_list'] if 'print_list' in kwargs.keys() else []

        self.init_func_dict()
        while self.rep < self.train_rep:
            sim = Simulation(**self.test_case, func_dict=self.func_dict)
            sim.single = False
            sim.policy = 'offline_VFA_train'
            # sim.print_action = True
            sim.run()
            self.update_func_dict(cost_list=sim.cost_list, property_list=sim.basis_func_property)
            if self.rep % 10 == 0 and self.rep > 0:
                cost_sum = sum(sim.cost_list)
                real_sum = sim.success_work
                print('Training process: {}/{}'.format(self.rep, self.train_rep))
                print(f'cost_sum: {cost_sum}, real_sum: {real_sum}')

            self.rep += 1
            if self.rep in print_list:
                self.to_csv()

    def update_func_dict(self, cost_list: list, property_list: list):
        """
        更新参数表

        :param cost_list: 每次仿真各阶段成本列表
        :param property_list: 每次仿真各阶段特征值列表
        :return:
        """
        if self.method == 'RLS':
            assert self.B is not None, 'B is None!'
            new_cost = list(np.cumsum(cost_list[::-1]))
            zipped = zip(property_list[-2::-1], new_cost[:-1])
            for pair in zipped:
                theta = self.convert_theta_to_vector()
                new_x, new_y = self.convert_x_to_vector(pair[0]).T, pair[1]
                # todo 导出数据单独查看回归准确性
                self.gamma = LAMBDA + float(new_x.T @ self.B @ new_x)
                assert self.gamma != 0, f'{new_x, float(new_x.T @ self.B @ new_x)}'
                # assert abs(new_x.T @ theta - new_y) < 1e10, f'{new_x, new_y, theta}'
                theta = theta - self.B @ new_x * (float(new_x.T @ theta) - new_y) / self.gamma
                self.B = 1 / LAMBDA * (self.B - self.B @ new_x @ new_x.T @ self.B / self.gamma)
                self.convert_vector_to_theta(theta_array=theta)

        elif self.method == 'SGA':
            pass
        else:
            assert False, 'Invalid method!'

    def convert_theta_to_vector(self):
        """将参数表转换为向量"""
        theta_array = np.array([self.func_dict[key] for key in self.func_names])
        return theta_array.reshape((-1, 1))

    def convert_vector_to_theta(self, theta_array: np.ndarray):
        """将向量转换为参数表"""
        for i in range(self.num_of_funcs):
            self.func_dict[self.func_names[i]] = float(theta_array[i])

    def convert_x_to_vector(self, property_dict: dict):
        """将特征值字典转换为向量"""
        property_array = np.array([property_dict[key] for key in self.func_names])
        return property_array.reshape((1, -1))

    def to_csv(self):
        """输出结果"""
        key_list, value_list = [], []
        for key, value in self.func_dict.items():
            key_list.append(key)
            value_list.append(value)
        dict_to_csv = {'key': key_list, 'value': value_list}
        df = pd.DataFrame(dict_to_csv)

        # format: params_{trainer_name}_{num_of_stations}_{train_rep}.csv
        df.to_csv(f'params/params_{self.name}_25_{self.rep}.csv', index=False)


if __name__ == '__main__':

    case_dict = {
        'phi_1': {'trainer_name': 'phi_1', 'num_of_funcs': 1, 'func_names': ['const']},
        'phi_2': {'trainer_name': 'phi_2', 'num_of_funcs': 2, 'func_names': ['const', 'veh_load']},
        'phi_3': {'trainer_name': 'phi_3', 'num_of_funcs': 2 + 25, 'func_names': ['const', 'veh_load']}
    }
    # fix case phi_3
    for i in range(1, 26):
        case_dict['phi_3']['func_names'].append(f'veh_des_{i}')

    # train process
    train_case = 'phi_3'
    train_rep_list = [100, 200, 500]  # 递增训练次数

    start = time.process_time()
    trainer = VFATrainer(
        **case_dict[train_case],
        train_rep=train_rep_list[-1],
        method='RLS'
    )
    trainer.train(print_list=train_rep_list)
    end = time.process_time()
    print('Running time: %s Seconds' % (end - start))
