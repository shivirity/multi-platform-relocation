import joblib
import time
import pickle
import numpy as np
import pandas as pd

from simulation.sim import Simulation
from simulation.consts import ORDER_INCOME_UNIT, DISTANCE_COST_UNIT
from test_case import test_case_25


def get_state_value_pair(var_list: list, cost_list: list, property_list: list, cost_after_work: float):
    """
    post-decision状态和动作配对

    :param var_list: 需要的post-decision变量列表
    :param cost_list: 价值函数值列表
    :param property_list: 特征值列表
    :param cost_after_work: relocation结束后的成本
    :return:
    """
    new_cost = list(np.cumsum(cost_list[::-1]))
    new_cost = [val + cost_after_work for val in new_cost]
    zipped = zip(property_list[-2::-1], new_cost[:-1])
    pair_list = []
    for pair in zipped:
        use_dict = {key: pair[0][key] for key in pair[0] if key in var_list}
        pair_list.append((use_dict, pair[1]))
    return pair_list


if __name__ == '__main__':

    test_case = test_case_25
    case_test = True
    selected_set = 'phi_7'
    selected_case = 25
    selected_duration = 30
    selected_rep = 5000

    # read in MLP model
    # model = joblib.load(f'../offline_VFA/MLP_unit_test/model/nn_state_value_GLA_test.pkl')

    # if case_test is True:
    #     # case test
    #     func_dict = {}
    #     func_params = pd.read_csv(f'../offline_VFA/params/params_{selected_set}_{selected_case}_{selected_duration}_{selected_rep}.csv', index_col=0)
    #     for i in list(func_params.index):
    #         func_dict[i] = {val: func_params.loc[i, val] for val in list(func_params.columns)}
    # else:
    #     # linear_regression_test
    #     with open(f'../offline_VFA/linear_regression_test/data/result_dict_{selected_set}_{selected_case}_{selected_rep}.pkl', 'rb') as f:
    #         func_dict = pickle.load(f)

    test_result, test_result_work, test_distance, test_value = [], [], [], []

    test_num = 50

    test_single = True
    test_policy = 'GLA'

    # MINLP model
    if test_single is True:
        with open('../expectation_calculation/EI_s_array_single.pkl', 'rb') as f:
            ei_s_arr = pickle.load(f)
        with open('../expectation_calculation/EI_c_array_multi.pkl', 'rb') as f:
            ei_c_arr = pickle.load(f)
        with open('../expectation_calculation/ESD_array_single.pkl', 'rb') as f:
            esd_arr = pickle.load(f)
    else:
        with open('../expectation_calculation/EI_s_array_multi.pkl', 'rb') as f:
            ei_s_arr = pickle.load(f)
        with open('../expectation_calculation/EI_c_array_multi.pkl', 'rb') as f:
            ei_c_arr = pickle.load(f)
        with open('../expectation_calculation/ESD_array_multi.pkl', 'rb') as f:
            esd_arr = pickle.load(f)

    start = time.process_time()
    for _ in range(test_num):
        # test = Simulation(**test_case, func_dict=func_dict, MLP_model=model)
        test = Simulation(**test_case, ei_s_arr=ei_s_arr, ei_c_arr=ei_c_arr, esd_arr=esd_arr)
        test.single = test_single
        test.policy = test_policy
        test.print_action = True
        test.run()
        # print(f'test_esd: {test.test_esd}')
        # print(f'val till work done: {test.success_work_till_done}, estimated val: {test.test_esd_till_work_done}')
        form_state_value_pair = False
        # print(test.success_work, test.success_work * ORDER_INCOME_UNIT - test.veh_distance * DISTANCE_COST_UNIT)
        # print(test.best_val_list)
        test_result.append(test.success)
        test_result_work.append(test.success_work)
        test_distance.append(sum(test.veh_distance))
        test_value.append(test.success_work * ORDER_INCOME_UNIT - sum(test.veh_distance) * DISTANCE_COST_UNIT)
        # cost_list.append(sum(test.cost_list))

        if _ % 1 == 0:
            print(f'testing process: {_} / {test_num}')
        # test.print_simulation_log()
        # test.print_stage_log()
    end = time.process_time()

    print(f"Running test_case with policy {test.policy} and {'single' if test.single else 'multi'} info")
    print('Running time: %s Seconds' % (end - start))
    print(f'Success avg: {np.mean(test_result)}')
    print(f'max: {max(test_result)}, min: {min(test_result)}, std: {np.std(test_result)}')
    print(f'Success after work avg: {np.mean(test_result_work)}')
    print(f'max: {max(test_result_work)}, min: {min(test_result_work)}, std: {np.std(test_result_work)}')
    print(f'Relocation vehicle distance (min): {np.mean(test_distance)}')
    print(f'max: {max(test_distance)}, min: {min(test_distance)}, std: {np.std(test_distance)}')
    print(f'Value avg: {np.mean(test_value)}')
    print(f'max: {max(test_value)}, min: {min(test_value)}, std: {np.std(test_value)}')
