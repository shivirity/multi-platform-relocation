import time
import numpy as np
import pandas as pd

from simulation.sim import Simulation
from simulation.consts import ORDER_INCOME_UNIT, DISTANCE_COST_UNIT
from test_case import test_case_25

if __name__ == '__main__':

    test_case = test_case_25
    selected_set = 'phi_3'
    selected_case = 25
    selected_rep = 500
    func_params = pd.read_csv(f'../offline_VFA/params/params_{selected_set}_{selected_case}_{selected_rep}.csv')
    func_dict = dict(zip(func_params['key'], func_params['value']))

    start = time.process_time()
    test_result, test_result_work, test_distance, test_value = [], [], [], []
    # initial settings should be the result of initialize_system()
    test = None
    for _ in range(100):
        test = Simulation(**test_case, func_dict=func_dict)
        test.single = False
        test.policy = 'online_VFA'
        test.print_action = False
        test.run()
        test_result.append(test.success)
        test_result_work.append(test.success_work)
        test_distance.append(test.veh_distance)
        test_value.append(test.success_work * ORDER_INCOME_UNIT - test.veh_distance * DISTANCE_COST_UNIT)
        if _ % 10 == 0:
            print(f'testing process: {_} / 100')
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
