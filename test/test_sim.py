import json
import numpy as np
import time
from sim.sim import Simulation
from sim.consts import *
from sim.system import Station
from sim.initialize import initialize_system

from test_case import test_case_all, test_case_25

if __name__ == '__main__':

    test_case = test_case_25

    start = time.process_time()
    test_result, test_result_work = [], []
    # initial settings should be the result of initialize_system()
    for _ in range(10):
        test = Simulation(**test_case)
        test.single = True
        test.policy = 'rollout'
        test.print_action = True
        test.run()
        test_result.append(test.success)
        test_result_work.append(test.success_work)
        test.print_simulation_log()
        test.print_stage_log()
    end = time.process_time()

    print(f"Running test_case with policy {test.policy} and {'single' if test.single else 'multi'} info")
    print('Running time: %s Seconds' % (end - start))
    print(f'Success avg: {np.mean(test_result)}, std: {np.std(test_result)}')
    print(f'Success after work avg: {np.mean(test_result_work)}, std: {np.std(test_result_work)}')

