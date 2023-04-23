import json
import numpy as np
import time
from sim.sim import Simulation
from sim.system import Station
from sim.initialize import initialize_system

from test_case import test_case_1, test_case_2

if __name__ == '__main__':

    test_case = test_case_1

    start = time.process_time()
    test_result = []
    # initial settings should be the result of initialize_system()
    for _ in range(200):
        test = Simulation(**test_case)
        test.policy = None
        test.run()
        test_result.append(test.success)
        test.print_simulation_log()
        test.print_stage_log()
    end = time.process_time()

    print(f'Running test_case_1 with policy {test.policy}')
    print('Running time: %s Seconds' % (end - start))
    print(f'Success avg: {np.mean(test_result)}, std: {np.std(test_result)}')

