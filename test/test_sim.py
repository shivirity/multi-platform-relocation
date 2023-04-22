import json
import numpy as np
import time
from sim.sim import Simulation
from sim.system import Station
from sim.initialize import initialize_system


stations = {i: Station(station_id=i,location=(i,i),capacity=50,capacity_opponent=50,num_self=i,num_opponent=50-i) for i in range(1,51)}
dist_array = np.random.randint(2, 7, size=(50, 50))
for i in range(50):
    dist_array[i, i] = 0
mu_array = np.random.uniform(0, 8, (720, 50))  # (time, station)
lambda_s_array = np.random.uniform(0, 2, (720, 50))  # (time, station)
lambda_c_array = np.random.uniform(0, 2, (720, 50))  # (time, station)


if __name__ == '__main__':
    start = time.process_time()
    # initial settings should be the result of initialize_system()
    test = Simulation(stations=stations, dist_array=dist_array,
                      mu_array=mu_array, lambda_s_array=lambda_s_array, lambda_c_array=lambda_c_array)
    test.run()
    end = time.process_time()
    test.print_simulation_log()
    test.print_stage_log()
    print('Running time: %s Seconds' % (end - start))

