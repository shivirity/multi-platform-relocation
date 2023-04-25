# random seed
SEED = 42

# simulation time
SIM_START_T = 0
SIM_END_T = 12 * 60

# number of stations
NUM_STATIONS = 50

# minimum simulation step
MIN_STEP = 5

# time duration for staying at current station
STAY_TIME = MIN_STEP

# inventory decision levels
DEC_LEVELS = [0.2, 0.4, 0.6, 0.8]

# vehicle capacity
VEH_CAP = 50

# operation time per bike
OPERATION_TIME = 0.5

# rollout simulation times
ROLLOUT_SIM_TIMES = 5

# STR balance parameter
GAMMA = 0.3

'''
### BIKE CONSTANTS ###

CLASSIC_BASE_RATE = 3.50
CLASSIC_ADD_RATE = 0.10

ELECTRIC_BASE_RATE = 5.25
ELECTRIC_ADD_RATE = 0.20
ELECTRIC_MAX_CHARGE = 100

### SIMULATION CONSTANTS ###

SMALL_STATION = 10  # No. of docks
MEDIUM_STATION = 15  # No. of docks
LARGE_STATION = 20  # No. of docks

NUM_BIKES = 80    # Total no. of bikes in system

LAMBDA = 4  # 1 ride / 10 mins or 0.1 ride / min
LENGTH = 60  # Length of simulation in minutes
SPEED = 0.5  # Distance units per minute
'''