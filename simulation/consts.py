# random seed
SEED = 42

# simulation time
SIM_START_T = 5 * 60  # 5 * 60
SIM_END_T = 22 * 60  # 22 * 60

RE_START_T = 14 * 60
RE_END_T = 20 * 60
RECORD_WORK_T = RE_START_T

# number of stations
NUM_STATIONS = 25

# station capacity
CAP_S = 40
CAP_C = 80

# minimum simulation step
MIN_STEP = 5
# minimum simulation run step
MIN_RUN_STEP = 10

# time duration for staying at current station
STAY_TIME = MIN_RUN_STEP

# inventory decision levels
DEC_LEVELS = [0.2, 0.4, 0.6, 0.8]

# vehicle capacity
VEH_CAP = 50

# operation time per bike
OPERATION_TIME = 0.5
# operation const time
CONST_OPERATION = 10

# rollout simulation times
ROLLOUT_SIM_TIMES = 32

# STR balance parameter
GAMMA = 0.3

# travel cost for every minute (single vehicle)
UNIT_TRAVEL_COST = 1

# (single-info) GLA lookahead horizon (in minute)
GLA_HORIZON = 120
GLA_delta = 0.5

# single rollout horizon
SINGLE_ROLLOUT_HORIZON = 360

# distance matrix (for vehicle) fix rate (15km/h, min -> new speed, min)
DIST_FIX = 4
# arrival rate (bikes/5min)
SELF_ARR_RATE_FIX = 0.8
OPPO_ARR_RATE_FIX = 0.8
# departure rate (bikes/5min)
DEP_FIX = 1

# offline VFA training params

POLICY_DURATION = 30  # in minute, the policy changes every POLICY_DURATION minutes

SAFETY_INV_LB = 0
SAFETY_INV_UB = 1
EPSILON = 0.1  # in epsilon-greedy policy
DISTANCE_COST_UNIT = 0.25
ORDER_INCOME_UNIT = 2
LAMBDA = 0.98  # in RLS algorithm
SMALL_CONST_IN_INIT = 0.01  # in RLS algorithm
