from sim.system import Station


def get_init_station() -> dict:

    tmp = {i: Station(station_id=i, location=(i, i), capacity=50, capacity_opponent=50, num_self=i, num_opponent=50 - i) for i
     in range(1, 51)}

    return tmp
