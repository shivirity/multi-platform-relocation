class Station:
    def __init__(self, station_id: int = 0, location: tuple = (0, 0), capacity: int = 0, capacity_opponent: int = 0,
                 num_self: int = 0, num_opponent: int = 0):
        """
        Station类

        :param station_id:
        :param location:
        :param num_self:
        :param num_opponent:
        """
        self.id = station_id
        self.loc = location
        self.cap = capacity
        self.cap_opponent = capacity_opponent
        self.num_self = num_self
        self.num_opponent = num_opponent

    def change_num(self, change_tuple: tuple):
        """
        站内车辆数变化

        :param change_tuple: (自身平台变化，竞对平台变化)
        :return:
        """
        self.num_self += change_tuple[0]
        self.num_opponent += change_tuple[1]
        assert self.num_self >= 0 and self.num_opponent >= 0
