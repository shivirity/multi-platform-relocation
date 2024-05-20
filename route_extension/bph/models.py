import copy
import numpy as np
import gurobipy as gp
from simulation.consts import ORDER_INCOME_UNIT


def generate_veh_mat(k, *args):
    total = sum(args)  # 计算总列数
    arr = np.zeros((k, total), dtype=int)  # 初始化全0数组
    start = 0  # 初始化起始指针
    for i, length in enumerate(args):
        arr[i, start:start + length] = 1  # 在指定区域设值为1
        start += length  # 更新指针位置上
    return arr


def generate_node_mat(num_stations: int, route_pool: list):
    """generate node_mat from the route pool (list of lists)"""
    num_routes = len(route_pool)
    node_mat = np.zeros((num_stations, num_routes))
    for j in range(num_routes):
        for i in range(1, num_stations + 1):
            if i in route_pool[j]:
                node_mat[i - 1, j] = 1
    return node_mat


class MasterProblem:
    """
    Model to for master problem (set covering problem)
    """

    def __init__(self, num_veh: int, num_stations: int, van_location: list,
                 van_dis_left: list, van_load: list, x_s_arr: list, x_c_arr: list,
                 init_routes: list = None, init_profits: list = None, esd_list: list = None, model: gp.Model = None):
        self.num_veh = num_veh
        self.num_stations = num_stations
        self.van_location = van_location
        self.van_dis_left = van_dis_left
        self.van_load = van_load
        self.x_s_arr = x_s_arr
        self.x_c_arr = x_c_arr
        if init_routes is not None:
            self.route_pool = [list(item) for sublist in init_routes for item in sublist]
            self.node_mat = generate_node_mat(num_stations, self.route_pool)
        else:
            self.route_pool = None
            self.node_mat = None
        if init_profits is not None:
            self.profit_pool = [item for sublist in init_profits for item in sublist]
            self.veh_mat = generate_veh_mat(num_veh, *map(len, init_profits))
        else:
            self.profit_pool = None
            self.veh_mat = None
        if esd_list is not None:
            self.esd_list = list(esd_list)
            self.no_repo_esd = sum(esd_list)
        else:
            self.esd_list = None
            self.no_repo_esd = None

        if model is None:
            self.model = gp.Model("MP")
        else:
            self.model = model.copy()

        # None for relax model
        self.relax_model = None

        # None for model constraints
        self.model_x = None
        self.model_veh_cons = None
        self.model_node_cons = None
        self.model_node_visit_cons = None  # disorder

        self.must_visit_station = []
        self.must_not_visit_station = []

    def __deepcopy__(self, memodict={}):
        new_mp = MasterProblem(num_veh=self.num_veh, num_stations=self.num_stations,
                               van_location=list(self.van_location), van_dis_left=list(self.van_dis_left),
                               van_load=list(self.van_load), x_s_arr=list(self.x_s_arr), x_c_arr=list(self.x_c_arr),
                               model=self.model)
        new_mp.route_pool = copy.deepcopy(self.route_pool)
        new_mp.node_mat = copy.deepcopy(self.node_mat)
        new_mp.profit_pool = copy.deepcopy(self.profit_pool)
        new_mp.veh_mat = copy.deepcopy(self.veh_mat)
        new_mp.esd_list = copy.deepcopy(self.esd_list)
        new_mp.no_repo_esd = self.no_repo_esd
        new_mp.relax_model = self.relax_model.copy() if self.relax_model is not None else None
        new_mp.must_visit_station = list(self.must_visit_station)
        new_mp.must_not_visit_station = list(self.must_not_visit_station)

        new_mp.model_x = {}
        new_mp.model_veh_cons = {}
        new_mp.model_node_cons = {}
        new_mp.model_node_visit_cons = {}
        for var_idx in self.model_x.keys():
            var_name = self.model_x[var_idx].VarName
            new_mp.model_x[var_idx] = new_mp.model.getVarByName(var_name)
        for veh_idx in self.model_veh_cons.keys():
            veh_con_name = self.model_veh_cons[veh_idx].ConstrName
            new_mp.model_veh_cons[veh_idx] = new_mp.model.getConstrByName(veh_con_name)
        for node_idx in self.model_node_cons.keys():
            node_con_name = self.model_node_cons[node_idx].ConstrName
            new_mp.model_node_cons[node_idx] = new_mp.model.getConstrByName(node_con_name)
        if self.model_node_visit_cons is not None:
            for node_idx in self.model_node_visit_cons.keys():
                node_con_name = self.model_node_visit_cons[node_idx].ConstrName
                new_mp.model_node_visit_cons[node_idx] = new_mp.model.getConstrByName(node_con_name)

        return new_mp

    def build_model(self):
        """build initial model for MP"""
        num_routes = len(self.route_pool)
        num_veh = self.num_veh
        # variables
        x = {}
        for j in range(num_routes):
            x[j] = self.model.addVar(vtype=gp.GRB.BINARY, name=f'x{j}')
        # constraints
        # vehicle constr
        veh_cons = {}
        for j in range(num_veh):
            veh_cons[j] = self.model.addConstr(
                gp.quicksum(self.veh_mat[j, k] * x[k] for k in range(num_routes)) <= 1, name=f'veh_{j}')
        # node constr
        node_cons = {}
        for j in range(self.num_stations):
            node_cons[j] = self.model.addConstr(gp.quicksum(self.node_mat[j, k] * x[k] for k in range(num_routes)) <= 1,
                                                name=f'node_{j}')
        # objective
        self.model.setObjective(
            ORDER_INCOME_UNIT * sum(self.esd_list) + gp.quicksum(self.profit_pool[j] * x[j] for j in range(num_routes)),
            gp.GRB.MAXIMIZE)

        self.model.update()

        self.model_x = x
        self.model_veh_cons = veh_cons
        self.model_node_cons = node_cons

    def add_station_visit_constr(self, station: int, visit: int):
        """add station visit constraint to the model"""
        if visit == 1:
            if self.model_node_visit_cons is None:
                self.model_node_visit_cons = {}
            self.model_node_visit_cons[station-1] = self.model.addConstr(gp.quicksum(
                self.node_mat[station-1, k] * self.model_x[k] for k in range(len(self.model_x))) >= 1,
                                                                         name=f'node_visit_{station-1}')
            self.model.update()
        elif visit == 0:
            del_con_name = f'node_{station-1}'
            # remove by name
            self.model.remove(self.model.getConstrByName(del_con_name))
            # add it back
            self.model_node_cons[station-1] = self.model.addConstr(gp.quicksum(
                self.node_mat[station-1, k] * self.model_x[k] for k in range(len(self.model_x))) <= 0,
                                                                   name=f'node_{station-1}')
            self.model.update()
        else:
            assert False, f'visit value {visit} is not valid. It should be 0 or 1.'

    def add_columns(self, column_pool: list, column_profit: list):
        """add latest generated columns to the model"""
        ex_veh_mat, ex_node_mat = None, None
        for van in range(self.num_veh):
            van_column = column_pool[van]
            van_profit = column_profit[van]
            for i in range(len(van_profit)):
                # update problem data
                self.route_pool.append(list(van_column[i]))
                self.profit_pool.append(van_profit[i])
                tmp_veh_mat = np.zeros((self.num_veh, 1))
                tmp_veh_mat[van, 0] = 1
                ex_veh_mat = tmp_veh_mat if ex_veh_mat is None else np.hstack((ex_veh_mat, tmp_veh_mat))
                tmp_node_mat = np.zeros((self.num_stations, 1))
                for val in van_column[i]:
                    if val != 0:
                        tmp_node_mat[val - 1, 0] = 1
                ex_node_mat = tmp_node_mat if ex_node_mat is None else np.hstack((ex_node_mat, tmp_node_mat))

                # update model
                col = gp.Column()
                for v in range(self.num_veh):
                    col.addTerms(tmp_veh_mat[v, 0], self.model_veh_cons[v])
                for n in range(self.num_stations):
                    col.addTerms(tmp_node_mat[n, 0], self.model_node_cons[n])
                if self.model_node_visit_cons is not None:
                    for k in self.model_node_visit_cons.keys():
                        col.addTerms(tmp_node_mat[k, 0], self.model_node_visit_cons[k])
                self.model_x[len(self.route_pool)-1] = \
                    self.model.addVar(obj=van_profit[i], vtype=gp.GRB.BINARY, name=f'x{len(self.profit_pool)-1}',
                                      column=col)

        self.veh_mat = np.hstack((self.veh_mat, ex_veh_mat))
        self.node_mat = np.hstack((self.node_mat, ex_node_mat))

        self.model.update()

    def relax_optimize(self):
        self.relax_model = self.model.relax()
        self.relax_model.setParam('OutputFlag', 0)
        self.relax_model.optimize()

    def integer_optimize(self):
        self.model.setParam('OutputFlag', 0)
        self.model.optimize()

    def get_integer_solution(self):
        return [self.model.getVarByName(f'x{j}').x for j in range(len(self.route_pool))]

    def get_relax_solution(self):
        return [self.relax_model.getVarByName(f'x{j}').x for j in range(len(self.route_pool))]

    def get_dual_vector(self) -> tuple:
        """Return the dual vector of the model constraints"""
        veh_con_names = [con.ConstrName for con in self.model_veh_cons.values()]
        veh_con_duals = [self.relax_model.getConstrByName(con_name).Pi for con_name in veh_con_names]
        node_con_names = [con.ConstrName for con in self.model_node_cons.values()]
        node_con_duals = [self.relax_model.getConstrByName(con_name).Pi for con_name in node_con_names]
        if self.model_node_visit_cons is not None:
            visit_con_keys = list(self.model_node_visit_cons.keys())
            visit_con_names = [con.ConstrName for con in self.model_node_visit_cons.values()]
            visit_con_duals = [self.relax_model.getConstrByName(con_name).Pi for con_name in visit_con_names]
        else:
            visit_con_duals = []
            visit_con_keys = []
        dual_vector = veh_con_duals + node_con_duals + visit_con_duals
        return dual_vector, visit_con_keys

    def get_relax_veh_vars(self):
        relax_sol = self.get_relax_solution()
        return [sum(self.veh_mat[van, :] * np.array(relax_sol)) for van in range(self.num_veh)]

    def get_relax_station_vars(self):
        relax_sol = self.get_relax_solution()
        return [sum(self.node_mat[station, :] * np.array(relax_sol)) for station in range(self.num_stations)]

    def get_non_zero_routes(self, model='relax') -> dict:
        """get the routes with non-zero values in the solution"""
        non_zero_routes_dict = {'route': [[] for _ in range(self.num_veh)], 'profit': [[] for _ in range(self.num_veh)]}
        if model == 'both':
            int_sol = self.get_integer_solution()
            relax_sol = self.get_relax_solution()
            int_routes_idx = [j for j in range(len(self.route_pool)) if int_sol[j] > 1e-5]
            relax_routes_idx = [j for j in range(len(self.route_pool)) if relax_sol[j] > 1e-5]
            routes_idx = list(set(int_routes_idx + relax_routes_idx))
            routes_belong = [np.where(self.veh_mat[:, k] == 1)[0][0] for k in routes_idx]
            for k, j in zip(routes_belong, routes_idx):
                non_zero_routes_dict['route'][k].append(self.route_pool[j])
                non_zero_routes_dict['profit'][k].append(self.profit_pool[j])
        else:
            assert model in ['integer', 'relax']
            if model == 'integer':
                sol = self.get_integer_solution()
            else:
                sol = self.get_relax_solution()
            routes_idx = [j for j in range(len(self.route_pool)) if sol[j] > 1e-5]
            routes_belong = [np.where(self.veh_mat[:, k] == 1)[0][0] for k in routes_idx]
            for k, j in zip(routes_belong, routes_idx):
                non_zero_routes_dict['route'][k].append(self.route_pool[j])
                non_zero_routes_dict['profit'][k].append(self.profit_pool[j])

        return non_zero_routes_dict


class HeuristicProblem:
    """
    Model for heuristic lower bound problem
    """
    def __init__(self, num_of_van: int, route_pool: list, profit_pool: list, veh_mat: np.ndarray, node_mat: np.ndarray,
                 model: gp.Model, cg_column_pool: list, cg_profit_pool: list):
        self.num_of_van = num_of_van
        self.route_pool = copy.deepcopy(route_pool)
        self.profit_pool = copy.deepcopy(profit_pool)
        self.veh_mat = copy.deepcopy(veh_mat)
        self.node_mat = copy.deepcopy(node_mat)
        self.model = model.copy()
        self.cg_column_pool = copy.deepcopy(cg_column_pool)
        self.cg_profit_pool = copy.deepcopy(cg_profit_pool)

    def solve(self) -> float:
        """solve the heuristic problem"""
        ex_veh_mat, ex_node_mat = None, None
        for van in range(self.num_of_van):
            van_column = self.cg_column_pool[van]
            van_profit = self.cg_profit_pool[van]
            for i in range(len(van_profit)):
                # update problem data
                self.route_pool.append(list(van_column[i]))
                self.profit_pool.append(van_profit[i])
                tmp_veh_mat = np.zeros((self.num_of_van, 1))
                tmp_veh_mat[van, 0] = 1
                ex_veh_mat = tmp_veh_mat if ex_veh_mat is None else np.hstack((ex_veh_mat, tmp_veh_mat))
                tmp_node_mat = np.zeros((self.node_mat.shape[0], 1))
                for val in van_column[i]:
                    if val != 0:
                        tmp_node_mat[val - 1, 0] = 1
                ex_node_mat = tmp_node_mat if ex_node_mat is None else np.hstack((ex_node_mat, tmp_node_mat))

                # update model
                col = gp.Column()
                for v in range(self.num_of_van):
                    col.addTerms(tmp_veh_mat[v, 0], self.model.getConstrByName(f'veh_{v}'))
                for n in range(self.node_mat.shape[0]):
                    col.addTerms(tmp_node_mat[n, 0], self.model.getConstrByName(f'node_{n}'))
                self.model.addVar(obj=van_profit[i], vtype=gp.GRB.BINARY, name=f'x{len(self.profit_pool)-1}', column=col)
        self.model.update()

        self.veh_mat = np.hstack((self.veh_mat, ex_veh_mat))
        self.node_mat = np.hstack((self.node_mat, ex_node_mat))

        self.model.setParam('OutputFlag', 0)
        self.model.optimize()
        return self.model.objVal

    def get_integer_solution(self):
        return [self.model.getVarByName(f'x{j}').x for j in range(len(self.profit_pool))]

    def obtain_non_zero_routes(self) -> dict:
        """obtain the routes with non-zero values in the solution"""
        non_zero_routes_dict = {'route': [[] for _ in range(self.num_of_van)],
                                'profit': [[] for _ in range(self.num_of_van)]}
        sol = self.get_integer_solution()
        routes_idx = [j for j in range(len(self.route_pool)) if sol[j] > 1e-5]
        routes_belong = [np.where(self.veh_mat[:, k] == 1)[0][0] for k in routes_idx]
        for k, j in zip(routes_belong, routes_idx):
            non_zero_routes_dict['route'][k].append(self.route_pool[j])
            non_zero_routes_dict['profit'][k].append(self.profit_pool[j])
        return non_zero_routes_dict
