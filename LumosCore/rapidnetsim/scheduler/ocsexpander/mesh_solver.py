import math

from ortools.linear_solver import pywraplp
import numpy as np


def solve(spine_num_per_pod, spine_up_port_num, c_ij: np.ndarray, u_ijkt: np.ndarray):
    pod_num = c_ij.shape[0]

    solver = pywraplp.Solver.CreateSolver('GUROBI')

    # 设置模型变量
    x_ijkt = np.empty((pod_num, pod_num, spine_up_port_num, spine_num_per_pod), dtype=pywraplp.Variable)
    for i in range(pod_num):
        for j in range(pod_num):
            for k in range(spine_up_port_num):
                for t in range(spine_num_per_pod):
                    x_ijkt[i, j, k, t] = solver.IntVar(0, 1, f'x_{i}_{j}_{k}_{t}')
    h_ijkt = np.empty((pod_num, pod_num, spine_up_port_num, spine_num_per_pod), dtype=pywraplp.Variable)
    for i in range(pod_num):
        for j in range(pod_num):
            for k in range(spine_up_port_num):
                for t in range(spine_num_per_pod):
                    h_ijkt[i, j, k, t] = solver.IntVar(0, spine_up_port_num, f'h_{i}_{j}_{k}_{t}')

    # 设置约束
    # 1. x_ijkt对k和t求和等于c_ij
    for i in range(pod_num):
        for j in range(pod_num):
            solver.Add(solver.Sum(x_ijkt[i, j, :, :].ravel().tolist()) == c_ij[i, j])
    # 2. x_ijkt对j和k求和等于spine_up_port_num
    for i in range(pod_num):
        for t in range(spine_num_per_pod):
            solver.Add(solver.Sum(x_ijkt[i, :, :, t].ravel().tolist()) == spine_up_port_num)
    # 3. x_ijkt对i和k求和等于spine_num_per_pod
    for j in range(pod_num):
        for t in range(spine_num_per_pod):
            solver.Add(solver.Sum(x_ijkt[:, j, :, t].ravel().tolist()) == spine_up_port_num)
    # 4. 每个spine连接其他pod的端口数量均匀分布
    for i in range(pod_num):
        for j in range(pod_num):
            for t in range(spine_num_per_pod):
                solver.Add(solver.Sum(x_ijkt[i, j, :, t].tolist()) <= math.ceil(c_ij[i, j] / spine_num_per_pod))
                solver.Add(solver.Sum(x_ijkt[i, j, :, t].tolist()) >= math.floor(c_ij[i, j] / spine_num_per_pod))
    # 5. h_ijkt >= x_ijkt - u_ijkt
    for i in range(pod_num):
        for j in range(pod_num):
            for k in range(spine_up_port_num):
                for t in range(spine_num_per_pod):
                    solver.Add(h_ijkt[i, j, k, t] >= x_ijkt[i, j, k, t] - u_ijkt[i, j, k, t])

    # 设置目标函数最小化h_ijkt
    obj = solver.Sum(h_ijkt.ravel().tolist())
    solver.Minimize(obj)

    # 模型求解
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL and status != pywraplp.Solver.FEASIBLE:
        raise RuntimeError('The problem does not have solution.')

    # 获取结果
    def get_solution_variable(x):
        return x.solution_value()
    get_solution_variable = np.vectorize(get_solution_variable)
    x_ijkt_solution = get_solution_variable(x_ijkt)
    return x_ijkt_solution


if __name__ == '__main__':
    import sys
    import time
    # open('time.log', 'w').close()
    spine_num_per_pod = int(sys.argv[1])
    spine_up_port_num = int(sys.argv[2])
    c_ij = np.load(sys.argv[3])
    c_ij = c_ij + c_ij.T
    print(c_ij)
    u_ijkt = np.load(sys.argv[4])
    start = time.time()
    x_ijkt_solution = solve(spine_num_per_pod, spine_up_port_num, c_ij, u_ijkt)
    end = time.time()
    with open('time.log', 'a') as f:
        f.write(f'{end - start}\n')
    print(end - start)
