import numpy as np
from ortools.graph.python import min_cost_flow


def solve(cluster_pod_num, spine_up_port_num, T_a_b, init_T_a_b=None):
    # 输入检查
    pod_num = cluster_pod_num
    if not isinstance(T_a_b, np.ndarray):
        T_a_b = np.array(T_a_b)
    assert T_a_b.shape == (pod_num, pod_num)
    pod_up_port_num = spine_up_port_num
    # print("debug T_a_b")
    # print(T_a_b)
    # print(pod_up_port_num,pod_num)

    arc_capacity_k0_map = {}
    arc_cost_k0_map = {}
    for i in range(pod_num):
        for j in range(pod_num):
            u = T_a_b[i, j]
            c = pod_up_port_num
            if i == j:
                arc_capacity_k0_map[f'{i}_{j}'] = [0]
                arc_cost_k0_map[f'{i}_{j}'] = [0]
            elif u < c:
                arc_capacity_k0_map[f'{i}_{j}'] = [u, c - u]
                arc_cost_k0_map[f'{i}_{j}'] = [-1, 1]
            else:
                arc_capacity_k0_map[f'{i}_{j}'] = [u]
                arc_cost_k0_map[f'{i}_{j}'] = [-1]

    start_nodes = []  # 在图中，supply node为0->pod_num*spine_num_per_pod - 1
    end_nodes = []
    capacities = []
    unit_costs = []

    supplies_map = {}

    global_arc_id = 0
    i_j_t_localarc_2_globalarc_map = {}

    for i in range(pod_num):
        for j in range(pod_num):
            supply_node_id_k0 = i
            end_node_id_k0 = pod_num + j
            for arc_id in range(len(arc_capacity_k0_map[f'{i}_{j}'])):
                if arc_capacity_k0_map[f'{i}_{j}'][arc_id] > 0:
                    start_nodes.append(supply_node_id_k0)
                    end_nodes.append(end_node_id_k0)
                    capacities.append(arc_capacity_k0_map[f'{i}_{j}'][arc_id])
                    unit_costs.append(arc_cost_k0_map[f'{i}_{j}'][arc_id])
                    i_j_t_localarc_2_globalarc_map[f'{i}_{j}_{arc_id}_0'] = global_arc_id
                    global_arc_id += 1

    supplies = []
    for i in range(2 * pod_num):
        if i < pod_num:
            supplies.append(pod_up_port_num)  # 求x_ij0t
        else:
            supplies.append(-1 * pod_up_port_num)

    for i in range(pod_num):
        for j in range(pod_num):
            supplies[i] -= init_T_a_b[i, j]
            supplies[j + pod_num] += init_T_a_b[i, j]

    min_cost_flow_ = min_cost_flow.SimpleMinCostFlow()

    # Add each arc.
    for i in range(0, len(start_nodes)):
        min_cost_flow_.add_arcs_with_capacity_and_unit_cost(start_nodes[i], end_nodes[i],
                                                            capacities[i], unit_costs[i])

    # Add node supplies.
    for i in range(0, len(supplies)):
        min_cost_flow_.set_node_supply(i, supplies[i])

    if min_cost_flow_.solve() == min_cost_flow_.OPTIMAL:
        c_ij = np.zeros((pod_num, pod_num), dtype=int)
        for i in range(pod_num):
            for j in range(pod_num):
                for local_arc_id in range(len(arc_capacity_k0_map[f'{i}_{j}'])):
                    if arc_capacity_k0_map[f'{i}_{j}'][local_arc_id] > 0:
                        global_arc_id = i_j_t_localarc_2_globalarc_map[f'{i}_{j}_{local_arc_id}_0']
                        c_ij[i, j] += min_cost_flow_.flow(global_arc_id)
                c_ij[i, j] += init_T_a_b[i, j]
        # for i in range(pod_num):
        #     for j in range(pod_num):
        #         assert c_ij[i, j] <= pod_up_port_num
        # c_ij_copy = c_ij.copy()
        # for i in range(pod_num):
        #     if np.sum(c_ij_copy[i,:]) != pod_up_port_num:
        #         print("debug c_ij_copy",np.sum(c_ij_copy[i,:]),pod_up_port_num,np.sum(init_T_a_b[i,:]),supplies[i])
        #     assert np.sum(c_ij_copy[i,:]) == pod_up_port_num
        # c_ij_copy = c_ij.copy()
        # if len(init_T_a_b)>0:
        #     for i in range(pod_num):
        #         for j in range(pod_num):
        #             if c_ij[i,j]<init_T_a_b[i,j]:
        #                 print("debug TESolver")
        #                 print(i,j,c_ij[i,j],init_T_a_b[i,j],T_a_b[i,j])
        #             assert c_ij[i,j]>=init_T_a_b[i,j]
        return c_ij

    else:
        success_MCF = False
        assert success_MCF
        print('There was an issue with the min cost flow input.')
