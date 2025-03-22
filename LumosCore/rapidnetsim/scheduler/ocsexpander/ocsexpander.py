import time
import logging
from collections import defaultdict
import numpy as np
import os
import warnings
from copy import deepcopy

from rapidnetsim.scheduler.ocsexpander.cijt_solver import CijtSolver
from rapidnetsim.scheduler.ocsexpander.mcf_solver import MCFSolver
from rapidnetsim.scheduler.ocsexpander.routing_solver import RoutingSolver
from rapidnetsim.scheduler.ocsexpander import mesh_solver, TE_solver_lp, TE_solver, gpu_placement, divide_oxc_matrix,divide_oxc_matrix_mcf
from rapidnetsim.communication_strategy.all2all import All2All
from rapidnetsim.communication_strategy.ring import Ring
from rapidnetsim.core.event.RepairEvent import RepairEvent
from rapidnetsim.core.network_refresh import handle_task_finish_immediately

logging.basicConfig(level=logging.ERROR)


class OCSExpander:
    def __init__(self, ocs_reconfiguration=True):
        # static variable
        from rapidnetsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        self.ocs_num = infra.ocs_num
        self.pod_num = infra.pod_num
        self.spine_num_per_pod = infra.spine_num_per_pod
        self.spine_up_port_num = infra.spine_up_port_num
        self.gpu_per_server = infra.NIC_num_in_a_server
        self.server_num_per_pod = infra.server_num_per_pod
        self.server_per_leaf = infra.server_per_leaf
        self.gpu_per_pod = self.gpu_per_server * self.server_num_per_pod
        self.gpu_per_leaf = self.gpu_per_server * self.server_per_leaf
        self.TP = self.gpu_per_server
        self.leaf_num_per_pod = infra.leaf_num_per_pod
        self.spine_oxc_link_num = 1
        self.total_leaf_num = infra.leaf_switch_num
        # dynamical variable
        self.T_a_b = np.zeros((self.pod_num, self.pod_num), dtype=int)
        # self.cur_real_link_demand = np.zeros((self.pod_num, self.pod_num), dtype=int)
        self.u_ijkt = np.zeros((self.pod_num, self.pod_num, self.spine_up_port_num, self.spine_num_per_pod), dtype=int)
        self.translate_link_new(self.u_ijkt)
        self.job_flow_demand_map = {}
        self.allocated_link_mapping = None
        self.ocs_reconfiguration = ocs_reconfiguration

        self.global_leaf_comm = np.zeros((self.total_leaf_num, self.total_leaf_num), dtype=int)
        self.scaled_global_leaf_comm = deepcopy(self.global_leaf_comm)
        # self.routing_solver = RoutingSolver(self.pod_num, self.spine_num_per_pod)
        self.gpu_placement_scheduler = gpu_placement.GPUPlacement()

        self.checkpoint_time = 60
        self.job_start_time_map = {}
        self.failure_id_gpu_status_map = {}

    def init_mesh_topo(self):
        if os.path.exists('x_ijkt.npy'):
            x_ijkt = np.load('x_ijkt.npy')
        else:
            c_ij = TE_solver_lp.solve(self.pod_num, self.spine_num_per_pod, self.spine_up_port_num,
                                      np.ones_like(self.T_a_b))
            x_ijkt = mesh_solver.solve(self.spine_num_per_pod, self.spine_up_port_num, c_ij)
            np.save('x_ijkt.npy', x_ijkt)
        # self.check_x_ijkt(x_ijkt)
        self.u_ijkt = deepcopy(x_ijkt)
        self.allocated_link_mapping = self.translate_link_new(x_ijkt)

    def schedule(self, TP, DP, PP, EP, task_id):
        from rapidnetsim.core.simulator import Simulator
        Simulator.task_has_computation_time[task_id] = 0.
        Simulator.task_has_communication_size[task_id] = 0.
        Simulator.task_need_comp_time[task_id] = 0.
        Simulator.task_need_comm_size[task_id] = 0.
        Simulator.task_expected_comm_time[task_id] = 0.
        Simulator.task_actual_comm_time[task_id] = 0.
        # base_time = time.time()
        print(f'task {task_id} with gpu size {TP * DP * PP} and EP is {EP}')

        # Step 1 gpu placement in each pod
        require_server_num = TP * DP * PP // self.gpu_per_server
        pod_used_server_list, job_gpu_used_list = self.gpu_placement_scheduler.occupy_resource(require_server_num, EP,
                                                                                               task_id)
        if len(pod_used_server_list) == 0:
            return False, None, None, None, None
        # print("stage 1 cost ", time.time() - base_time)

        # base_time = time.time()
        # Step 2 choose gpu in each pod
        # job_gpu_used_list_print_str = []
        # for index, val in np.ndenumerate(job_gpu_used_list):
        #     if val == 1:
        #         job_gpu_used_list_print_str.append(index)
        # job_gpu_used_list_print_str.sort()
        # with open('./gpu_utilization.txt', 'a') as f:
        #     f.write(str(taskid) + ' ' + ','.join(map(str, job_gpu_used_list_print_str)) + '\n')
        # print("debug server")
        # print(pod_used_server_list)
        # print(np.sum(job_gpu_used_list))
        # print("stage 2 cost ", time.time() - base_time)

        # base_time = time.time()
        # Step 3 intra leaf flow demand generator
        dp_pp_array, ep_array = self.generate_flow_demand(pod_used_server_list, TP, DP, PP, EP, job_gpu_used_list)
        ep_qp, pp_qp, dp_qp = self.generate_qp(ep_array, dp_pp_array)
        # print("stage 3 cost ", time.time() - base_time)

        if self.ocs_reconfiguration is False:
            return True, ep_qp, pp_qp, dp_qp, self.allocated_link_mapping

        # base_time = time.time()
        # Step 4 calculate global intra leaf flow demand
        if Simulator.CONF_DICT['rail_optimized'] == 'no':
            self.job_flow_demand_map[task_id] = self.generate_leaf_communication(pp_qp, dp_qp)
            self.global_leaf_comm += self.job_flow_demand_map[task_id]
            leaf_ij_copy = deepcopy(self.global_leaf_comm)
        else:
            self.job_flow_demand_map[task_id] = self.generate_leaf_communication_rail_optimized(pp_qp, dp_qp)
            self.global_leaf_comm += self.job_flow_demand_map[task_id]
            self.scaled_global_leaf_comm = self.scaling_leaf_comm()
            leaf_ij_copy = deepcopy(self.scaled_global_leaf_comm)
        # Step 4.2 use c++ to calculate the intra spine flow demand
        leaf_spine_link_num = Simulator.get_infrastructure().leaf_spine_link_num
        if int(Simulator.CONF_DICT['layers']) == '2':
            leaf_ij_copy, leaf_ij_copy_T = divide_oxc_matrix_mcf.solve(leaf_ij_copy, self.total_leaf_num)
        else:
            leaf_ij_copy, leaf_ij_copy_T = divide_oxc_matrix.solve(leaf_ij_copy, self.total_leaf_num)
        base_time = time.time()
        solver = CijtSolver(leaf_ij_copy, self.spine_num_per_pod, self.pod_num * self.leaf_num_per_pod)
        l_abt = solver.solve()
        print("spine balance cost ", time.time() - base_time)
        for t in range(self.spine_num_per_pod):
            l_abt[:, :, t] += l_abt[:, :, t].T
        # Step 4.3 do cross-link and TE-solver
        c_ijt = self.generate_link_demand(l_abt)  # 生成leaf demand
        # print("stage 4 cost ", time.time() - base_time)
        # c_ij = np.sum(c_ijt, axis=2)
        # np.save(f'c_ij_{task_id}.npy', c_ij)
        # np.save(f'u_ijkt_{task_id}.npy', self.u_ijkt)

        # base_time = time.time()
        # Step 5.1 generate reconfiguration
        x_ijkt = self.generate_ocs_configuration(c_ijt)
        # tmp_cij = np.sum(c_ijt,axis=2)
        # tmp_l_abt = np.sum(l_abt,axis=2)
        x_ij = np.sum(x_ijkt, axis=(2, 3))
        print("x_ij\n", x_ij)
        # print("30,27", x_ij[30,27], x_ij[30,27])
        # print("30,27", tmp_cij[30,27],tmp_cij[27,30])
        # print("self.global_leaf_comm,247,216",self.global_leaf_comm[247,216])
        # print("self.tmp_l_abt,247,216",tmp_l_abt[247,216])
        # change_link = np.zeros((self.pod_num, self.pod_num, self.spine_up_port_num, self.spine_num_per_pod), dtype=int)
        # change_link_mask = (x_ijkt > self.u_ijkt) & (
        #         np.arange(self.pod_num)[:, None, None, None] != np.arange(self.pod_num)[None, :, None, None])
        # change_link[change_link_mask] = x_ijkt[change_link_mask] - self.u_ijkt[change_link_mask]
        # print("change link num", np.sum(change_link))
        # print("stage 5 cost ", time.time() - base_time)

        # base_time = time.time()
        self.u_ijkt = deepcopy(x_ijkt)
        allocated_link_mapping = self.translate_link_new(self.u_ijkt)
        self.allocated_link_mapping = allocated_link_mapping
        # print("stage 6 cost ", time.time() - base_time)
        self.job_start_time_map[task_id] = Simulator.get_current_time()
        return True, ep_qp, pp_qp, dp_qp, allocated_link_mapping

    @staticmethod
    def generate_qp(ep_array, dp_pp_array):
        """
        根据输入生成EP、PP、DP的QP。
        :param ep_array: EP的GPU分配
        :param dp_pp_array: DP和PP的GPU分配
        :return: EP、PP、DP的QP。其中EP是个二维数组，第一个维度是round，代表每个stage的所有EP域的通信对。
        PP是一个二维数组，表示一轮训练的所有前向和反向过程（按前向、反向的顺序）。第一个维度是round，代表当前阶段的所有PP域的通信对。
        DP是一个二维数组，表示一轮训练的所有DP通信对。第一个维度是从上到下的每个DP域（前向传播中的PP顺序），
        第二个维度代表DP通信的一个round。因为每个DP round都是相同的，所以这里省略。
        """
        # 生成EP的QP
        ep_nums, ep_size = ep_array.shape[0], ep_array.shape[1]
        if ep_size == 8:
            ep_qp = []
        else:
            ep_qp = [[] for _ in range(ep_size // 8 - 1)]
            for i in range(ep_nums):
                gpu_list = ep_array[i, :]
                # 分平面通信
                for j in range(8):
                    gpu_list_face = gpu_list[j:][::8]
                    # 这里communication_size设置成0原因是为了复用All2All的代码，只是为了生成QP，不需要真实的communication_size
                    ep_round_pairs = All2All.get_pairwise_every_round_pair(len(gpu_list_face), 0)
                    for round_id in range(len(ep_round_pairs)):
                        for pair in ep_round_pairs[round_id]:
                            ep_qp[round_id].append((gpu_list_face[pair[0]], gpu_list_face[pair[1]]))

        # 生成PP的QP。按照先前向，后反向的顺序生成QP，所以一共包含2*(PP-1)个round
        PP, DP, TP = dp_pp_array.shape
        pp_qp = []
        for i in range(PP - 1):
            pp = []
            for j in range(DP):
                for k in range(TP):
                    pp.append((dp_pp_array[i, j, k], dp_pp_array[i + 1, j, k]))
            pp_qp.append(pp)
        for i in range(PP - 1, 0, -1):
            pp = []
            for j in range(DP):
                for k in range(TP):
                    pp.append((dp_pp_array[i, j, k], dp_pp_array[i - 1, j, k]))
            pp_qp.append(pp)

        # 生成DP的QP
        if DP == 1:
            return ep_qp, pp_qp, []
        dp_qp_pairs = Ring.get_ring_every_round_pair(DP, 0)[0]
        qp_num = len(dp_qp_pairs)
        reverse_dp_round_pairs = dp_qp_pairs[qp_num // 2:] + dp_qp_pairs[:qp_num // 2]
        from rapidnetsim.core.simulator import Simulator
        dp_qp = [[] for _ in range(PP)]
        if Simulator.CONF_DICT['rail_optimized'] == 'yes':
            for i in range(PP):
                for j in range(TP):
                    dp_qp[i].extend([(dp_pp_array[i, pair[0], j], dp_pp_array[i, pair[1], j])
                                     for pair in dp_qp_pairs[:qp_num]])
                for j in range(TP):
                    dp_qp[i].extend([(dp_pp_array[i, pair[0], j], dp_pp_array[i, pair[1], j])
                                     for pair in reverse_dp_round_pairs[:qp_num]])
        else:
            for i in range(PP):
                for j in range(TP // 2):
                    dp_qp[i].extend([(dp_pp_array[i, pair[0], j], dp_pp_array[i, pair[1], j])
                                     for pair in dp_qp_pairs[:qp_num // 2]])
                for j in range(TP // 2, TP):
                    dp_qp[i].extend([(dp_pp_array[i, pair[0], j], dp_pp_array[i, pair[1], j])
                                     for pair in reverse_dp_round_pairs[:qp_num // 2]])
        return ep_qp, pp_qp, dp_qp

    @staticmethod
    def generate_flow_demand(pod_used_server_list, TP, DP, PP, EP, job_gpu_used_list):
        # remain_job_gpu_used_list = deepcopy(job_gpu_used_list)
        from rapidnetsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        pod_num = infra.pod_num
        server_num_per_pod = infra.server_num_per_pod
        server_per_leaf = infra.server_per_leaf
        gpu_per_server = infra.NIC_num_in_a_server
        pod_server_pair = []
        for pod_id in range(pod_num):
            if pod_used_server_list[pod_id] > 0:
                pod_server_pair.append((pod_id, pod_used_server_list[pod_id]))
        pod_server_pair.sort(key=lambda x: x[1])
        PP_DP_domain_matrix = np.zeros((PP, DP), dtype=int)
        down_flag = True

        curr_DP = 0
        curr_PP = 0
        for pod_id, server_num in pod_server_pair:
            for _ in range(server_num):
                PP_DP_domain_matrix[curr_PP, curr_DP] = pod_id
                if down_flag:
                    curr_PP += 1
                else:
                    curr_PP -= 1
                if curr_PP == PP:
                    curr_PP = PP - 1
                    curr_DP += 1
                    down_flag = False
                elif curr_PP == -1:
                    curr_PP = 0
                    curr_DP += 1
                    down_flag = True

        gpu_num_per_leaf = gpu_per_server * server_per_leaf
        gpu_num_per_pod = gpu_per_server * server_num_per_pod
        # 构建一个从TP到server_list的映射
        # job_gpu_used_list标注了该任务在每个Pod，每个leaf中用到的GPU
        dp_pp_array = np.zeros((PP, DP, TP), dtype=int)
        select_gpus = np.where(job_gpu_used_list == 1)
        tmp_pod_id, tmp_leaf_id, k = select_gpus
        gpu_indices = np.array(select_gpus, dtype=int).T
        num_selected_gpus = len(gpu_indices)
        global_gpu_ids = np.zeros(num_selected_gpus, dtype=int)
        global_gpu_ids += gpu_indices[:, 0] * gpu_num_per_pod + gpu_indices[:, 1] * gpu_num_per_leaf + gpu_indices[:, 2]

        select_pods = [pair[0] for pair in pod_server_pair]
        gpu_indices = {pod: gpu_indices[tmp_pod_id == pod, :].tolist() for pod in select_pods}
        for pod in select_pods:
            gpu_indices[pod].reverse()

        for (curr_PP, curr_DP), chosen_pod_id in np.ndenumerate(PP_DP_domain_matrix):
            select_gpu_indices = []
            for _ in range(TP):
                index = gpu_indices[chosen_pod_id].pop()
                select_gpu_indices.append(index)
            select_gpu_indices = np.array(select_gpu_indices, dtype=int)
            select_gpu_ids = np.zeros(TP, dtype=int)
            select_gpu_ids += select_gpu_indices[:, 0] * gpu_num_per_pod + \
                              select_gpu_indices[:, 1] * gpu_num_per_leaf + select_gpu_indices[:, 2]
            dp_pp_array[curr_PP, curr_DP, :] = select_gpu_ids
        assert not len(np.where(dp_pp_array == 0)[0]) >= 8

        # 生成EP流量
        EP_domain_num = num_selected_gpus // EP
        ep_array = np.zeros((EP_domain_num, EP), dtype=int)
        if EP <= TP:
            return dp_pp_array, ep_array

        EP_ids = np.arange(num_selected_gpus) // EP
        gpu_indices_within_ep = np.arange(num_selected_gpus) % EP
        ep_array[EP_ids, gpu_indices_within_ep] = global_gpu_ids
        ep_array_pod = ep_array // gpu_num_per_pod
        if not np.all(ep_array_pod == ep_array_pod[:, [0]]):
            warning_text = "EP communication is not in the same pod. The task info is:\n"
            warning_text += f"TP: {TP}, DP: {DP}, PP: {PP}, EP: {EP}\n"
            warning_text += f"pod_server_pair: {pod_server_pair}\n"
            warning_text += f"PP_DP_domain_matrix: {PP_DP_domain_matrix}\n"
            warning_text += f"job_gpu_used_list: {job_gpu_used_list}\n"
            warning_text += f"dp_pp_array: {dp_pp_array}\n"
            warning_text += f"ep_array: {ep_array}\n"
            warning_text += f"ep_array_pod: {ep_array_pod}\n"
            warnings.warn("EP communication is not in the same pod.")
            print(warning_text)
        return dp_pp_array, ep_array

    def generate_link_demand(self, l_abt):
        from rapidnetsim.core.simulator import Simulator
        print("start calculate init spine communication")
        init_C_ijt = np.zeros((self.pod_num, self.pod_num, self.spine_num_per_pod), dtype=int)
        for start_leaf in range(self.total_leaf_num):
            for end_leaf in range(self.total_leaf_num):
                if np.sum(l_abt[start_leaf, end_leaf, :]) == 0:
                    continue
                start_pod = start_leaf // self.leaf_num_per_pod
                end_pod = end_leaf // self.leaf_num_per_pod
                if start_pod == end_pod:
                    continue
                init_C_ijt[start_pod, end_pod, :] += l_abt[start_leaf, end_leaf, :]

        print("start TE_solver")
        c_ijt = self.calculate_max_min_fairness(init_C_ijt)
        print("start cross link")
        tmp_C_ijt = np.zeros((self.pod_num, self.pod_num, self.spine_num_per_pod), dtype=int)
        tmp_C_ijt_T = np.zeros((self.pod_num, self.pod_num, self.spine_num_per_pod), dtype=int)
        for t in range(self.spine_num_per_pod):
            tmp_res, tmp_res_T = divide_oxc_matrix.solve(c_ijt[:, :, t], self.pod_num)
            init_tmp_res, init_tmp_res_T = divide_oxc_matrix.solve(init_C_ijt[:, :, t], self.pod_num)
            tmp_res = TE_solver.solve(self.pod_num, self.spine_up_port_num // 2, tmp_res, init_tmp_res)
            # print("debug tmp_res", t)
            tmp_C_ijt[:, :, t] += tmp_res
            tmp_C_ijt_T[:, :, t] += tmp_res.T
            # print(tmp_res)
            # print(tmp_res.T)
            # print(tmp_C_ijt_T[:, :, t] + tmp_C_ijt[:, :, t])
        # c_ij_T = c_ijt.T
        # c_ijt += c_ij_T
        # print("c_ij:\n", c_ij, np.sum(c_ij-self.T_a_b))
        # print("debug tmp_C_ijt", tmp_C_ijt)
        return tmp_C_ijt

    def generate_ocs_configuration(self, c_ijt):
        start = time.time()
        # print("start mcf")
        oxc_list = list(range(self.ocs_num))
        m_solver = MCFSolver(self.pod_num, self.spine_num_per_pod, oxc_list, self.spine_oxc_link_num,
                             c_ijt, self.u_ijkt, self.spine_up_port_num)
        x_ijkt = m_solver.solve()
        end = time.time()
        # print("debug flow_demand_matrix")
        # for i in range(self.pod_num):
        #     for j in range(self.pod_num):
        #         if pp_flow_demand_matrix[i,j]>0:
        #             print("debug pp_flow_demand_matrix",i,j,pp_flow_demand_matrix[i,j])
        #         if dp_flow_demand_matrix[i,j]>0:
        #             print("debug dp_flow_demand_matrix",i,j,dp_flow_demand_matrix[i,j])
        print("mcf time cost:", end - start)
        return x_ijkt

    def translate_link(self, x_ijkt):
        from rapidnetsim.core.simulator import Simulator
        allocated_link_mapping = []

        total_port_num = int(Simulator.CONF_DICT['NIC_num'])
        max_leaf_size_each_layer = int(Simulator.CONF_DICT['leaf_switch_num'])
        port_per_leaf = int(Simulator.CONF_DICT['leaf_switch_port_num'])
        max_spine_size_each_layer = int(Simulator.CONF_DICT['spine_switch_num'])
        port_per_spine = int(Simulator.CONF_DICT['spine_switch_port_num'])
        final_port_per_pod = total_port_num // int(Simulator.CONF_DICT['pod_num'])
        tmp_pod_num = int(Simulator.CONF_DICT['NIC_num'])
        tmp_pod_port_size = total_port_num // tmp_pod_num
        stage_num = 0

        Simulator.clos_up_table = defaultdict(list)

        Simulator.clos_down_table = defaultdict(lambda: defaultdict(list))
        base_time = time.time()
        while stage_num < 2:
            tmp_pod_port_size *= port_per_spine
            tmp_pod_num = max(1, total_port_num // tmp_pod_port_size)  # 根据层数确定pod数量
            base_node_id = max(0, (stage_num - 1)) * max_spine_size_each_layer + min(1,
                                                                                     stage_num) * total_port_num  # 当前
            next_base_node_id = max(0, stage_num) * max_spine_size_each_layer + total_port_num
            if stage_num == 0:
                cur_layer_size = total_port_num
            else:
                cur_layer_size = max_spine_size_each_layer
            for tmp_node_id in range(cur_layer_size):
                node_per_pod = int(cur_layer_size) // tmp_pod_num  # 根据当前层+下一层能组成的pod大小，计算需要的pod数量
                spine_per_pod = int(max_spine_size_each_layer) // tmp_pod_num
                if stage_num == 0:
                    port_per_node = 1
                else:
                    port_per_node = port_per_spine
                if stage_num == 0:
                    tmp_start_node_id = tmp_node_id // node_per_pod
                else:
                    tmp_start_node_id = tmp_node_id // spine_per_pod * spine_per_pod + ((port_per_spine * (
                            tmp_node_id % spine_per_pod)) % spine_per_pod)  # 第几个pod的第一个node,pod内的第几个node
                for tmp_node_port in range(int(port_per_node)):
                    if 'is_two_iter' in Simulator.CONF_DICT and Simulator.CONF_DICT['is_two_iter'] == 'yes':
                        allocated_link_mapping.append(
                            [base_node_id + tmp_node_id, next_base_node_id + tmp_start_node_id + tmp_node_port,
                             self.spine_up_port_num])
                        allocated_link_mapping.append(
                            [next_base_node_id + tmp_start_node_id + tmp_node_port, base_node_id + tmp_node_id,
                             self.spine_up_port_num])
                    else:
                        allocated_link_mapping.append(
                            [base_node_id + tmp_node_id, next_base_node_id + tmp_start_node_id + tmp_node_port, 1])
                        allocated_link_mapping.append(
                            [next_base_node_id + tmp_start_node_id + tmp_node_port, base_node_id + tmp_node_id, 1])
                    # 当前节点在上行时连到哪些node
                    Simulator.clos_up_table[base_node_id + tmp_node_id].append(
                        next_base_node_id + tmp_start_node_id + tmp_node_port)
                    if stage_num == 0:
                        Simulator.clos_down_table[tmp_node_id][tmp_node_id].append(tmp_node_id)
                        Simulator.clos_down_table[next_base_node_id + tmp_start_node_id + tmp_node_port][
                            base_node_id + tmp_node_id].append(base_node_id + tmp_node_id)
                    else:
                        for potential_dst in range(total_port_num):
                            if potential_dst in Simulator.clos_down_table[base_node_id + tmp_node_id]:
                                Simulator.clos_down_table[
                                    next_base_node_id + tmp_start_node_id + tmp_node_port][potential_dst].append(
                                    base_node_id + tmp_node_id)

            stage_num += 1
        print("debug translate_link stage 1", time.time() - base_time)
        base_time = time.time()
        tmp_clos_down_connection_dst_map = deepcopy(Simulator.clos_down_table)
        # print("debug translate_link stage 2.1",time.time()-base_time)
        # base_time = time.time()
        # new_allocation_link_mapping = self.routing_solver.find_down_routing(
        #     Simulator.clos_down_table,
        #     tmp_clos_down_connection_dst_map,
        #     x_ijkt,
        #     total_port_num,
        #     max_leaf_size_each_layer
        # )
        # print("debug translate_link stage 2.2",time.time()-base_time)
        # base_time = time.time()
        # allocated_link_mapping.extend(new_allocation_link_mapping)

        for pod_i in range(self.pod_num):
            for pod_j in range(self.pod_num):
                for t in range(self.spine_num_per_pod):
                    start_spine = pod_i * self.spine_num_per_pod + t + total_port_num + max_leaf_size_each_layer
                    end_spine = pod_j * self.spine_num_per_pod + t + total_port_num + max_leaf_size_each_layer
                    link_num = np.sum(x_ijkt[pod_i, pod_j, :, t])
                    if link_num > 0:
                        allocated_link_mapping.append([start_spine, end_spine, link_num])
                        allocated_link_mapping.append([end_spine, start_spine, link_num])
                        assert start_spine % self.spine_num_per_pod == end_spine % self.spine_num_per_pod

                        for potential_dst in range(total_port_num):
                            if potential_dst in tmp_clos_down_connection_dst_map[start_spine]:
                                Simulator.clos_down_table[end_spine][potential_dst].append(start_spine)
                        for potential_dst in range(total_port_num):
                            if potential_dst in tmp_clos_down_connection_dst_map[end_spine]:
                                Simulator.clos_down_table[start_spine][potential_dst].append(end_spine)

        print("debug translate_link stage 2", time.time() - base_time)
        base_time = time.time()

        leaf_base_id = total_port_num
        spine_base_id = total_port_num + max_leaf_size_each_layer
        pod_leaf_size = int(Simulator.CONF_DICT['spine_switch_num']) // int(Simulator.CONF_DICT['pod_num'])
        for tmp_leaf_id in range(max_leaf_size_each_layer):
            tmp_start_node_id = tmp_leaf_id // spine_per_pod * spine_per_pod + (
                    (port_per_spine * (tmp_leaf_id % spine_per_pod)) % spine_per_pod)
            for tmp_node_port in range(int(port_per_node)):
                for potential_dst in range(total_port_num):
                    if potential_dst in Simulator.clos_down_table[
                        spine_base_id + tmp_start_node_id + tmp_node_port] and (
                            potential_dst // final_port_per_pod != tmp_leaf_id // pod_leaf_size):
                        if potential_dst not in Simulator.clos_down_table[leaf_base_id + tmp_leaf_id]:
                            Simulator.clos_down_table[leaf_base_id + tmp_leaf_id][potential_dst] = []
                        Simulator.clos_down_table[leaf_base_id + tmp_leaf_id][potential_dst].append(
                            spine_base_id + tmp_start_node_id + tmp_node_port)
        print("debug translate_link stage 3", time.time() - base_time)
        base_time = time.time()
        # assert False
        return allocated_link_mapping

    def translate_link_new(self, x_ijkt):
        from rapidnetsim.core.simulator import Simulator
        spine_num = self.spine_num_per_pod * self.pod_num
        nic_num = self.gpu_per_server * self.server_num_per_pod * self.pod_num
        try:
            leaf_spine_link_num = int(Simulator.CONF_DICT['leaf_spine_link_num'])
        except (KeyError, ValueError):
            leaf_spine_link_num = 1
        try:
            is_rail_optimized = Simulator.CONF_DICT['rail_optimized'] == 'yes'
        except KeyError:
            is_rail_optimized = False
        routing_solver = RoutingSolver(self.pod_num, spine_num, nic_num, self.server_num_per_pod,
                                       self.spine_up_port_num, leaf_spine_link_num, 1.0, is_rail_optimized)
        routing_solver.generate_routing_table(x_ijkt)
        Simulator.intra_pod_up_table = routing_solver.get_intra_pod_up_table()
        Simulator.intra_pod_down_table = routing_solver.get_intra_pod_down_table()
        Simulator.inter_pod_table = routing_solver.get_inter_pod_routing_table()
        Simulator.inter_pod_weighted_direct_table = routing_solver.get_inter_pod_weighted_direct_routing_table()
        Simulator.inter_pod_weighted_twohop_table = routing_solver.get_inter_pod_weighted_twohop_routing_table()
        allocated_link_mapping = routing_solver.get_connection_info_list()
        # print("debug x_ijkt")
        # print(x_ijkt[3,17,:,:])
        # # print(Simulator.intra_pod_up_table[0][0])
        # # print(Simulator.intra_pod_up_table[15][0])
        # # print(Simulator.intra_pod_up_table[31][0])
        # # print(Simulator.intra_pod_up_table[503][0])
        # # print(Simulator.intra_pod_up_table[504][0])
        # leaf_id_list = [4096,4097,4098,4159]
        # # for i in range(4096):
        # #         if Simulator.intra_pod_up_table[i][0] == 0:
        # #             print("debug connect",0,i)
        # for j in leaf_id_list:
        #     for i in range(4096):
        #         if Simulator.intra_pod_up_table[i][0] == j:
        #             print("debug connect",j,i)
        # assert False
        return allocated_link_mapping

    def generate_leaf_communication(self, pp_qp, dp_qp):
        from rapidnetsim.core.simulator import Simulator
        pp_flow, dp_flow = set(), set()
        for pp in pp_qp:
            for pair in pp:
                src_leaf, dst_leaf = pair[0] // self.gpu_per_leaf, pair[1] // self.gpu_per_leaf
                src_pod, dst_pod = src_leaf // self.leaf_num_per_pod, dst_leaf // self.leaf_num_per_pod
                if src_pod != dst_pod:
                    pp_flow.add((src_leaf, dst_leaf))
        for dp in dp_qp:
            for pair in dp:
                src_leaf, dst_leaf = pair[0] // self.gpu_per_leaf, pair[1] // self.gpu_per_leaf
                src_pod, dst_pod = src_leaf // self.leaf_num_per_pod, dst_leaf // self.leaf_num_per_pod
                if src_pod != dst_pod:
                    dp_flow.add((src_leaf, dst_leaf))

        leaf_comm = np.zeros((self.pod_num * self.leaf_num_per_pod, self.pod_num * self.leaf_num_per_pod), dtype=int)
        leaf_spine_link_num = Simulator.get_infrastructure().leaf_spine_link_num
        if leaf_spine_link_num == 1 and self.server_per_leaf == 1:
            for start_leaf, end_leaf in pp_flow:
                leaf_comm[start_leaf, end_leaf] += self.TP // 4
            for start_leaf, end_leaf in dp_flow:
                leaf_comm[start_leaf, end_leaf] += self.TP // 4
        else:
            for start_leaf, end_leaf in pp_flow:
                leaf_comm[start_leaf, end_leaf] += self.TP
            for start_leaf, end_leaf in dp_flow:
                leaf_comm[start_leaf, end_leaf] += self.TP // 2
        return leaf_comm

    def generate_leaf_communication_rail_optimized(self, pp_qp, dp_qp):
        from rapidnetsim.core.simulator import Simulator
        gpu_num = Simulator.get_infrastructure().NIC_num
        leaf_comm = np.zeros((self.pod_num * self.leaf_num_per_pod, self.pod_num * self.leaf_num_per_pod), dtype=int)
        intra_pod_up_table = Simulator.intra_pod_up_table
        for pp in pp_qp:
            for pair in pp:
                src_leaf, dst_leaf = intra_pod_up_table[pair[0]][0], intra_pod_up_table[pair[1]][0]
                src_leaf -= gpu_num
                dst_leaf -= gpu_num
                src_pod, dst_pod = src_leaf // self.leaf_num_per_pod, dst_leaf // self.leaf_num_per_pod
                if src_pod != dst_pod:
                    leaf_comm[src_leaf, dst_leaf] += 1
        prev_dp_pod_comm = set()
        for dp in dp_qp:
            curr_dp_pod_comm = set()
            for pair in dp:
                src_leaf, dst_leaf = intra_pod_up_table[pair[0]][0], intra_pod_up_table[pair[1]][0]
                src_leaf -= gpu_num
                dst_leaf -= gpu_num
                src_pod, dst_pod = src_leaf // self.leaf_num_per_pod, dst_leaf // self.leaf_num_per_pod
                if src_pod != dst_pod and (src_pod, dst_pod) not in prev_dp_pod_comm:
                    leaf_comm[src_leaf, dst_leaf] += 1
                    leaf_comm[dst_leaf, src_leaf] += 1
                    curr_dp_pod_comm.add((src_pod, dst_pod))
            prev_dp_pod_comm = curr_dp_pod_comm
        return leaf_comm

    def scaling_leaf_comm(self):
        from rapidnetsim.core.simulator import Simulator
        infra = Simulator.get_infrastructure()
        leaf_comm_sum = np.sum(self.global_leaf_comm, axis=0)
        leaf_up_port_num = infra.leaf_switch_port_num // 2
        filtered_indices = np.where(leaf_comm_sum > leaf_up_port_num)[0]
        arg_sort = filtered_indices[np.argsort(leaf_comm_sum[filtered_indices])]
        leaf_comm = np.triu(self.global_leaf_comm)
        for i in arg_sort:
            comm_sum = np.sum(leaf_comm[i, :]) + np.sum(leaf_comm[:, i])
            for j in range(i):
                if leaf_comm[j, i] > 0:
                    leaf_comm[j, i] = max(1, np.floor(leaf_comm[j, i] / comm_sum * leaf_up_port_num))
            for j in range(i + 1, self.total_leaf_num):
                if leaf_comm[i, j] > 0:
                    leaf_comm[i, j] = max(1, np.floor(leaf_comm[i, j] / comm_sum * leaf_up_port_num))
            new_comm_sum = np.sum(leaf_comm[i, :]) + np.sum(leaf_comm[:, i])
            temp_comm = np.zeros((self.total_leaf_num,), dtype=int)
            temp_comm[:i] = leaf_comm[:i, i]
            temp_comm[i + 1:] = leaf_comm[i, i + 1:]
            temp_arg_sort = np.argsort(temp_comm)
            temp_arg_sort = temp_arg_sort[::-1]
            if new_comm_sum > leaf_up_port_num:
                j = 0
                while new_comm_sum > leaf_up_port_num:
                    if temp_comm[temp_arg_sort[j]] > 1:
                        temp_comm[temp_arg_sort[j]] -= 1
                        new_comm_sum -= 1
                    j += 1
                    if j == self.total_leaf_num - 1:
                        j = 0
            assert new_comm_sum <= leaf_up_port_num
            # if new_comm_sum < leaf_up_port_num:
            #     j = 0
            #     while new_comm_sum < leaf_up_port_num:
            #         if leaf_comm[j, i] > 0:
            #             leaf_comm[j, i] += 1
            #             new_comm_sum += 1
            #         j += 1
            #     while new_comm_sum < leaf_up_port_num:
            #         if leaf_comm[i, j] > 0:
            #             leaf_comm[i, j] += 1
            #             new_comm_sum += 1
            #         j += 1
        leaf_comm = leaf_comm + leaf_comm.T
        assert np.all(np.sum(leaf_comm, axis=0) <= leaf_up_port_num)
        assert np.all(leaf_comm >= 0)
        return leaf_comm

    def update_finished_job(self, taskid, sim_time, waiting_list):
        """
        任务结束时，释放GPU资源，更新网络状态。当前版本中注释的内容(step2 & step3)是任务结束后仍然进行OCS重配置。
        """
        # Step 1 update demand
        if self.ocs_reconfiguration:
            self.global_leaf_comm -= self.job_flow_demand_map[taskid]
        # from rapidnetsim.core.simulator import Simulator
        # Simulator.task_has_computation_time[taskid] = 0
        # # Step 2 link demand generator
        # c_ij = self.generate_link_demand()
        # # Step 3.1 generate reconfiguration
        # x_ijkt, tmp_nothing1, tmp_nothing1 = self.generate_ocs_configuration(
        #     c_ij,
        #     np.zeros((self.pod_num, self.pod_num), dtype=int),
        #     np.zeros((self.pod_num, self.pod_num), dtype=int),
        #     [(-1, -1)]
        # )
        # self.u_ijkt = deepcopy(x_ijkt)
        # # Step 3.2 generate flow
        # allocation_link_mapping = self.translate_link(x_ijkt)

        # Step 4 release gpu
        self.gpu_placement_scheduler.release_resource(taskid)
        # return allocation_link_mapping

    def calculate_max_min_fairness(self, init_c_ijt):
        # c_ijt = deepcopy(init_c_ijt)
        c_ijt_copy = deepcopy(init_c_ijt)

        upper_triangular = np.triu(np.ones_like(c_ijt_copy, dtype=bool).transpose(2, 0, 1), 1).transpose((1, 2, 0))
        condition_mask = np.sum(c_ijt_copy, axis=1) < self.spine_num_per_pod * self.spine_up_port_num
        condition_mask = np.repeat(condition_mask[:, np.newaxis, :], self.pod_num, axis=1) & upper_triangular
        c_ijt_copy[condition_mask] += 1
        c_ijt_copy[condition_mask.transpose(1, 0, 2)] += 1

        # for t in range(self.spine_num_per_pod):
        #     for row_col_id in range(self.pod_num):
        #         for element_id in range(row_col_id + 1, self.pod_num):
        #             if np.sum(c_ijt[row_col_id, :, t]) < self.spine_num_per_pod * self.spine_up_port_num:
        #                 assert np.sum(c_ijt[row_col_id, :, t]) == np.sum(c_ijt[:, row_col_id, t])
        #                 c_ijt[row_col_id, element_id, t] += 1
        #                 c_ijt[element_id, row_col_id, t] += 1
        # assert np.all(c_ijt == c_ijt_copy)
        return c_ijt_copy

    def check_x_ijkt(self, x_ijkt: np.ndarray):
        pod_num = self.pod_num
        spine_num_per_pod = self.spine_num_per_pod
        spine_up_port_num = self.spine_up_port_num

        assert x_ijkt.shape == (pod_num, pod_num, spine_up_port_num, spine_num_per_pod), "x_ijkt shape does not match"
        x_ijt = np.sum(x_ijkt, axis=2)
        assert np.all(np.sum(x_ijt, axis=1) == spine_up_port_num), "spine's port should be fully used"

    def handle_failure_event(self, failure_id, duration_time):
        new_banned_gpu_status, new_banned_server_per_pod = self.random_fail_gpu(failure_id)
        if len(new_banned_gpu_status) > 0:
            self.failure_id_gpu_status_map[failure_id] = (new_banned_gpu_status, new_banned_server_per_pod)
            # 生成任务恢复事件
            from rapidnetsim.core.simulator import Simulator
            Simulator.register_event(
                RepairEvent(
                    duration_time,
                    failure_id,
                )
            )

    def random_fail_gpu(self, failure_id):
        # failure_id = -1*failure_id # gpus occupied by task i -1 is in failure
        new_banned_gpu_status, influenced_task_id, new_banned_server_per_pod = \
            self.gpu_placement_scheduler.random_fail_gpu(failure_id)
        if influenced_task_id != -1:
            # change remain model size of influenced_task
            from rapidnetsim.core.simulator import Simulator
            from rapidnetsim.task.waiting_task import WaitingTask
            Simulator.need_immediately_finish_task.append(influenced_task_id)
            handle_task_finish_immediately(influenced_task_id)
            # 任务相关流立即完成，同时任务再次进入等待队列
            task = Simulator.TASK_LIST[influenced_task_id]
            has_comp_ratio = Simulator.task_has_computation_time[influenced_task_id] / \
                                Simulator.task_need_comp_time[influenced_task_id]
            has_comm_ratio = Simulator.task_has_communication_size[influenced_task_id] / \
                                Simulator.task_need_comm_size[influenced_task_id]
            print(f'task {influenced_task_id} need comp {Simulator.task_need_comp_time[influenced_task_id]} '
                  f'need comm {Simulator.task_need_comm_size[influenced_task_id]}')
            print(f'task {influenced_task_id} has ratio {has_comp_ratio},{has_comm_ratio} '
                  f'has comp {Simulator.task_has_computation_time[influenced_task_id]} has comm '
                  f'{Simulator.task_has_communication_size[influenced_task_id]}')
            if has_comp_ratio > 1:
                print("debug has_comp_ratio", has_comp_ratio)
            assert has_comp_ratio <= 1.1
            if has_comm_ratio > 1:
                print("debug has_comp_ratio", has_comm_ratio)
            assert has_comm_ratio <= 1.1
            has_comp_ratio = min(has_comp_ratio, 0.9999)
            has_comm_ratio = min(has_comm_ratio, 0.9999)

            comm_time = task.duration_time - task.computation_time * task.computation_round
            remain_duration_time = (task.duration_time - has_comp_ratio * task.computation_time * task.computation_round
                                    - has_comm_ratio * comm_time)
            task.duration_time = remain_duration_time

            a_waiting_task = WaitingTask(Simulator.get_current_time(), task.model_size, task.gpu_num,
                                         Simulator.task_type_map[influenced_task_id], influenced_task_id, 0)
            Simulator.push_a_waiting_task(a_waiting_task)

        return new_banned_gpu_status, new_banned_server_per_pod

    def handle_repair_event(self, failure_id):
        self.gpu_placement_scheduler.repair_fail_gpu(self.failure_id_gpu_status_map[failure_id][0],
                                                     self.failure_id_gpu_status_map[failure_id][1])
