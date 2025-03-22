
from rapidnetsim.communication_strategy.strategy_base import StrategyBase
from rapidnetsim.core.infrastructure.flow import Flow

class HwOxcAll2AllSz(StrategyBase):
    def __init__(self) -> None:
        pass


    def deal_job(self, taskid, model_size, task_occupied_NIC_num, use_NIC_list, NIC_num_in_a_server):
        """The initial jobs are assigned according to communication strategy.
        """
        from rapidnetsim.core.simulator import Simulator
        from rapidnetsim.core.event.flow_transmit_event import FlowTransmitEvent

        print(f'Time {Simulator.get_current_time()} start task {taskid} occuping NIC num {len(use_NIC_list)}')
        Simulator.task_time_logger.write(f'taskid,{taskid},start_time,{Simulator.get_current_time()}\n')


        # Deal with only 1 GPU occupation
        if task_occupied_NIC_num == 1:
            flow_list = []
            flow = Flow(Simulator.FLOWID, model_size, None, use_NIC_list[0], use_NIC_list[0], model_size, None, taskid,
                        0, task_occupied_NIC_num, False)
            self.record_network_occupy(taskid, 0, flow, use_NIC_list[0])
            flow_list.append(flow)
            Simulator.register_event(FlowTransmitEvent(0, flow_list))
            Simulator.FLOWID += 1
            return

        round_pair_list = self.get_task_a_iteration_pair_list(task_occupied_NIC_num, model_size, NIC_num_in_a_server)

        roundid = 0
        roundidflag_list = Simulator.ITERATION_FINISH_ROUNDID_DICT[taskid]
        max_roundid = int(roundidflag_list[-1])
        
        flag = False
        while flag == False:
            for pair_list in round_pair_list:
                # Every round
                for (src, dst, communication_size) in pair_list:
                    # use_NIC_list[src] maps old may-occupied NIC_id to new unoccupied NIC_id
                    flow = Flow(Simulator.FLOWID, communication_size, None, use_NIC_list[src], use_NIC_list[dst],
                                communication_size, None, taskid, roundid, task_occupied_NIC_num, False)
                    self.record_network_occupy(taskid, roundid, flow, use_NIC_list[src])
                    Simulator.FLOWID += 1
                if roundid == max_roundid:
                    flag = True
                roundid += 1


        # Register first round job flows
        flow_list = []
        for flowid, flow in Simulator.get_wait_transmit_dict()[f'{taskid}_0'].items():
            flow_list.append(flow)
        Simulator.register_event(FlowTransmitEvent(0, flow_list))


    def get_task_a_iteration_pair_list(self, task_occupied_NIC_num, model_size, NIC_num_in_a_server):
        if task_occupied_NIC_num <= NIC_num_in_a_server:
            round_pair_list = self.get_all_to_all_every_round_pair(task_occupied_NIC_num, model_size)
        else:
            round_pair_list = self.get_oxc_all2all_sz_every_round_pair(task_occupied_NIC_num, model_size, NIC_num_in_a_server)
        return round_pair_list


    def get_all_to_all_every_round_pair(self, task_occupied_NIC_num, model_size):
        """Return communication pair in every round under all-to-all strategy.
        [
            [(NIC_src, NIC_dst, communication_size), (NIC_src, NIC_dst, communication_size)] ...
            [(NIC_src, NIC_dst, communication_size), (NIC_src, NIC_dst, communication_size)], ...
            ...
        ]
        """
        all_to_all_pair_list = []
        
        round_num = task_occupied_NIC_num - 1
    
        # all_to_all
        mask = 1
        communication_size = model_size / task_occupied_NIC_num
        for _ in range(0, round_num):
            a_round = []
            for pair in range(0, task_occupied_NIC_num):
                NIC_src = pair
                NIC_dst = (pair ^ mask)
                a_round.append((NIC_src, NIC_dst, communication_size))
            all_to_all_pair_list.append(a_round)
            mask = mask + 1
        return all_to_all_pair_list


    def get_oxc_all2all_sz_every_round_pair(self, task_occupied_NIC_num, model_size, NIC_num_in_a_server):
        pair_list = []
        server_num = int(task_occupied_NIC_num / NIC_num_in_a_server)
        NIC_count = 0
        switch_map = [[0 for _ in range(NIC_num_in_a_server)] for _ in range(server_num)]
        for i in range(server_num):
            for j in range(NIC_num_in_a_server):
                switch_map[i][j] = NIC_count
                NIC_count += 1

        for k in range(NIC_num_in_a_server):
            for m in range(NIC_num_in_a_server):
                round_list = []
                for i in range(NIC_num_in_a_server):
                    src_server = i
                    dst_NIC = (i + m) % NIC_num_in_a_server
                    for j in range(NIC_num_in_a_server):
                        src_NIC = j
                        src = switch_map[src_server][src_NIC]
                        dst_server = (k + j) % NIC_num_in_a_server
                        dst = switch_map[dst_server][dst_NIC]
                        if src != dst:
                            round_list.append((src, dst, model_size))
                pair_list.append(round_list)
        return pair_list

if __name__ == '__main__':
    test = HwOxcAll2AllSz()
    res = test.get_oxc_all2all_sz_every_round_pair(16, 220, 4)
    print(res)
