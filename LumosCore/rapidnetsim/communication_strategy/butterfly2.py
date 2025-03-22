import math
from rapidnetsim.communication_strategy.strategy_base import StrategyBase
from rapidnetsim.core.infrastructure.flow import Flow


class Butterfly2(StrategyBase):
    def __init__(self):
        super().__init__()

    def deal_job(self, taskid, model_size, task_occupied_NIC_num, use_NIC_list, NIC_num_in_a_server, special_pair=None):
        """The initial jobs are assigned according to communication strategy.
        """
        from rapidnetsim.core.simulator import Simulator
        from rapidnetsim.core.event.flow_transmit_event import FlowTransmitEvent

        print(f'Time {Simulator.get_current_time()} start task {taskid} occuping NIC num {len(use_NIC_list)}')
        Simulator.task_time_logger.write(f'taskid,{taskid},start_time,{Simulator.get_current_time()}\n')
        computation_time = float(Simulator.CONF_DICT['computation_time'])
        # computation_time = float(Simulator.TASK_LIST[taskid][3])
        conservative = Simulator.CONF_DICT['find_next_hop_method'] == 'conservative'

        schedule_time_cost = Simulator.SCHEDULER_TIME_COST[taskid]
        # Deal with only 1 GPU occupation
        if task_occupied_NIC_num == 1:
            flow_list = []
            flow = Flow(Simulator.FLOWID, model_size, None, use_NIC_list[0], use_NIC_list[0], model_size, None, taskid,
                        0, task_occupied_NIC_num, conservative)
            self.record_network_occupy(taskid, 0, flow, use_NIC_list[0])
            flow_list.append(flow)
            Simulator.register_event(FlowTransmitEvent(computation_time + schedule_time_cost, flow_list))
            Simulator.FLOWID += 1
            return

        print("debug continue_record_more_iteration_if_need")
        round_pair_list = self.get_task_a_iteration_pair_list(task_occupied_NIC_num, model_size, NIC_num_in_a_server,
                                                              special_pair)

        roundid = 0
        for pair_list in round_pair_list:
            # Every round
            for (src, dst, communication_size) in pair_list:
                # use_NIC_list[src] maps old may-occupied NIC_id to new unoccupied NIC_id
                if 'flowletsize' not in Simulator.CONF_DICT or Simulator.CONF_DICT['flowletsize'] == 'MAX' or \
                        Simulator.CONF_DICT['flowletsize'] == '' or int(use_NIC_list[src] / NIC_num_in_a_server) == int(
                        use_NIC_list[dst] / NIC_num_in_a_server):
                    flow = Flow(Simulator.FLOWID, communication_size, None, use_NIC_list[src], use_NIC_list[dst],
                                communication_size, None, taskid, roundid, task_occupied_NIC_num, False)
                    self.record_network_occupy(taskid, roundid, flow, use_NIC_list[src])
                    Simulator.FLOWID += 1
                else:
                    flowletsize = float(Simulator.CONF_DICT['flowletsize'])
                    flowlet_num = min(10, math.ceil(communication_size / flowletsize))
                    flowletsize = communication_size / flowlet_num
                    remain_size = communication_size

                    while (remain_size > 0):
                        flow = Flow(Simulator.FLOWID, min(remain_size, flowletsize), None, use_NIC_list[src],
                                    use_NIC_list[dst], min(remain_size, flowletsize), None, taskid, roundid,
                                    task_occupied_NIC_num, False)
                        self.record_network_occupy(taskid, roundid, flow, use_NIC_list[src])
                        Simulator.FLOWID += 1
                        remain_size = remain_size - flowletsize
            roundid += 1

        # Register first round job flows
        flow_list = []
        for flowid, flow in Simulator.get_wait_transmit_dict()[f'{taskid}_0'].items():
            flow_list.append(flow)
        Simulator.register_event(FlowTransmitEvent(computation_time + schedule_time_cost, flow_list))

    def is_power_of_2(self, n):
        return n & (n - 1) == 0

    def closest_power_of_2(self, initial):
        temp = initial - 1
        temp |= temp >> 1
        temp |= temp >> 2
        temp |= temp >> 4
        temp |= temp >> 8
        temp |= temp >> 16
        target = 1 if temp < 0 else temp + 1
        return target

    def closest_min_power_of_2(self, initial):
        closest_power_num = self.closest_power_of_2(initial)
        if closest_power_num > initial:
            return int(2 ** (math.log(closest_power_num, 2) - 1))
        else:
            return int(closest_power_num)

    def get_task_a_iteration_pair_list(self, task_occupied_NIC_num, model_size, NIC_num_in_a_server, special_pair=None):
        if self.is_power_of_2(task_occupied_NIC_num):
            return self.get_butterfly2_every_round_pair(task_occupied_NIC_num, model_size)
        else:
            closest_min_power_num = self.closest_min_power_of_2(task_occupied_NIC_num)
            tmp = self.get_butterfly2_every_round_pair(closest_min_power_num, model_size)
            before_extra_pair_list = []
            after_extra_pair_list = []
            diff_num = task_occupied_NIC_num - closest_min_power_num
            # for i in range(0, diff_num):
            #     before_extra_pair_list.append((task_occupied_NIC_num + i, i, model_size))
            #     after_extra_pair_list.append((i, task_occupied_NIC_num + i, model_size))

            if special_pair == None:
                return [[(0, 0, 0)]] + tmp + [[(0, 0, 0)]]
            print("debug special_pair", special_pair)
            for src, dst in special_pair:
                before_extra_pair_list.append((src, dst, model_size))
                after_extra_pair_list.append((dst, src, model_size))
            return [before_extra_pair_list] + tmp + [after_extra_pair_list]

    def get_butterfly2_every_round_pair(self, task_occupied_NIC_num, model_size):
        """Return communication pair in every round under butterfly strategy.
        [
            [(NIC_src, NIC_dst, communication_size)], [(NIC_src, NIC_dst, communication_size)] ...
            [(NIC_src, NIC_dst, communication_size)], [(NIC_src, NIC_dst, communication_size)], ...
            ...
        ]
        """
        butterfly_pair_list = []
        round_num = math.log2(task_occupied_NIC_num)
        assert (round_num.is_integer())

        round_num = int(round_num)

        # Reduce-Scatter
        mask = 1
        communication_size = model_size / 2
        for _ in range(0, round_num):
            a_round = []
            for pair in range(0, task_occupied_NIC_num):
                NIC_src = pair
                NIC_dst = (pair ^ mask)
                a_round.append((NIC_src, NIC_dst, communication_size))
            butterfly_pair_list.append(a_round)
            mask = mask * 2
            communication_size = communication_size / 2

        # All-Gather
        # ---- error ----
        # mask = 1
        # communication_size = model_size / task_occupied_NIC_num
        # for _ in range(0, round_num):
        #     a_round = []
        #     for pair in range(0, task_occupied_NIC_num):
        #         NIC_src = pair
        #         NIC_dst = (pair ^ mask)
        #         a_round.append((NIC_src, NIC_dst, communication_size))
        #     butterfly_pair_list.append(a_round)
        #     mask = mask * 2
        #     communication_size = communication_size * 2
        # ---- error ----
        final_butterfly_pair_list = butterfly_pair_list.copy()
        length = len(butterfly_pair_list)
        for i in range(length - 1, -1, -1):
            final_butterfly_pair_list.append(butterfly_pair_list[i])

        return final_butterfly_pair_list

    def get_expected_completion_time(self, task_seq=0):
        from rapidnetsim.core.simulator import Simulator
        task = Simulator.TASK_LIST[task_seq]
        model_size, task_occupied_NIC_num = task.model_size, task.gpu_num
        NIC_num_in_a_server = int(Simulator.CONF_DICT['NIC_num_in_a_server'])
        node_num = int(task_occupied_NIC_num / NIC_num_in_a_server)
        expected_intra_time = 2 * model_size * (NIC_num_in_a_server - 1) / (
                    NIC_num_in_a_server * float(Simulator.CONF_DICT['inner_server_bandwidth']))
        expected_inter_time = 2 * model_size * (node_num - 1) / (
                    task_occupied_NIC_num * float(Simulator.CONF_DICT['switch_port_bandwidth']))
        expected_completion_time = expected_intra_time + expected_inter_time
        return expected_completion_time

# import math
# from rapidnetsim.communication_strategy.strategy_base import StrategyBase
# from rapidnetsim.core.infrastructure.flow import Flow

# class Butterfly2(StrategyBase):
#     def __init__(self) -> None:
#         pass


#     # def deal_job(self, taskid, model_size, task_occupied_NIC_num, use_NIC_list, NIC_num_in_a_server):
#     #     """The initial jobs are assigned according to communication strategy.
#     #     """
#     #     from rapidnetsim.core.simulator import Simulator
#     #     from rapidnetsim.core.event.flow_transmit_event import FlowTransmitEvent

#     #     print(f'Time {Simulator.get_current_time()} start task {taskid} occuping NIC num {len(use_NIC_list)}')
#     #     Simulator.task_time_logger.write(f'taskid,{taskid},start_time,{Simulator.get_current_time()}\n')
#     #     #computation_time = float(Simulator.CONF_DICT['computation_time'])
#     #     computation_time = float(Simulator.TASK_LIST[taskid][3])
#     #     conservative = False
#     #     if Simulator.CONF_DICT['find_next_hop_method'] == 'conservative':
#     #         conservative = True

#     #     schedule_time_cost = Simulator.SCHEDULER_TIME_COST[taskid]
#     #     # Deal with only 1 GPU occupation
#     #     if task_occupied_NIC_num == 1:
#     #         flow_list = []
#     #         flow = Flow(
#     #             Simulator.FLOWID, model_size, None, use_NIC_list[0], use_NIC_list[0],
#     #             model_size, None,
#     #             taskid, 0, task_occupied_NIC_num, conservative
#     #         )
#     #         self.record_network_occupy(taskid, 0, flow, use_NIC_list[0])
#     #         flow_list.append(flow)
#     #         Simulator.register_event(FlowTransmitEvent(computation_time+schedule_time_cost, flow_list))
#     #         Simulator.FLOWID += 1
#     #         return


#     #     round_pair_list = self.get_task_a_iteration_pair_list(task_occupied_NIC_num, model_size, NIC_num_in_a_server)

#     #     roundid = 0
#     #     roundidflag_list = Simulator.ITERATION_FINISH_ROUNDID_DICT[taskid]
#     #     max_roundid = int(roundidflag_list[-1])

#     #     flag = False
#     #     while flag == False:
#     #         for pair_list in round_pair_list:
#     #             # Every round
#     #             for (src, dst, communication_size) in pair_list:
#     #                 # use_NIC_list[src] maps old may-occupied NIC_id to new unoccupied NIC_id
#     #                 flow = Flow(
#     #                     Simulator.FLOWID, communication_size, None, use_NIC_list[src], use_NIC_list[dst],
#     #                     communication_size, None,
#     #                     taskid, roundid, task_occupied_NIC_num, conservative
#     #                 )
#     #                 self.record_network_occupy(taskid, roundid, flow, use_NIC_list[src])
#     #                 Simulator.FLOWID += 1
#     #             if roundid == max_roundid:
#     #                 flag = True
#     #             roundid += 1


#     #     # Register first round job flows
#     #     flow_list = []
#     #     for flowid, flow in Simulator.get_wait_transmit_dict()[f'{taskid}_0'].items():
#     #         flow_list.append(flow)
#     #     Simulator.register_event(FlowTransmitEvent(computation_time+schedule_time_cost, flow_list))


#     # def get_task_a_iteration_pair_list(self, task_occupied_NIC_num, model_size, NIC_num_in_a_server):
#     #     return self.get_butterfly2_every_round_pair(task_occupied_NIC_num, model_size)


#     # def get_butterfly2_every_round_pair(self, task_occupied_NIC_num, model_size):
#     #     """Return communication pair in every round under butterfly strategy.
#     #     [
#     #         [(NIC_src, NIC_dst, communication_size)], [(NIC_src, NIC_dst, communication_size)] ...
#     #         [(NIC_src, NIC_dst, communication_size)], [(NIC_src, NIC_dst, communication_size)], ...
#     #         ...
#     #     ]
#     #     """
#     #     butterfly_pair_list = []
#     #     round_num = math.log2(task_occupied_NIC_num)
#     #     assert(round_num.is_integer())
#     #     round_num = int(round_num)

#     #     # Reduce-Scatter
#     #     mask = 1
#     #     communication_size = model_size / 2
#     #     for _ in range(0, round_num):
#     #         a_round = []
#     #         for pair in range(0, task_occupied_NIC_num):
#     #             NIC_src = pair
#     #             NIC_dst = (pair ^ mask)
#     #             a_round.append((NIC_src, NIC_dst, communication_size))
#     #         butterfly_pair_list.append(a_round)
#     #         mask = mask * 2
#     #         communication_size = communication_size / 2

#     #     final_butterfly_pair_list = butterfly_pair_list.copy()
#     #     length = len(butterfly_pair_list)
#     #     for i in range(length - 1, -1, -1):
#     #         final_butterfly_pair_list.append(butterfly_pair_list[i])

#     #     return final_butterfly_pair_list
#     def deal_job(self, taskid, model_size, task_occupied_NIC_num, use_NIC_list, NIC_num_in_a_server):
#         from rapidnetsim.core.simulator import Simulator
#         from rapidnetsim.core.event.flow_transmit_event import FlowTransmitEvent

#         computation_time = float(Simulator.TASK_LIST[taskid][3])
#         schedule_time_cost = Simulator.SCHEDULER_TIME_COST[taskid]
#         if(task_occupied_NIC_num == pow(2,math.ceil(math.log2(task_occupied_NIC_num)))):
#             """The initial jobs are assigned according to communication strategy.
#             """
#             from rapidnetsim.core.simulator import Simulator
#             from rapidnetsim.core.event.flow_transmit_event import FlowTransmitEvent

#             print(f'Time {Simulator.get_current_time()} start task {taskid} occuping NIC num {len(use_NIC_list)}')
#             Simulator.task_time_logger.write(f'taskid,{taskid},start_time,{Simulator.get_current_time()}\n')


#             conservative = False
#             if Simulator.CONF_DICT['find_next_hop_method'] == 'conservative':
#                 conservative = True


#             # Deal with only 1 GPU occupation
#             if task_occupied_NIC_num == 1:
#                 flow_list = []
#                 flow = Flow(
#                     Simulator.FLOWID, model_size, None, use_NIC_list[0], use_NIC_list[0],
#                     model_size, None,
#                     taskid, 0, task_occupied_NIC_num, conservative
#                 )
#                 self.record_network_occupy(taskid, 0, flow, use_NIC_list[0])
#                 flow_list.append(flow)
#                 Simulator.register_event(FlowTransmitEvent(computation_time+schedule_time_cost, flow_list))
#                 Simulator.FLOWID += 1
#                 return

#             round_pair_list = self.get_task_a_iteration_pair_list(task_occupied_NIC_num, model_size, NIC_num_in_a_server)

#             roundid = 0
#             roundidflag_list = Simulator.ITERATION_FINISH_ROUNDID_DICT[taskid]
#             max_roundid = int(roundidflag_list[-1])

#             flag = False
#             while flag == False:
#                 for pair_list in round_pair_list:
#                     # Every round
#                     for (src, dst, communication_size) in pair_list:
#                         # use_NIC_list[src] maps old may-occupied NIC_id to new unoccupied NIC_id
#                         flow = Flow(
#                             Simulator.FLOWID, communication_size, None, use_NIC_list[src], use_NIC_list[dst],
#                             communication_size, None,
#                             taskid, roundid, task_occupied_NIC_num, conservative
#                         )
#                         self.record_network_occupy(taskid, roundid, flow, use_NIC_list[src])
#                         Simulator.FLOWID += 1
#                     if roundid == max_roundid:
#                         flag = True
#                     roundid += 1


#             # Register first round job flows
#             flow_list = []
#             for flowid, flow in Simulator.get_wait_transmit_dict()[f'{taskid}_0'].items():
#                 flow_list.append(flow)
#             Simulator.register_event(FlowTransmitEvent(computation_time+schedule_time_cost, flow_list))
#         else:
#             """The initial jobs are assigned according to communication strategy.
#             """
#             from rapidnetsim.core.simulator import Simulator
#             from rapidnetsim.core.event.flow_transmit_event import FlowTransmitEvent

#             print(f'Time {Simulator.get_current_time()} start task {taskid} occuping NIC num {len(use_NIC_list)}')
#             Simulator.task_time_logger.write(f'taskid,{taskid},start_time,{Simulator.get_current_time()}\n')

#             # Deal with only 1 GPU occupation
#             if task_occupied_NIC_num == 1:
#                 flow_list = []
#                 flow = Flow(
#                     Simulator.FLOWID, model_size, None, use_NIC_list[0], use_NIC_list[0],
#                     model_size, None,
#                     taskid, 0, task_occupied_NIC_num, False
#                 )
#                 self.record_network_occupy(taskid, 0, flow, use_NIC_list[0])
#                 flow_list.append(flow)
#                 Simulator.register_event(FlowTransmitEvent(computation_time+schedule_time_cost, flow_list))
#                 Simulator.FLOWID += 1
#                 return

#             communication_size = model_size / task_occupied_NIC_num / 2

#             round_pair_list = self.get_butterfly3_every_round_pair(task_occupied_NIC_num, model_size)

#             roundid = 0
#             roundidflag_list = Simulator.ITERATION_FINISH_ROUNDID_DICT[taskid]
#             max_roundid = int(roundidflag_list[-1])    # For supporting multiple interation

#             flag = False
#             while flag == False:
#                 for pair_list in round_pair_list:
#                     # Every round
#                     for (src, dst) in pair_list:
#                         # use_NIC_list[src] maps old may-occupied NIC_id to new unoccupied NIC_id
#                         flow = Flow(
#                             Simulator.FLOWID, communication_size, None, use_NIC_list[src], use_NIC_list[dst],
#                             communication_size, None,
#                             taskid, roundid, task_occupied_NIC_num
#                         )
#                         self.record_network_occupy(taskid, roundid, flow, use_NIC_list[src])
#                         Simulator.FLOWID += 1
#                     if roundid == max_roundid:
#                         flag = True
#                     roundid += 1

#             # Register first round job flow
#             flow_list = []
#             for flowid, flow in Simulator.get_wait_transmit_dict()[f'{taskid}_0'].items():
#                 flow_list.append(flow)
#             Simulator.register_event(FlowTransmitEvent(computation_time+schedule_time_cost, flow_list))


#     def get_task_a_iteration_pair_list(self, task_occupied_NIC_num, model_size, NIC_num_in_a_server):
#         round_pair_list = self.get_butterfly3_every_round_pair(task_occupied_NIC_num, model_size)
#         return round_pair_list


#     def get_butterfly3_every_round_pair(self, task_occupied_NIC_num, model_size):
#         """Return communication pair in every round under butterfly strategy.
#         [
#             [(NIC_src, NIC_dst, communication_size)], [(NIC_src, NIC_dst, communication_size)] ...
#             [(NIC_src, NIC_dst, communication_size)], [(NIC_src, NIC_dst, communication_size)], ...
#             ...
#         ]
#         """
#         if(task_occupied_NIC_num == pow(2,math.ceil(math.log2(task_occupied_NIC_num)))):
#             butterfly_pair_list = []
#             round_num = math.log2(task_occupied_NIC_num)
#             assert(round_num.is_integer())
#             round_num = int(round_num)

#             # Reduce-Scatter
#             mask = 1
#             communication_size = model_size / 2
#             for _ in range(0, round_num):
#                 a_round = []
#                 for pair in range(0, task_occupied_NIC_num):
#                     NIC_src = pair
#                     NIC_dst = (pair ^ mask)
#                     a_round.append((NIC_src, NIC_dst, communication_size))
#                 butterfly_pair_list.append(a_round)
#                 mask = mask * 2
#                 communication_size = communication_size / 2

#             final_butterfly_pair_list = butterfly_pair_list.copy()
#             length = len(butterfly_pair_list)
#             for i in range(length - 1, -1, -1):
#                 final_butterfly_pair_list.append(butterfly_pair_list[i])

#             return final_butterfly_pair_list
#         else:
#             ring_pair_list = []
#             round_num = 2 * (task_occupied_NIC_num - 1)

#             for _ in range(round_num):
#                 forward = []
#                 backward = []
#                 for i in range(task_occupied_NIC_num):
#                     src = i
#                     if i == task_occupied_NIC_num - 1:
#                         dst = 0
#                     else:
#                         dst = i + 1
#                     forward.append((src, dst))
#                     backward.append((dst, src))
#                 ring_pair_list.append(forward + backward)

#             return ring_pair_list
