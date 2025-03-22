import copy
import math
import re
import time
import rapidnetsim.scheduler.static_locality_AI.utils as utils
import rapidnetsim.scheduler.static_locality_AI.job as job
import rapidnetsim.scheduler.static_locality_AI.connection_manager as connection_manager
import rapidnetsim.scheduler.static_locality_AI.leaf_resource_manager as leaf_resource_manager
import rapidnetsim.scheduler.static_locality_AI.server_resource_manager as server_resource_manager


class StaticPlacementerAI:
    def __init__(self, spine_switch_num, leaf_switch_num, spine_switch_port_num, leaf_switch_port_num, server_num, oxc_num=32, inference_model=None):
        self.gpu_num = spine_switch_num*spine_switch_port_num
        self.server_num = server_num
        self.leaf_num = leaf_switch_num
        self.spine_num = spine_switch_num
        self.gpu_per_server = int(self.gpu_num/server_num)
        self.gpu_per_leaf = int(self.gpu_num/self.leaf_num)
        self.port_per_spine = int(self.gpu_num/self.spine_num)
        print("Cluster Info:")
        print("server_num: "+" ")
        print(server_num)
        print("leaf_num: "+" ")
        print(leaf_switch_num)
        print("gpu_num: "+" ")
        print(self.gpu_num)
        self.server_resource_manager_ = server_resource_manager.ServerResourceManager(server_num, self.gpu_per_server, self.leaf_num)
        self.leaf_resource_manager_ = leaf_resource_manager.LeafResourceManager(self.leaf_num, self.gpu_per_leaf)
        self.connection_manager_ = connection_manager.ConnectionManager(self.gpu_num, self.server_num, self.leaf_num, self.spine_num, inference_model)

        # job queue
        self.current_job_list = {}
        self.history_job_list = {}
        self.task_contention_map = {}
        self.link_weight = {}
        

    def schedule(self, gpu_num, job_id, sim_time, queued_jobs):
        from rapidnetsim.core.simulator import Simulator
        print("some job arrive: "+str(job_id)+","+str(gpu_num))
        time_start = time.perf_counter()
        new_job = job.Job(job_id)
        chosen_gpu_list = []
        allocation_link_mapping = []
        # 情况零：GPU数量不足
        if gpu_num > self.server_resource_manager_.cal_remain_gpu_num():
            print("finish allocation, no resource0")
            return False, None, None, None,None,None,None
        if not self.server_resource_manager_.whether_can_find_valid_server(gpu_num):
            print("finish allocation, no resource1")
            return False, None, None, None,None,None,None

        potentional_leaf_list = []
        # Step1. 在leaf_resource_manager中选取合适的leafgroup
        for temp_leaf_id in range(self.leaf_num):
            require_server_num = math.ceil(gpu_num/self.gpu_per_server)
            require_gpu_num_in_server = min(self.gpu_per_server,gpu_num)
            valid_server_num = 0
            for temp_server_num in range(int(temp_leaf_id*self.gpu_per_leaf/self.gpu_per_server), int((1+temp_leaf_id)*self.gpu_per_leaf/self.gpu_per_server),1):
                if self.server_resource_manager_.server_list[temp_server_num].remain_gpu_num()>=require_gpu_num_in_server:
                    valid_server_num += 1
            if valid_server_num>=require_server_num:
                potentional_leaf_list.append([temp_leaf_id, sum(self.leaf_resource_manager_.leaf_list[temp_leaf_id].leaf_group)])
        potentional_leaf_list.sort( key=lambda x: (x[1])) 
        if len(potentional_leaf_list)>0:
            temp_leaf_id = potentional_leaf_list[0][0]
            #  Step2 在选择的leaf交换机下联的server中按照locality选择gpu
            chosen_gpu_list = self.server_resource_manager_.choose_gpu_in_one_leaf(temp_leaf_id, gpu_num)
            self.leaf_resource_manager_.leaf_list[temp_leaf_id].update_leaf_group_with_required_num(gpu_num)
            # gpu - leaf links
            for output_gpu_index in chosen_gpu_list:
                assert int(output_gpu_index/self.gpu_per_leaf) == temp_leaf_id
                output_leaf_index = utils.get_leaf_module_id(temp_leaf_id, self.gpu_num)
                allocation_link_mapping.append([output_gpu_index, output_leaf_index, 1])
                allocation_link_mapping.append([output_leaf_index, output_gpu_index, 1])
            #  记录job
                new_job.start_time = sim_time
                new_job.allocated_gpus = chosen_gpu_list
                self.current_job_list[job_id] = new_job
            print("finish allocation1")
            f2 = open('queue_length.txt','a')
            f2.write(str(len(queued_jobs)))
            f2.write(",")
            f2.write(str(sim_time) )
            f2.write("\n" )
            f2.close()         
            time_end = time.perf_counter()
            time_sum = time_end-time_start
            Simulator.SCHEDULER_TIME_COST[job_id] = 0   
            f3 = open('schedule_time_cost.txt','a')
            f3.write(str(job_id))
            f3.write(",")
            f3.write(str(time_sum) )
            f3.write("\n" )
            f3.close()          
            return True, True, chosen_gpu_list, allocation_link_mapping,None,None,None

        # 否则需要跨leaf通信

        self.server_resource_manager_.release_gpu_in_server(chosen_gpu_list)
        self.leaf_resource_manager_.release_group_with_given_gpu_list(chosen_gpu_list)
        # temp_z = pow(2,int(math.log2(int(gpu_num))))
        # temp_require_leaf_num = max(1,int(temp_z/self.gpu_per_leaf))
        temp_k_value = gpu_num
        temp_two_part = 1
        while temp_k_value%2 == 0:
            temp_k_value = int(temp_k_value/2)
            temp_two_part *= 2
        temp_require_leaf_num = max(temp_k_value,int(gpu_num/self.gpu_per_leaf))
        
        allocate_success = False
        
        leaf_remain_empt_server_list = []
        leaf_remain_empt_gpu_list = []
        for temp_leaf_id in range(self.leaf_num):
            leaf_remain_empt_server_list.append(0)
            leaf_remain_empt_gpu_list.append(0)
        for temp_server_id in range(self.server_num):
            temp_leaf_id = int(temp_server_id/self.gpu_per_leaf*self.gpu_per_server)
            if self.gpu_per_server in self.server_resource_manager_.server_list[temp_server_id].gpu_group:
                leaf_remain_empt_server_list[temp_leaf_id] += 1
            leaf_remain_empt_gpu_list[temp_leaf_id] += sum(self.server_resource_manager_.server_list[temp_server_id].gpu_group)
        allocate_success, allocation_link_mapping, leaf_occupy_gpu_num_map, job_allocated_leaf_spine_link, may_contention_link = self.connection_manager_.find_valid_ai_iter(gpu_num, leaf_remain_empt_server_list, temp_require_leaf_num, leaf_remain_empt_gpu_list,self.link_weight)
        if allocate_success:
            chosen_gpu_list = []
            for chosen_leaf_id in leaf_occupy_gpu_num_map:
                chosen_gpu_list.extend(self.server_resource_manager_.choose_gpu_in_one_leaf(chosen_leaf_id, leaf_occupy_gpu_num_map[chosen_leaf_id]))
                self.leaf_resource_manager_.leaf_list[chosen_leaf_id].update_leaf_group_with_required_num(leaf_occupy_gpu_num_map[chosen_leaf_id])
            for output_gpu_index in chosen_gpu_list:
                output_leaf_index = utils.get_leaf_module_id(int(output_gpu_index/self.gpu_per_leaf), self.gpu_num)
                allocation_link_mapping.append([output_gpu_index, output_leaf_index, 1])
                allocation_link_mapping.append([output_leaf_index, output_gpu_index, 1])
            new_job.start_time = sim_time
            new_job.allocated_gpus = chosen_gpu_list
            assert len(chosen_gpu_list) == gpu_num
            new_job.job_allocated_leaf_spine_link = job_allocated_leaf_spine_link
            self.current_job_list[job_id] = new_job
            print("finish allocation2")
            f2 = open('queue_length.txt','a')
            f2.write(str(len(queued_jobs)))
            f2.write(",")
            f2.write(str(sim_time) )
            f2.write("\n" )
            f2.close()    
            time_end = time.perf_counter()
            time_sum = time_end-time_start
            Simulator.SCHEDULER_TIME_COST[job_id] = 0    
            f3 = open('schedule_time_cost.txt','a')
            f3.write(str(job_id))
            f3.write(",")
            f3.write(str(time_sum) )
            f3.write("\n" )
            f3.close()    
            self.task_contention_map[job_id] = may_contention_link
            return True, True, chosen_gpu_list, allocation_link_mapping,may_contention_link,None,None
            # else:
            #     temp_require_leaf_num*=2
        if not allocate_success:
            print("finish allocation, no resource2 ",gpu_num)
            f1 = open('fragmention.txt','a')
            f1.write(str(job_id) )
            f1.write(",")
            f1.write(str(sim_time) )
            f1.write("\n" )
            f1.close()
            self.leaf_resource_manager_.print_remain_leaf_port_num()
            return False, None, None, None,None,None,None

                
    def update_finished_job(self, job_id, sim_time, queued_jobs):
        print("some job finish" + str(job_id))
        from rapidnetsim.core.simulator import Simulator
        if job_id in self.task_contention_map:
            for k,v in self.task_contention_map[job_id].items():
                if k in Simulator.contention_link and Simulator.contention_link[k]>0:
                    Simulator.contention_link[k] -= v
                if Simulator.contention_link[k]>=0:
                    print("debug contention ",k, Simulator.contention_link[k])
        to_leave_job = copy.deepcopy(self.current_job_list[job_id])
        to_leave_job.finish_time = sim_time
        self.history_job_list[job_id] = to_leave_job
        self.server_resource_manager_.release_gpu_in_server(to_leave_job.allocated_gpus)
        self.leaf_resource_manager_.release_group_with_given_gpu_list(to_leave_job.allocated_gpus)
        spine_portNum_map = {}
        for leaf_id in to_leave_job.job_allocated_leaf_spine_link:
            for spine_id in to_leave_job.job_allocated_leaf_spine_link[leaf_id]:
                used_port_num = to_leave_job.job_allocated_leaf_spine_link[leaf_id][spine_id]
                if used_port_num>0:
                    # print("debug link_weight",self.link_weight[str(leaf_id)+'_'+str(spine_id)],len(to_leave_job.allocated_gpus))
                    self.link_weight[str(leaf_id)+'_'+str(spine_id)] -= len(to_leave_job.allocated_gpus)
                    assert self.link_weight[str(leaf_id)+'_'+str(spine_id)]>=0
                if used_port_num>0:
                    if spine_id not in spine_portNum_map:
                        spine_portNum_map[spine_id] = 0
                    spine_portNum_map[spine_id] += used_port_num
        self.connection_manager_.release_connection_resource(to_leave_job.job_allocated_leaf_spine_link)
        del self.current_job_list[job_id]
        f2 = open('queue_length.txt','a')
        f2.write(str(len(queued_jobs)))
        f2.write(",")
        f2.write(str(sim_time) )
        f2.write("\n" )
        f2.close()                   


            
  