import math
import matplotlib.pyplot as plt

class clos:
    def __init__(self, max_switch_port_size = 256, layer_size = 2, switch_chip_bandwidth = 51200, oversubscription = 1):
        self.max_switch_port_size = max_switch_port_size
        self.layer_size = layer_size
        self.switch_chip_bandwidth = switch_chip_bandwidth
        self.oversubscription = oversubscription
        
    def caculate_bandwidth_per_pord(self, port_size):
        up_port_size = port_size // 2
        total_gpu_num = 1
        for i in range(self.layer_size):
            if i == 1:
                total_gpu_num = total_gpu_num*(math.floor(up_port_size*2/(1+self.oversubscription)*self.oversubscription))
            elif i == self.layer_size-1:
                total_gpu_num = total_gpu_num*port_size
            else:
                total_gpu_num = total_gpu_num*up_port_size
        bandwidth_per_port = self.switch_chip_bandwidth // port_size
        total_bandwidth = bandwidth_per_port * total_gpu_num 
        
        return bandwidth_per_port,total_bandwidth
    
    def caculate_total_bandwidth_according_to_bandwidth_per_port(self, bandwidth_per_port=200):
        up_port_per_switch = min(self.switch_chip_bandwidth // bandwidth_per_port,self.max_switch_port_size) / 2
        last_layer_port_per_switch =  min(self.switch_chip_bandwidth // bandwidth_per_port,self.max_switch_port_size)
        # print("debug layer size")
        # print(self.layer_size,up_port_per_switch,(math.floor(up_port_per_switch*2/(1+self.oversubscription)*self.oversubscription)),self.switch_chip_bandwidth, bandwidth_per_port,last_layer_port_per_switch)
        total_gpu_num = 1
        for i in range(self.layer_size):
            if i == 1 and self.layer_size!=2:
                total_gpu_num = total_gpu_num*(math.floor(up_port_per_switch*2/(1+self.oversubscription)*self.oversubscription))
            elif i == self.layer_size-1:
                total_gpu_num = total_gpu_num*last_layer_port_per_switch
            else:
                total_gpu_num = total_gpu_num*up_port_per_switch
        total_bandwidth = round(total_gpu_num * bandwidth_per_port //1000000)
        total_gpu_num = round(total_gpu_num/1000,2)*1
        return total_gpu_num,total_bandwidth
    
class Expander:
    def __init__(self, max_switch_port_size = 256, layer_size = 1, switch_chip_bandwidth = 51200, MEMS_port = 256):
        self.max_switch_port_size = max_switch_port_size
        self.layer_size = layer_size
        self.switch_chip_bandwidth = switch_chip_bandwidth
        self.MEMS_port = MEMS_port
        
    def caculate_total_bandwidth_according_to_bandwidth_per_port(self, bandwidth_per_port=200):
        up_port_per_switch = min(self.switch_chip_bandwidth // bandwidth_per_port,self.max_switch_port_size) / 2
        
        total_gpu_num = 1
        for i in range(self.layer_size):
            total_gpu_num = total_gpu_num*up_port_per_switch
        total_gpu_num = total_gpu_num*self.MEMS_port
        total_bandwidth = round(total_gpu_num * bandwidth_per_port //1000000)
        total_gpu_num = round(total_gpu_num/1000,2)*1
        return total_gpu_num,total_bandwidth
    
# print("LumosCore(51.2T)")
# cross_expander_2 = Expander(256,2,51200,512)
# res_cross_expander_2_size = [cross_expander_2.caculate_total_bandwidth_according_to_bandwidth_per_port(i)[0] for i in [200,300,400,500,600,700,800,1600]]
# for i in res_cross_expander_2_size:
#     print('    -',i)
# print()
# #     print(i,end="k&")
# # print()
# print()

print("LumosCore(51.2T),tau2")
cross_expander_2 = Expander(256,2,12800,512)
res_cross_expander_2_size = [cross_expander_2.caculate_total_bandwidth_according_to_bandwidth_per_port(i)[0] for i in [200,400,800,1600]]
for i in res_cross_expander_2_size:
#     print('    -',i//2)
# print()
    print(i//2,end="k&")
print()
print()

print("LumosCore(51.2T),tau2")
cross_expander_2 = Expander(256,2,25600,512)
res_cross_expander_2_size = [cross_expander_2.caculate_total_bandwidth_according_to_bandwidth_per_port(i)[0] for i in [200,400,800,1600]]
for i in res_cross_expander_2_size:
#     print('    -',i//2)
# print()
    print(i//2,end="k&")
print()
print()

print("LumosCore(51.2T),tau2")
cross_expander_2 = Expander(256,2,51200,512)
res_cross_expander_2_size = [cross_expander_2.caculate_total_bandwidth_according_to_bandwidth_per_port(i)[0] for i in [200,400,800,1600]]
for i in res_cross_expander_2_size:
#     print('    -',i//2)
# print()
    print(i//2,end="k&")
print()
print()

# print("LumosCore(25.6T)")
# cross_expander_2 = Expander(256,2,25600,512)
# res_cross_expander_2_size = [cross_expander_2.caculate_total_bandwidth_according_to_bandwidth_per_port(i)[0] for i in [200,300,400,500,600,700,800,1600]]
# for i in res_cross_expander_2_size:
#     print('    -',i)
# print()
# #     print(i,end="k&")
# # print()
# print()

# print("Clos(3-tier-51.2T)")
# three_tier_clos = clos(256,3,51200)
# res_three_tier_clos_size = [three_tier_clos.caculate_total_bandwidth_according_to_bandwidth_per_port(i)[0] for i in [200,300,400,500,600,700,800,1600]]
# for i in res_three_tier_clos_size:
#     print('    -',i)
# print()
# #     print(i,end="k&")
# # print()
# print()

# print("3-tier 2:1")
# three_tier_clos_over2 = clos(256,3,51200,2)
# res_three_tier_clos_over2_size = [three_tier_clos_over2.caculate_total_bandwidth_according_to_bandwidth_per_port(i)[0] for i in [200,300,400,500,600,700,800,1600]]
# for i in res_three_tier_clos_over2_size:
#     print('    -',i)
# print()
# #     print(i,end="k&")
# # print()
# print()

# print("Clos(3tier 15:1)")
# three_tier_clos_over16 = clos(256,3,51200,15)
# res_three_tier_clos_over16_size = [three_tier_clos_over16.caculate_total_bandwidth_according_to_bandwidth_per_port(i)[0] for i in [200,300,400,500,600,700,800,1600]]
# for i in res_three_tier_clos_over16_size:
#     print('    -',i)
# print()
# #     print(i,end="k&")
# # print()
# print()

# print("Clos(4-tier-51.2T)")
# four_tier_clos = clos(256,4,51200)
# res_four_tier_clos_size = [four_tier_clos.caculate_total_bandwidth_according_to_bandwidth_per_port(i)[0] for i in [200,300,400,500,600,700,800,1600]]
# for i in res_four_tier_clos_size:
#     print('    -',i)
# print()
# #     print(i,end="k&")
# # print()
# print()

# print("Clos(2-tier-51.2T)")
# two_tier_clos = clos(256,2,51200)
# res_two_tier_clos_size = [two_tier_clos.caculate_total_bandwidth_according_to_bandwidth_per_port(i)[0] for i in [200,300,400,500,600,700,800,1600]]
# for i in res_two_tier_clos_size:
#     print('    -',i)
# print()
# #     print(i,end="k&")
# # print()


# # print("leaf-ocs")
# # cross_expander_1 = Expander(256,1,51200,512)
# # res_cross_expander_1_size = [cross_expander_1.caculate_total_bandwidth_according_to_bandwidth_per_port(i)[0] for i in [200,300,400,500,600,700,800,1600]]
# # for i in res_cross_expander_1_size:
# #     print('    -',i)
# # print()

# # print("leaf-ocs(1024)")
# # cross_expander_3 = Expander(256,1,51200,1024)
# # res_cross_expander_3_size = [cross_expander_3.caculate_total_bandwidth_according_to_bandwidth_per_port(i)[0] for i in [200,300,400,500,600,700,800,1600]]
# # for i in res_cross_expander_3_size:
# #     print('    -',i)
# # print()


# print("LumosCore(1024)")
# cross_expander_4 = Expander(256,2,51200,1024)
# res_cross_expander_4_size = [cross_expander_4.caculate_total_bandwidth_according_to_bandwidth_per_port(i)[0] for i in [200,300,400,500,600,700,800,1600]]
# for i in res_cross_expander_4_size:
#     print('    -',i)
# print()
# #     print(i,end="k&")
# # print()
# print()
# print()


# [200,400,800,1600]


# plt.plot(res_two_tier_clos_size,[200,300,400,500,600,700,800,1600],  label="2-tier")
# plt.plot(res_three_tier_clos_size,[200,300,400,500,600,700,800,1600],  label="3-tier")
# plt.plot(res_three_tier_clos_over2_size, [200,300,400,500,600,700,800,1600],  label="3-tier with 2:1")
# plt.plot(res_three_tier_clos_over16_size, [200,300,400,500,600,700,800,1600],  label="3-tier with 16:1")
# plt.plot(res_four_tier_clos_size,[200,300,400,500,600,700,800,1600],  label="4-tier")
# plt.plot(res_cross_expander_1_size, [200,300,400,500,600,700,800,1600], label="leaf-ocs")
# plt.plot(res_cross_expander_3_size, [200,300,400,500,600,700,800,1600],  label="leaf-ocs-1024")
# plt.plot(res_cross_expander_2_size, [200,300,400,500,600,700,800,1600],  label="leaf-spine-ocs")
# plt.plot(res_cross_expander_4_size, [200,300,400,500,600,700,800,1600],  label="leaf-spine-ocs-1024")
# plt.xscale("log")
# plt.ylabel('Bandwidth per Port(gbps)')
# plt.xlabel('Total Node')
# plt.legend(loc="upper right")
# plt.ylim(200, 1600)
# plt.savefig('total_size_change.png')
# plt.show()







# 200,300,400,500,600,700,800,1600




# plt.plot(res_two_tier_clos_bandwidth,[200,300,400,500,600,700,800,1600],  label="2-tier")
# plt.plot(res_three_tier_clos_bandwidth,[200,300,400,500,600,700,800,1600],  label="3-tier")
# plt.plot(res_cross_expander_1_bandwidth, [200,300,400,500,600,700,800,1600], label="leaf-ocs")
# plt.plot(res_cross_expander_2_bandwidth, [200,300,400,500,600,700,800,1600],  label="leaf-spine-ocs")
# plt.xscale("log")
# plt.ylabel('Bandwidth per Port(gbps)')
# plt.xlabel('Total Bandwidth(gbps)')
# plt.legend(loc="upper right")
# plt.ylim(200, 800)
# plt.savefig('total_bandwidth_change.png')
# plt.show()