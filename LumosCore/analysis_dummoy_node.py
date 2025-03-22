import math
import matplotlib.pyplot as plt

up_port_num = 32
cur_list = []
opt_ele_ratio = []
for N in range(up_port_num+2):
    comm_target = N - 1
    cur_val = 1
    for k in range(1, N):
        if comm_target*k < up_port_num:
            cur_val = comm_target*k
    res_port_num = up_port_num%cur_val
    print(N, N*up_port_num*up_port_num, cur_val, res_port_num, math.ceil(N*res_port_num/(up_port_num*2)))
    cur_list.append(math.ceil(N*res_port_num/(up_port_num*2)))
    opt_ele_ratio.append(res_port_num/up_port_num)
print(sum(cur_list)/len(cur_list))
print(sum(opt_ele_ratio)/len(opt_ele_ratio))
print(opt_ele_ratio)

# Pod的数量从1开始，因此index + 1就是Pod的数量
pod_counts = [i + 1 for i in range(len(opt_ele_ratio))]

# 创建图像
plt.figure(figsize=(10, 6))

# 绘制曲线图
plt.plot(pod_counts, opt_ele_ratio, marker='o', linestyle='-', color='b', label='Fragmentation Ratio')

# 设置标题和坐标轴标签
plt.title('Port Fragmentation Ratio of Ports with Increasing Pod Count')
plt.xlabel('Number of Pods')
plt.ylabel('Port Fragmentation Ratio')

# 添加网格线
plt.grid(True)

# 显示图例
plt.legend()

# 显示图形
plt.savefig('port_utilization.pdf', bbox_inches='tight')