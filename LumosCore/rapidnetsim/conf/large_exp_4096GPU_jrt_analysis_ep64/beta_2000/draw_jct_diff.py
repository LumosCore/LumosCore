import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

def date_time_str_to_long(input_date_time_string):
    if input_date_time_string == 'None':
        return 0
    time_array = time.strptime(input_date_time_string, "%Y-%m-%d %H:%M:%S")
    time_stamp = int(time.mktime(time_array))

    return time_stamp


def load_csv_get_beta(filepath):
    df = pd.read_csv(filepath, header=None)
    df.columns = ['taskidname', 'taskid', 'type', 'value']
    return df


def get_completion_time(df_data, task_num):
    res_list = []
    for i in range(task_num):
        start_time = df_data.loc[(df_data['taskid'] == i) & (df_data['type'] == 'start_time')]['value'].values[0]
        finish_time = df_data.loc[(df_data['taskid'] == i) & (df_data['type'] == 'finish_time')]['value'].values[0]
        res_list.append(finish_time - start_time)

    return res_list


def get_finish_time(df_data, task_num):
    res_list = []
    for i in range(task_num):
        arriving_time = df_data.loc[(df_data['taskid'] == i) & (df_data['type'] == 'arriving_time')]['value'].values[0]
        finish_time = df_data.loc[(df_data['taskid'] == i) & (df_data['type'] == 'finish_time')]['value'].values[0]
        res_list.append(finish_time - arriving_time)

    return res_list


def get_wait_time(df_data, task_num):
    res_list = []
    for i in range(task_num):
        arriving_time = df_data.loc[(df_data['taskid'] == i) & (df_data['type'] == 'arriving_time')]['value'].values[0]
        start_time = df_data.loc[(df_data['taskid'] == i) & (df_data['type'] == 'start_time')]['value'].values[0]
        res_list.append(start_time - arriving_time)

    return res_list



task_nums = 1000
best = load_csv_get_beta(f'best/task_time.log')
lumoscore_2tau = load_csv_get_beta(f'lumoscore_tau_2/task_time.log')
lumoscore_1tau = load_csv_get_beta(f'lumoscore_tau_1/task_time.log')
lumoscore_25_6t = load_csv_get_beta(f'lumoscore_tau_1_26T/task_time.log')
clos = load_csv_get_beta(f'ele/task_time.log')
clos_4tier_25_6t = load_csv_get_beta(f'ele_4tier/task_time.log')
clos_3tier_25_6t = load_csv_get_beta(f'ele_26T/task_time.log')

# 定义颜色
# colors = [
#     (169/255, 111/255, 176/255),  # LumosCore(51.2T,2-$\tau$)
#     (216/255, 160/255, 199/255),  # LumosCore(51.2T,1-$\tau$)
#     (247/238/246, 238/246, 246/255),  # LumosCore(25.6T,1-$\tau$)
#     (43/255, 48/255, 122/255),  # Clos(51.2T,3Tier)
#     (119/255, 194/255, 243/255),  # Clos(25.6T,4Tier)
#     (218/226/255, 226/226, 237/255)   # Clos(25.6T,3Tier)
# ]
colors = [(169/255, 111/255, 176/255), 
          (216/255, 160/255, 199/255), 
          (247/255, 167/255, 181/255), 
          (43/255, 48/255, 122/255), 
          (119/255, 194/255, 243/255), 
          (218/255, 226/255, 237/255)]

# 定义架构名称
architectures = [
    r'LumosCore(51.2T,2-$\tau$)',
    r'LumosCore(51.2T,1-$\tau$)',
    r'LumosCore(25.6T,1-$\tau$)',
    r'Clos(51.2T,3Tier)',
    r'Clos(25.6T,4Tier)',
    r'Clos(25.6T,3Tier)'
]

# 示例数据
data = {
    '2k': [
        get_finish_time(lumoscore_2tau, task_nums),  # LumosCore(51.2T,2-$\tau$)
        get_finish_time(lumoscore_1tau, task_nums),  # LumosCore(51.2T,1-$\tau$)
        get_finish_time(lumoscore_25_6t, task_nums),  # LumosCore(25.6T,1-$\tau$)
        get_finish_time(clos, task_nums),  # Clos(51.2T,3Tier)
        get_finish_time(clos_4tier_25_6t, task_nums),  # Clos(25.6T,4Tier)
        get_finish_time(clos_3tier_25_6t, task_nums)   # Clos(25.6T,3Tier)
    ],
    '4k': [
        get_finish_time(lumoscore_2tau, task_nums),  # LumosCore(51.2T,2-$\tau$)
        get_finish_time(lumoscore_1tau, task_nums),  # LumosCore(51.2T,1-$\tau$)
        get_finish_time(lumoscore_25_6t, task_nums),  # LumosCore(25.6T,1-$\tau$)
        get_finish_time(clos, task_nums),  # Clos(51.2T,3Tier)
        get_finish_time(clos_4tier_25_6t, task_nums),  # Clos(25.6T,4Tier)
        get_finish_time(clos_3tier_25_6t, task_nums)   # Clos(25.6T,3Tier)
    ],
    '8k': [
        get_finish_time(lumoscore_2tau, task_nums),  # LumosCore(51.2T,2-$\tau$)
        get_finish_time(lumoscore_1tau, task_nums),  # LumosCore(51.2T,1-$\tau$)
        get_finish_time(lumoscore_25_6t, task_nums),  # LumosCore(25.6T,1-$\tau$)
        get_finish_time(clos, task_nums),  # Clos(51.2T,3Tier)
        get_finish_time(clos_4tier_25_6t, task_nums),  # Clos(25.6T,4Tier)
        get_finish_time(clos_3tier_25_6t, task_nums)   # Clos(25.6T,3Tier)
    ],
    '16k': [
        get_finish_time(lumoscore_2tau, task_nums),  # LumosCore(51.2T,2-$\tau$)
        get_finish_time(lumoscore_1tau, task_nums),  # LumosCore(51.2T,1-$\tau$)
        get_finish_time(lumoscore_25_6t, task_nums),  # LumosCore(25.6T,1-$\tau$)
        get_finish_time(clos, task_nums),  # Clos(51.2T,3Tier)
        get_finish_time(clos_4tier_25_6t, task_nums),  # Clos(25.6T,4Tier)
        get_finish_time(clos_3tier_25_6t, task_nums)   # Clos(25.6T,3Tier)
    ]
}

# 计算平均值
mean_data = {
    'Scale': [],
    'Architecture': [],
    'Mean_JCT': []
}

scales = ['2k', '4k', '8k', '16k']

for i, scale in enumerate(scales):
    for j, arch in enumerate(architectures):
        mean_jct = np.mean(data[scale][j])
        mean_data['Scale'].append(scale)
        mean_data['Architecture'].append(arch)
        mean_data['Mean_JCT'].append(mean_jct)

df_mean = pd.DataFrame(mean_data)

# 绘制条形图
plt.figure(figsize=(3, 1.5))
barplot = sns.barplot(x='Scale', y='Mean_JCT', hue='Architecture', data=df_mean, palette=colors, errorbar=None)

# 为每个条形添加边框以增加对比度
for bar in barplot.patches:
    bar.set_edgecolor('black')  # 设置边框颜色为黑色
    bar.set_linewidth(1.5)       # 设置边框宽度

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 设置 X 轴、Y 轴标签和标题
plt.xlabel('Cluster Scale', fontsize=10)
plt.ylabel('Avg. JCT(s)', fontsize=10)

# 使用科学记数法格式化 Y 轴
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))
barplot.yaxis.set_major_formatter(formatter)

# 移除图例
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([], [], frameon=False)
plt.ylim((30000,50000))
# 保存直方图为 diff_jct.pdf
plt.savefig('diff_jct.pdf', bbox_inches='tight')
plt.close()

# 绘制图例
fig, ax = plt.subplots(figsize=(4, 1))
for i, key in enumerate(architectures):
    ax.plot([], [], color=colors[i], label=key, linewidth=2)
plt.legend(handles, labels, ncol=3, loc='center', frameon=False)
ax.axis('off')  # 关闭坐标轴

# 保存图例为 legend.pdf
plt.savefig('jct_legend.pdf', bbox_inches='tight', pad_inches=0.01)
plt.close()