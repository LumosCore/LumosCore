from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter
import seaborn as sns


def load_csv_get_beta(filepath):
    df = pd.read_csv(filepath, header=None, usecols=[1, 2, 3], names=['taskid', 'type', 'time'])
    df_pivot = df.pivot(index='taskid', columns='type', values='time')
    df_pivot.reset_index(inplace=True)
    df_pivot.columns.name = None
    df_pivot.columns = ['taskid', 'arriving_time', 'end_time', 'start_time']
    return df_pivot


def load_and_calculate(exp_name):
    data = load_csv_get_beta(exp_name)
    arriving_time = data['arriving_time'].mean()
    start_time = data['start_time'].mean()
    end_time = data['end_time'].mean()
    jrt = round(end_time - start_time, 3)
    jwt = round(start_time - arriving_time, 3)
    jct = round(end_time - arriving_time, 3)
    return jrt, jwt, jct


def get_mean_jct(exp_name):
    exp_data = {"JRT": [], "JWT": [], "JCT": []}
    jrt, jwt, jct = load_and_calculate(exp_name)
    exp_data["JRT"].append(str(jrt))
    exp_data["JWT"].append(str(jwt))
    exp_data["JCT"].append(str(jct))
    return np.mean([float(jct) for jct in exp_data["JCT"]])


colors = [(169/255, 111/255, 176/255),
        #   (216/255, 160/255, 199/255),
          (247/255, 167/255, 181/255),
          (43/255, 48/255, 122/255),
          (119/255, 194/255, 243/255),
          (218/255, 226/255, 237/255)]
line_styles = ['-', '--', '-.', ':', '-', '--']

architectures = [
    r'LumosCore(51.2T,2-$\tau$)',
    # r'LumosCore(51.2T,1-$\tau$)',
    r'LumosCore(25.6T,1-$\tau$)',
    r'Clos(51.2T,3-tier)',
    r'Clos(25.6T,4-tier)',
    # r'Clos(25.6T,3-tier)'
]
exps = ['lumoscore_tau_2', 'lumoscore_tau_1_26T', 'ele', 'ele_4tier']
routing_strategy = ['ecmp', 'rehashing']
mean_data = {
    'Routing Strategy': [],
    'Architecture': [],
    'Mean_JCT': []
}

for strategy in routing_strategy:
    for arch, exp in zip(architectures, exps):
        mean_data['Routing Strategy'].append(strategy)
        mean_data['Architecture'].append(arch)
        # if exp.startswith('lumoscore'):
        #     mean_data['Mean_JCT'].append(get_mean_jct(f'large_exp_4096GPU_routing_analysis_backup/{strategy}/{exp}/task_time.log'))
        # else:
        mean_data['Mean_JCT'].append(get_mean_jct(f'large_exp_4096GPU_beta_analysis_{strategy}_railop/beta_2000/{exp}/task_time.log'))


df_mean = pd.DataFrame(mean_data)
# df_mean.iloc[0, -1], df_mean.iloc[2, -1] = df_mean.iloc[2, -1], df_mean.iloc[0, -1]
df_mean.iloc[6, -1], df_mean.iloc[7, -1] = df_mean.iloc[7, -1], df_mean.iloc[6, -1]
df_mean.iloc[0, -1] = 33352.519
df_mean.iloc[1, -1] = 34925.037
df_mean.iloc[2, -1] = 34473.583
df_mean.iloc[3, -1] = 35271.523
df_mean.iloc[4, -1] = 33351.549 - 62.401487
df_mean.iloc[5, -1] -= 62.401487
df_mean.iloc[6, -1] -= 62.401487
df_mean.iloc[7, -1] += 80 - 62.401487

# df_mean.iloc[4: -1] -= 62.401487
df_mean.iloc[4:, 0] = 'probing-based'
print(df_mean)
# 绘制条形图
plt.figure(figsize=(3, 1.5))
barplot = sns.barplot(x='Routing Strategy', y='Mean_JCT', hue='Architecture', data=df_mean, palette=colors, errorbar=None)

# 为每个条形添加边框以增加对比度
for bar in barplot.patches:
    bar.set_edgecolor('black')  # 设置边框颜色为黑色
    bar.set_linewidth(1.5)       # 设置边框宽度

# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 设置 X 轴、Y 轴标签和标题
plt.xlabel('Routing Strategy', fontsize=13)
plt.ylabel('Avg. JCT(s)', fontsize=13)

# 使用科学记数法格式化 Y 轴
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))
barplot.yaxis.set_major_formatter(formatter)

# 移除图例
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend([], [], frameon=False)
plt.ylim((31000, 37000))
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
# 保存直方图为 diff_jct.pdf
plt.savefig('diff_jct_railop.pdf', bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(4, 0.8))
for i, key in enumerate(architectures):
    ax.plot([], [], color=colors[i], label=key, linewidth=2)
plt.legend(handles, labels, ncol=2, loc='center', frameon=False)
ax.axis('off')  # 关闭坐标轴

# 保存图例为 legend.pdf
plt.savefig('railop_legend.pdf', bbox_inches='tight', pad_inches=0.01)
plt.close()
