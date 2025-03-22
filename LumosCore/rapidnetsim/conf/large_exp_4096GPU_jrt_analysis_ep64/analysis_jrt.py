import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 示例数据：任务运行时间（单位：秒）
np.random.seed(42)  # 设置随机种子以确保结果可重复
num_tasks = 100

# 生成随机数据以模拟100个任务在六种架构上的运行时间
lumoscore_2tau_times = [np.random.normal(loc=np.random.uniform(0.5, 1.5), scale=0.2) for _ in range(num_tasks)]
lumoscore_1tau_times = [np.random.normal(loc=np.random.uniform(0.6, 1.6), scale=0.2) for _ in range(num_tasks)]
lumoscore_25_6t_times = [np.random.normal(loc=np.random.uniform(0.7, 1.7), scale=0.2) for _ in range(num_tasks)]
clos_times = [np.random.normal(loc=np.random.uniform(0.8, 1.8), scale=0.2) for _ in range(num_tasks)]
clos_4tier_25_6t_times = [np.random.normal(loc=np.random.uniform(0.9, 1.9), scale=0.2) for _ in range(num_tasks)]
clos_3tier_25_6t_times = [np.random.normal(loc=np.random.uniform(1.0, 2.0), scale=0.2) for _ in range(num_tasks)]

# 将RGB值转换为Matplotlib可识别的格式 (0-1范围)
color_lumoscore_2tau = tuple(x / 255 for x in (169, 111, 176))
color_lumoscore_1tau = tuple(x / 255 for x in (216, 160, 199))
color_lumoscore_25_6t = tuple(x / 255 for x in (247,167,181))
color_clos = tuple(x / 255 for x in (43, 48, 122))
color_clos_4tier_25_6t = tuple(x / 255 for x in (119, 194, 243))
color_clos_3tier_25_6t = tuple(x / 255 for x in (218, 226, 237))

# 创建DataFrame
data = {
    'Architecture': ['LumosCore(2-$\\tau$)'] * num_tasks +
                    ['LumosCore(1-$\\tau$)'] * num_tasks +
                    ['LumosCore(25.6T)'] * num_tasks +
                    ['Clos(51.2T)'] * num_tasks +
                    ['Clos(4-tier,25.6T)'] * num_tasks +
                    ['Clos(3-tier,25.6T)'] * num_tasks,
    'JRT': lumoscore_2tau_times + lumoscore_1tau_times + lumoscore_25_6t_times + clos_times + clos_4tier_25_6t_times + clos_3tier_25_6t_times
}

df = pd.DataFrame(data)

# 定义颜色映射
palette = {
    'LumosCore(2-$\\tau$)': color_lumoscore_2tau,
    'LumosCore(1-$\\tau$)': color_lumoscore_1tau,
    'LumosCore(25.6T)': color_lumoscore_25_6t,
    'Clos(51.2T)': color_clos,
    'Clos(4-tier,25.6T)': color_clos_4tier_25_6t,
    'Clos(3-tier,25.6T)': color_clos_3tier_25_6t
}

# 创建图形
plt.figure(figsize=(7, 3), facecolor='none')  # 设置背景为透明

# 绘制小提琴图


# # 绘制箱线图

sns.boxplot(
        data=df,
        x='Architecture',
        y='JRT',
        hue='Architecture',
        palette=palette,
        width=0.4,
        whis=[0, 100],
        meanprops={"marker": "s", "markerfacecolor": "white", "markersize": 8},
        boxprops={'zorder': 2},
        medianprops={'color': 'black'},
        whiskerprops={'color': 'gray'},
        capprops={'color': 'gray'}
    )

sns.violinplot(
        data=df,
        x='Architecture',
        y='JRT',
        hue='Architecture',
        palette=palette,
        inner=None,
        width=0.7,
        alpha=0.5
    )

# 绘制点图

sns.stripplot(
        data=df,
        x='Architecture',
        y='JRT',
        hue='Architecture',
        dodge=True,
        palette=palette,
        s=4,
        alpha=0.7,
        jitter=0.1
    )

# 移除X轴标签
plt.xlabel('')

# 添加标题和Y轴标签
# plt.title('Job Runtime of Tasks on Different Architectures', fontsize=14, fontweight='bold')
plt.ylabel('Job Runtime (seconds)', fontsize=14, fontweight='bold')

# 调整刻度字体大小
plt.xticks(fontsize=8)
plt.yticks(fontsize=14)

# 添加网格
plt.grid(True, linestyle=':', linewidth=1.5, alpha=0.7, axis='y')

# 自动调整布局
plt.tight_layout()

# 显示图形
plt.show()
plt.savefig('dis_of_jrt.png')



average_jrts = [
    avg_lumoscore_2tau,
    avg_lumoscore_1tau,
    avg_lumoscore_25_6t,
    avg_clos,
    avg_clos_4tier_25_6t,
    avg_clos_3tier_25_6t
]

colors = [
    color_lumoscore_2tau,
    color_lumoscore_1tau,
    color_lumoscore_25_6t,
    color_clos,
    color_clos_4tier_25_6t,
    color_clos_3tier_25_6t
]

# 创建图形
plt.figure(figsize=(8.5, 5.5), facecolor='none')  # 半张A4纸大小

# 绘制条形图
bars = plt.bar(
    architectures,
    average_jrts,
    color=colors,
    edgecolor='black'
)

# 添加标题和轴标签
plt.title('Average Job Runtime of Tasks on Different Architectures', fontsize=14, fontweight='bold')
plt.xlabel('Architecture', fontsize=12, fontweight='bold')
plt.ylabel('Average Job Runtime (seconds)', fontsize=12, fontweight='bold')

# 调整刻度字体大小
plt.xticks(fontsize=10, rotation=45, ha='right')
plt.yticks(fontsize=10)

# 在每个柱子上显示数值
for bar, avg_jrt in zip(bars, average_jrts):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', va='bottom', ha='center', fontsize=9)

# 添加网格
plt.grid(True, linestyle=':', linewidth=0.5, alpha=0.7, axis='y')

# 自动调整布局
plt.tight_layout()

# 显示图形
plt.show()

plt.savefig('avg_jrt.png')