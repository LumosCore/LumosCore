from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.sans-serif'] = ['Times New Roman']

styles = ['-', '-.', '--', ':', 'solid',
          'dashed', 'dotted', 'dashdot', 'dashed']
markers = [' ', '>', '8', '*', 'x', '+', 'p', 'D']
colors = ["red", "green", "blue", "c", "cyan",
          "brown", "mediumvioletred", "dodgerblue", "orange"]


def load_csv_get_beta(filepath):
    df = pd.read_csv(filepath, header=None, usecols=[1, 2, 3], names=['taskid', 'type', 'time'])
    df_pivot = df.pivot(index='taskid', columns='type', values='time')
    df_pivot.reset_index(inplace=True)
    df_pivot.columns.name = None
    df_pivot.columns = ['taskid', 'arriving_time', 'end_time', 'start_time']
    return df_pivot


def draw_cdf_from_dict(data_dict):
    """绘制CDF图
    Input: 接受任意数量的数据，key充当画图的图例，value是画图用的原始数据
    """
    # plt.figure(figsize=(6, 4))
    # 适配曲线数量
    count = 0
    for k, data in data_dict.items():
        data = list(data)
        y = data
        x = [i for i in range(len(y))]
        plt.plot(x, y, label=k,
                 linestyle=styles[count], color=colors[count], linewidth=2.5)

        count += 1

    # plt.ylim(0.8, 1)
    # plt.xlim(0, 500)
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)
    # plt.yscale("symlog", linthreshy=0.0001)
    plt.xlabel("Taskid", fontsize=24)
    plt.ylabel("Completion Time", fontsize=24)
    plt.grid()
    plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower center", fontsize=23,
               mode="expand", borderaxespad=0, ncol=2, frameon=False,
               handletextpad=0.1, handlelength=1)
    return plt


def cal_fairness(running_time_list, duration_time_list):
    sum_fariness = 0
    for i in range(len(running_time_list)):
        sum_fariness += running_time_list[i] / duration_time_list[i]
    return sum_fariness / len(running_time_list)


def load_and_calculate(exp_name):
    data = load_csv_get_beta(f'{exp_name}/task_time.log')
    arriving_time = data['arriving_time'].mean()
    start_time = data['start_time'].mean()
    end_time = data['end_time'].mean()
    jrt = round(end_time - start_time, 3)
    jwt = round(start_time - arriving_time, 3)
    jct = round(end_time - arriving_time, 3)
    return jrt, jwt, jct


def main(exp_names):
    exp_data = {"JRT": [], "JWT": [], "JCT": []}
    for exp_name in exp_names:
        jrt, jwt, jct = load_and_calculate(exp_name)
        exp_data["JRT"].append(str(jrt))
        exp_data["JWT"].append(str(jwt))
        exp_data["JCT"].append(str(jct))
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    with open(f'JCT/exp_data_{now}.csv', mode='w') as f:
        f.write('target,')
        f.write(','.join(exp_names))
        f.write('\n')
        for k, v in exp_data.items():
            f.write(f"{k},{','.join(v)}\n")
        # f.write('\n')
        # for k, v in exp_data.items():
        #     f.write(f"{k}& {'& '.join(v)}\n")


if __name__ == "__main__":
    # cal_value()
    main([
        "best",
        "lumoscore_tau_2",
        "lumoscore_tau_1",
        "ele",
        "ele_26T",
        "lumoscore_tau_1_26T",
    ])
