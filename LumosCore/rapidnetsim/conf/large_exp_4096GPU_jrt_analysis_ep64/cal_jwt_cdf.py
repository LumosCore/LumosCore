import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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



# 定义数据
architectures = ["LumosCore(2-$\\tau$)", "LumosCore(1-$\\tau$)", "LumosCore(25.6T)", "Clos", "Clos(4-tier,25.6T)", "Clos(3-tier,25.6T)"]
colors = [(169/255, 111/255, 176/255), (216/255, 160/255, 199/255), 
          (247/255, 238/255, 246/255), (43/255, 48/255, 122/255), 
          (119/255, 194/255, 243/255), (218/255, 226/255, 237/255)]
line_styles = ['-', '--', '-.', ':', '-', '--']
num_tasks = 1000
task_nums = num_tasks
best = load_csv_get_beta(f'best_b30000/task_time.log')
lumoscore_2tau = load_csv_get_beta(f'lumoscore_b30000_tau_2/task_time.log')
lumoscore_1tau = load_csv_get_beta(f'lumoscore_b30000_tau_1/task_time.log')
lumoscore_25_6t = load_csv_get_beta(f'lumoscore_b30000_tau_1_26T/task_time.log')
clos = load_csv_get_beta(f'ele_b30000/task_time.log')
clos_4tier_25_6t = load_csv_get_beta(f'ele_b30000_4tier/task_time.log')
clos_3tier_25_6t = load_csv_get_beta(f'ele_b30000_4tier_new/task_time.log')

oct_list = np.array(get_wait_time(best, task_nums))
lumoscore_2tau_times = get_wait_time(lumoscore_2tau, task_nums)
lumoscore_1tau_times = get_wait_time(lumoscore_1tau, task_nums)
lumoscore_25_6t_times = get_wait_time(lumoscore_25_6t, task_nums)
clos_times = get_wait_time(clos, task_nums)
clos_4tier_25_6t_times = get_wait_time(clos_4tier_25_6t, task_nums)
clos_3tier_25_6t_times = get_wait_time(clos_3tier_25_6t, task_nums)


#  Sample data for JRT (Job Run Time) of each architecture

jrt_data = {
    'LumosCore(2-$\\tau$)': lumoscore_2tau_times,
    'LumosCore(1-$\\tau$)': lumoscore_1tau_times,
    'LumosCore(25.6T)': lumoscore_25_6t_times,
    'Clos(51.2T)': clos_times,
    'Clos(4-tier,25.6T)': clos_4tier_25_6t_times,
    'Clos(3-tier,25.6T)': clos_3tier_25_6t_times,
}

# Colors and line styles for each architecture
colors = [(169/255, 111/255, 176/255), 
          (216/255, 160/255, 199/255), 
          (247/255, 167/255, 181/255), 
          (43/255, 48/255, 122/255), 
          (119/255, 194/255, 243/255), 
          (218/255, 226/255, 237/255)]
line_styles = ['-', '--', '-.', ':', '-', '--']



# Calculate drag factor for each architecture
drag_factors = {}
for key, jrt in jrt_data.items():
    drag_factor = jrt 
    drag_factors[key] = drag_factor

# Create CDF plot
plt.figure(figsize=(4, 3))
for i, (key, value) in enumerate(drag_factors.items()):
    sorted_data = np.sort(value)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    plt.plot(sorted_data, cdf, color=colors[i], linestyle=line_styles[i], linewidth=2)

plt.ylim(bottom=0.7)  # Set y-axis to start from 0.5
plt.xlabel('Job Waiting Time(s)', fontsize=12)  # Add x-axis label with font size
plt.yticks([0.7, 0.8, 0.9, 1.0], fontsize=10)  # Set y-ticks and font size
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines

# Save the CDF plot to a PDF file
plt.savefig('cdf_of_jwt.pdf', bbox_inches='tight')
plt.close()




