from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_csv_get_beta(filepath):
    df = pd.read_csv(filepath, header=None, usecols=[1, 2, 3], names=['taskid', 'type', 'time'])
    df_pivot = df.pivot(index='taskid', columns='type', values='time')
    df_pivot.reset_index(inplace=True)
    df_pivot.columns.name = None
    df_pivot.columns = ['taskid', 'arriving_time', 'end_time', 'start_time']
    return df_pivot


def load_and_calculate(exp_name):
    data = load_csv_get_beta(f'{exp_name}/task_time.log')
    arriving_time = data['arriving_time'].mean()
    start_time = data['start_time'].mean()
    end_time = data['end_time'].mean()
    jrt = round(end_time - start_time, 3)
    jwt = round(start_time - arriving_time, 3)
    jct = round(end_time - arriving_time, 3)
    return jrt, jwt, jct


def get_jct(exp_name):

    exp_data = {}
    jrt, jwt, jct = load_and_calculate(exp_name)
    exp_data["JRT"] = jrt
    exp_data["JWT"] = jwt
    exp_data["JCT"] = jct
    return float(exp_data['JCT'])


def main():
    exps = ('lumoscore_tau_1_26T', 'ele_26T')
    scales = (2048, 4096, 8192, 16384)
    prefixes = ('/beta_9800', '_routing_analysis/rehashing', '/beta_820', '/beta_250')
    for scale, prefix in zip(scales, prefixes):
        jcts = []
        for exp in exps:
            if scale == 16384 and exp.startswith('lum'):
                jct = get_jct(f'./large_exp_{scale}GPU{prefix}/lumoscore_tau_2')
            else:
                jct = get_jct(f'./large_exp_{scale}GPU{prefix}/{exp}')
            jcts.append(jct)
        print(1 - jcts[0] / jcts[1])


if __name__ == "__main__":
    main()
