from datetime import time
from pathlib import Path
import os
from re import T
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.core import numeric
import seaborn as sns
from numpy import dtype, histogram
import pandas as pd
from pandas import json_normalize
import math
import matplotlib.pyplot as plt


def json_processing(json_name, json_dir):
    with open(f'{json_dir}/{json_name}.json', 'r', encoding='utf-8-sig') as f:
        json_data = json.load(f)

    conf_lst = []
    x_lst = []
    y_lst = []
    width_lst = []
    height_lst = []
    f_no_lst = []

    for dictionary in json_data['frames']:
        if dictionary['frame'] == []:
            conf_lst.append(None)
            x_lst.append(None)
            y_lst.append(None)
            height_lst.append(None)
            width_lst.append(None)
        else:
            conf_lst.append(round(dictionary['frame'][0]['confidence'],2))
            x_lst.append(round(dictionary['frame'][0]['center_x'],2))
            y_lst.append(round(dictionary['frame'][0]['center_y'],2))
            width_lst.append(round(dictionary['frame'][0]['width'],2))
            height_lst.append(round(dictionary['frame'][0]['height'],2))

        f_no_lst.append(dictionary['frame_no'])
    
    #총 시간(단위 : 초)
    total_time = len(f_no_lst)
    print("영상의 시간(초) : ", total_time)

    #처리를 위해 모든 데이터에 대한 데이터 프레임 생성 및 csv로 저장
    all_data =[]
    for idx in range(total_time):
        all_data.append({
            'time' : idx+1,
            'x' : x_lst[idx],
            'y' : y_lst[idx],
        })
    df1 = pd.DataFrame(all_data, columns=['time', 'x', 'y'])
    
    df2 = df1.dropna()

    detect_time = df2['time'].tolist()
    detect_x = df2['x'].tolist()
    detect_y = df2['y'].tolist()
    length = len(detect_time)

    distance = []
    velocity = []
    activity = []
    act_times = []
    is_in = []

    for idx in range(length):
        

        if idx == 0:
            distance.append(None)
            velocity.append(None)
            activity.append(None)
            act_times.append(None)
            is_in.append(None)

        else:
            is_dog = 'living room'
            if detect_x[idx]>650 and detect_x[idx] < 730:
                if detect_y[idx] > 107 and detect_y[idx] < 120:
                    is_dog = 'near door'
            is_in.append(is_dog)        
            dist = round(math.sqrt((detect_x[idx] - detect_x[idx-1])**2 + (detect_y[idx] - detect_y[idx-1])**2), 2)
            distance.append(dist)
            act_time = detect_time[idx] - detect_time[idx-1]
            act_times.append(act_time)
            velo = round(dist / act_time, 2)
            velocity.append(velo)
            if velo > 35:
                activity.append('run')
            elif velo > 1:
                activity.append('walk')
            else:
                activity.append('rest')
    
    
        #x : 650~720 / y : 107 ~ 118 >> 현관문 근처에 있음
        

    df2['dist'] = distance
    df2['act time'] = act_times
    df2['velo'] = velocity
    df2['act'] = activity
    df2['where'] = is_in

    '''
              time       x       y    dist  act time   velo   act        where
        0        1   85.71  408.99     NaN       NaN    NaN  None         None
        4        5  462.90  479.46  383.72       4.0  95.93   run  living room
        5        6  439.43  486.96   24.64       1.0  24.64  walk  living room
        7        8  397.74  500.99   43.99       2.0  22.00  walk  living room
        8        9  421.05  475.54   34.51       1.0  34.51  walk  living room
        ...    ...     ...     ...     ...       ...    ...   ...          ...
        3795  3796  578.13  179.16    2.00       1.0   2.00  walk  living room
        3796  3797  573.10  175.44    6.26       1.0   6.26  walk  living room
        3797  3798  573.55  177.34    1.95       1.0   1.95  walk  living room
        3798  3799  573.04  178.48    1.25       1.0   1.25  walk  living room
        3799  3800  571.35  178.96    1.76       1.0   1.76  walk  living room
    '''

    #활동 시간을 분류 하여 움직인 거리와 활동 시간을 모두 더한다
    act_lst = ['rest', 'walk', 'run', 'Non_detect']
    act_dist_time_lst = []

    for act in act_lst:
        if act != 'Non_detect':
            df_act = df2[df2['act'] == act]
            dist = round(sum(df_act['dist'].to_list()), 2)
            time = len(df_act)
        else:
            dist = 0
            time = total_time - len(df2)
        
        act_dist_time_lst.append([act, dist, time])
        #[['rest', 404.18, 1338], ['walk', 14333.48, 1072], ['run', 16470.4, 214], ['Non_detect', 0, 1175]]    
    
    act_df = pd.DataFrame(columns=['act', 'distance', 'time'])
    for data in act_dist_time_lst:
        act_df = act_df.append(pd.Series(data, index = act_df.columns), ignore_index=True)


    #머문 장소에 따른 분류
    where_lst = ['living room', 'near door', 'Non_detect']
    where_time_lst = []

    for where in where_lst:
        if where != 'Non_detect':
            df_where = df2[df2['where'] == where]
            dist = round(sum(df_where['dist'].to_list()), 2)
            time = len(df_where)
        else:
            dist = 0
            time = total_time - len(df2)

        where_time_lst.append([where, dist, time])
        #[['living room', 29776.27, 1604], ['near door', 1431.79, 1020], ['Non_detect', 0, 1175]]
    
    where_df = pd.DataFrame(columns=['where', 'distance', 'time'])
    for data in where_time_lst:
        where_df = where_df.append(pd.Series(data, index=where_df.columns), ignore_index=True)



    #장소와 운동상태를 이용한 pivot테이블 작성
    act_lst = ['rest', 'walk', 'run']
    is_in_lst = ['living room', 'near door']

    lst = []
    for i in act_lst:
        for j in is_in_lst:
            lst.append({
                'act' : i,
                'where' : j,
                'count' : len(df2[(df2['where'] == j) & (df2['act'] == i)])
            })
            #print(len(df2[(df2['where'] == j) & (df2['act'] == i)]))

    df3 = pd.DataFrame(lst, columns = ['act', 'where', 'count'])
    non_detect_data = {'act' : 'non_detect', 'where' : 'non_detect', 'count' : total_time - len(df2)}
    df3 = df3.append(non_detect_data, ignore_index=True)
    pivot_df = df3.pivot('act', 'where', 'count')
    pivot_df = pivot_df.fillna(0)
    pivot_df = pivot_df.astype(int)
    pivot_df

    
    #차트로 그리기
    sns.relplot(x = 'time', y = 'velo', hue = 'where', data = df2)
    plt.title('time-velo')
    #plt.show()
    plt.savefig(os.path.join(json_dir,'./graph1.png'))

    sns.relplot(x = 'x', y = 'y', hue = 'act', data=df2)
    plt.title('x-y')
    #plt.show()
    plt.savefig(os.path.join(json_dir,'./graph2.png'))

    f, ax = plt.subplots(figsize=(6, 6))
    plt.title('heatmap')
    sns.heatmap(pivot_df, cmap = 'Blues', annot=True, fmt="d", linewidths=.5, ax=ax)
    #plt.show()
    plt.savefig(os.path.join(json_dir,'./graph3.png'))





    
    
    


if __name__ == '__main__':
    json_processing(json_data)
    

