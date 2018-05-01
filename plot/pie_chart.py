import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import csv
seaborn.set()

train = pd.read_csv('../training.csv',
            header=0,
            names=['','finishing_position','horse_number','horse_name','horse_id','jockey','trainer','actual_weight','declared_horse_weight','draw','length_behind_winner','running_position_1','running_position_2','running_position_3','running_position_4','finish_time','win_odds','running_position_5','running_position_6','race_id','recent_6_runs','recent_ave_rank','jockey_ave_rank','trainer_ave_rank','race_distance'],index_col=False,na_values=[])

draw = np.arange(1,15)
win_rate = []
win_rate = np.asarray(win_rate)
for i in draw:
  temp = train
  temp = temp.loc[temp['draw'] == i]
  appear = temp.shape[0]
  temp = temp.loc[temp['finishing_position'] == 1]
  win = temp.shape[0]
  win_rate = np.append(win_rate, win)

plt.pie(win_rate, labels=draw, autopct='%.0f%%')
plt.title("Correspondence Between Draw and Winning Rate")
plt.xlabel("1-14: Draw Number")
plt.show()