# CHEN Jiamin

from __future__ import print_function
import numpy as np
import pandas as pd
import time
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor as GBRT
from sklearn.metrics import mean_squared_error as MSE

# Read data from csv files
train_raw_data = pd.read_csv('training.csv')
test_raw_data = pd.read_csv('testing.csv')

# Test: Average rank of jockey.
temp_j = pd.DataFrame()
temp_j = test_raw_data[['jockey_index', 'finishing_position']]
temp_j['jockey_ave_rank'] = temp_j.jockey_index.map(temp_j.groupby(['jockey_index']).finishing_position.mean())
temp_j = temp_j.drop_duplicates(subset='jockey_index', keep='last')
temp_j = temp_j.drop('finishing_position',axis = 1)
test_raw_data = test_raw_data.merge(temp_j, left_on='jockey_index', right_on='jockey_index', how='left')
test_raw_data['jockey_ave_rank'].fillna(7,inplace=True)
# Test: Average rank of trainer.
temp_t = pd.DataFrame()
temp_t = test_raw_data[['trainer_index', 'finishing_position']]
temp_t['trainer_ave_rank'] = temp_t.trainer_index.map(temp_t.groupby(['trainer_index']).finishing_position.mean())
temp_t = temp_t.drop_duplicates(subset='trainer_index', keep='last')
temp_t = temp_t.drop('finishing_position',axis = 1)
test_raw_data = test_raw_data.merge(temp_t, left_on='trainer_index', right_on='trainer_index', how='left')
test_raw_data['trainer_ave_rank'].fillna(7,inplace=True)

# Select features and convert dataframe to array
train_data = train_raw_data[['actual_weight','declared_horse_weight','draw',
                        'win_odds','jockey_ave_rank','trainer_ave_rank',
                        'recent_ave_rank','race_distance','finish_time']].values
test_data = test_raw_data[['actual_weight','declared_horse_weight','draw',
                        'win_odds','jockey_ave_rank','trainer_ave_rank',
                        'recent_ave_rank','race_distance','finish_time']].values

X_train = train_data[:,:8]
y_train = train_data[:,8]
X_test = test_data[:,:8]
y_test = test_data[:,8]

# 4.2
#X_train_scaler = StandardScaler()
#y_train_scaler = StandardScaler()
#X_test_scaler = StandardScaler()
#X_train = train_scaler.fit_transform(X_train)
#y_train = train_scaler.fit_transform(y_train)
#X_test = test_scaler.fit_transform(X_test)



def timeinseconds(timestring):
    if timestring.isdigit():
        return float(frac_str)
    else:
        min,sec,point = timestring.split(".")
        return float(min)*60+float(sec)+float(point)/100.0
y_train = map(lambda x: timeinseconds(x),y_train)
y_test = map(lambda x: timeinseconds(x),y_test)
# Max iteration
mi = 3000

# 4.1.1
def svr_model(X, y, X_test, y_test):
    print("SVR:")
    start_time = time.time()
    svr_model = SVR(kernel='rbf', C=1e3, gamma=0.1,max_iter=mi)
    rbf = svr_model.fit(X, y).score(X, y)
    #print(rbf)
    #result = svr_model.predict(X_test)
    #evaluation(result,y_test)
    # Run time
    running_time = time.time() - start_time
    print("Running time : %s seconds" % running_time)

#4.1.2
def gbrt_model(X, y, X_test, y_test):
    print("GBRT:")
    start_time = time.time()
    gbrt_model = GBRT(loss='huber')
    huber = gbrt_model.fit(X, y).score(X, y)
    #print(huber)
    #result = gbrt_model.predict(X_test)
    #evaluation(result,y_test)
    # Run time
    running_time = time.time() - start_time
    print("Running time : %s seconds" % running_time)

# 4.2
def evaluation(result,truth):
    #result = y_train_scaler.inverse_transform(result)
    RMSE = np.sqrt(MSE(truth,result))
    top1_sum = 0
    top3_sum = 0
    rank_sum = 0
    race_sum = 0
    race_id = ""
    i = 0
    start_id = -1
    top_id = -1
    while (i < len(result)):
        if (race_id != test_raw_data['race_id'][i]):
            if (race_id != ""):
                rank_sum += (np.argsort(result[start_id:i])[0] + 1)
                if (test_raw_data['finishing_position'][top_id] == 1):
                    top1_sum += 1
                if (test_raw_data['finishing_position'][top_id] <= 3):
                    top3_sum += 1
            start_id = i
            race_id = test_raw_data['race_id'][i]
            race_sum += 1
            top_id = i
        elif (result[i] < result[top_id]):
                top_id = i
        # Next sample
        i += 1

    # the last race in the list
    rank_sum += (np.argsort(result[start_id:len(result)])[0] + 1)
    if (test_raw_data['finishing_position'][top_id] == 1):
        top1_sum += 1
    if (test_raw_data['finishing_position'][top_id] <= 3):
        top3_sum += 1
    print("Top1: ", top1_sum * 1.0/race_sum)
    print("Top3: ", top1_sum * 1.0/race_sum)
    print("Average_rank: ", rank_sum * 1.0/race_sum)

# 4.2
svr_model(X_train,y_train,X_test,y_test)
gbrt_model(X_train,y_train,X_test,y_test)
