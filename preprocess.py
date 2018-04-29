import numpy as np
import sklearn as sk
import pandas as pd

#2.2.1 Read data and clean.
data = pd.read_csv('data/race-result-horse.csv')
#Drop rows where "finish_position" is not a number.
data = data[pd.to_numeric(data['finishing_position'], errors='coerce').notnull()]

#2.2.2 Recent performance.
#Add a column of "recent_6_runs".
data['recent_6_runs'] = data.finishing_position.astype(str)+'/'
#Find out all past race results.
data['recent_6_runs'] = data.groupby('horse_id').recent_6_runs.apply(lambda x : x.cumsum().shift().fillna('').str[:-1])
#Keep the recent 6 record(the original data is actually reverse order).
def clean(string):
    count = string.count('/') + 1
    if count > 6:
        return string.split('/',count - 5)[-1]
    else:
        return string
data['recent_6_runs'] = data.recent_6_runs.apply(lambda x : clean(x))
#Calculate the average.
def ave(string):
    count = string.count('/') + 1
    if string != '':
        number = np.array([int(x) for x in np.array(string.split('/'))])
        return np.sum(number)/(count)
    else:
        return 7
data['recent_ave_rank'] = data.recent_6_runs.apply(lambda x : ave(x))

#2.2.3 Indexes
#Horse.
horse = np.array(data.horse_id.unique())
def get_horse_index(string):
        return np.where(horse == string)[0][0]
data['horse_index'] = data.horse_id.apply(lambda x: get_horse_index(x))
#Jockey.
jockey = np.array(data.jockey.unique())
def get_jockey_index(string):
        return np.where(jockey == string)[0][0]
data['jockey_index'] = data.jockey.apply(lambda x: get_jockey_index(x))
#Trainer.
trainer = np.array(data.trainer.unique())
def get_trainer_index(string):
        return np.where(trainer == string)[0][0]
data['trainer_index'] = data.trainer.apply(lambda x: get_trainer_index(x))
#Calculate the average rank of the jockey and trainer.
#Caster into int type.
data['finishing_position'] = data['finishing_position'].astype(int)

#Import the race distance data before split(2.2.4).
race = pd.read_csv("data/race-result-race.csv")
temp_race = pd.DataFrame()
temp_race = race[['race_id', 'race_distance']]
temp_race = temp_race.drop_duplicates(subset='race_id', keep='last')
data = data.merge(temp_race, left_on='race_id', right_on='race_id', how='left')

#In order to do the ranking, split first.
test_data = data.loc[data['race_id'] > '2016-327']
train_data = data.loc[data['race_id'] <= '2016-327']
#Average rank of jockey.
temp_j = pd.DataFrame()
temp_j = train_data[['jockey_index', 'finishing_position']]
temp_j['jockey_ave_rank'] = temp_j.jockey_index.map(temp_j.groupby(['jockey_index']).finishing_position.mean())
temp_j = temp_j.drop_duplicates(subset='jockey_index', keep='last')
temp_j = temp_j.drop('finishing_position',axis = 1)
train_data = train_data.merge(temp_j, left_on='jockey_index', right_on='jockey_index', how='left')
train_data['jockey_ave_rank'].fillna(7,inplace=True)
#Average rank of trainer.
temp_t = pd.DataFrame()
temp_t = train_data[['trainer_index', 'finishing_position']]
temp_t['trainer_ave_rank'] = temp_t.trainer_index.map(temp_t.groupby(['trainer_index']).finishing_position.mean())
temp_t = temp_t.drop_duplicates(subset='trainer_index', keep='last')
temp_t = temp_t.drop('finishing_position',axis = 1)
train_data = train_data.merge(temp_t, left_on='trainer_index', right_on='trainer_index', how='left')
train_data['trainer_ave_rank'].fillna(7,inplace=True)

#2.2.5 Save to csv file.
train_data.to_csv(path_or_buf='traing.csv')
test_data.to_csv(path_or_buf='test.csv')