import pandas as pd
import numpy as np
import time
from sklearn import linear_model
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import warnings

def lr_model(training_df, testing_df):
    start_time = time.time()
    #Data set split.
    #Data for training.
    X_train = training_df[['horse_index','jockey_index','trainer_index','draw','actual_weight','race_distance']].values
    y_train = training_df[['finishing_position']].values
    #Data for testing.
    X_test = testing_df[['horse_index','jockey_index','trainer_index','draw','actual_weight','race_distance']].values
    y_test = testing_df[['race_id', 'horse_id', 'horse_index','jockey_ave_rank', 'trainer_ave_rank','finishing_position']].values
    #Model training and predict.
    lr_model = linear_model.LogisticRegression()
    lr_model.fit(X_train, y_train[:,0])
    test_predict = lr_model.predict(X_test)
    #Label and foramt change.
    result = testing_df[['race_id', 'horse_id', 'horse_index', 'jockey_ave_rank', 'trainer_ave_rank']].values
    result = np.append(result, test_predict[:,None], axis = 1)
    result_df = pd.DataFrame(data=result,columns=['race_id', 'horse_id', 'horse_index','jockey_ave_rank', 'trainer_ave_rank', 'finishing_position'])
    result_df = reorder(result_df)
    result_df = add_All(result_df)
    result_df.to_csv(path_or_buf = 'predictions/lr_predictions.csv')  
    #Evaluation.
    truth_df = pd.DataFrame(data=y_test,columns=['race_id', 'horse_id', 'horse_index','jockey_ave_rank', 'trainer_ave_rank', 'finishing_position'])
    truth_df = add_All(truth_df)
    print("lr_model: ")
    prediction_Evaluation(result_df, truth_df)
    #Running time.
    running_time = time.time() - start_time
    print("Running time : %s seconds" % running_time)
    return result_df,truth_df

def nb_model(training_df, testing_df):
    start_time = time.time()
    #Data set split.
    #Data for training.
    X_train = training_df[['horse_index','jockey_index','trainer_index']].values
    y_train = training_df[['finishing_position']].values
    #Data for testing.
    X_test = testing_df[['horse_index','jockey_index','trainer_index']].values
    y_test = testing_df[['race_id', 'horse_id', 'horse_index','jockey_ave_rank', 'trainer_ave_rank','finishing_position']].values
    #Model training and predict.
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train[:,0])
    test_predict = nb_model.predict(X_test)
    #Label and foramt change.
    result = testing_df[['race_id', 'horse_id', 'horse_index', 'jockey_ave_rank', 'trainer_ave_rank']].values
    result = np.append(result, test_predict[:,None], axis = 1)
    result_df = pd.DataFrame(data=result,columns=['race_id', 'horse_id', 'horse_index','jockey_ave_rank', 'trainer_ave_rank', 'finishing_position'])
    result_df = reorder(result_df)
    result_df = add_All(result_df)
    result_df.to_csv(path_or_buf = 'predictions/nb_predictions.csv')  
    #Evaluation.
    truth_df = pd.DataFrame(data=y_test,columns=['race_id', 'horse_id', 'horse_index','jockey_ave_rank', 'trainer_ave_rank', 'finishing_position'])
    truth_df = add_All(truth_df)
    print("nb_model: ")
    prediction_Evaluation(result_df, truth_df)
    #Running time.
    running_time = time.time() - start_time
    print("Running time : %s seconds" % running_time)
    return result_df, truth_df

def svm_model(training_df, testing_df):
    start_time = time.time()
    #Data set split.
    #Data for training.
    X_train = training_df[['recent_ave_rank','jockey_ave_rank','trainer_ave_rank']].values
    y_train = training_df[['finishing_position']].values
    #Data for testing.
    X_test = testing_df[['recent_ave_rank','jockey_ave_rank','trainer_ave_rank']].values
    y_test = testing_df[['race_id', 'horse_id', 'horse_index','jockey_ave_rank', 'trainer_ave_rank','finishing_position']].values
    #Model training and predict.
    mi = 1500
    svm_model = svm.SVC(kernel='rbf', max_iter=mi)
    svm_model.fit(X_train, y_train[:,0])
    test_predict = svm_model.predict(X_test)
    #Label and foramt change.
    result = testing_df[['race_id', 'horse_id', 'horse_index', 'jockey_ave_rank', 'trainer_ave_rank']].values
    result = np.append(result, test_predict[:,None], axis = 1)
    result_df = pd.DataFrame(data=result,columns=['race_id', 'horse_id', 'horse_index','jockey_ave_rank', 'trainer_ave_rank', 'finishing_position'])
    result_df = reorder(result_df)
    result_df = add_All(result_df)
    result_df.to_csv(path_or_buf = 'predictions/svm_predictions.csv')  
    #Evaluation.
    truth_df = pd.DataFrame(data=y_test,columns=['race_id', 'horse_id', 'horse_index','jockey_ave_rank', 'trainer_ave_rank', 'finishing_position'])
    truth_df = add_All(truth_df)
    print("svm_model: ")
    print("Max iteration: ", mi)
    prediction_Evaluation(result_df, truth_df)
    #Running time.
    running_time = time.time() - start_time
    print("Running time : %s seconds" % running_time)
    return result_df, truth_df

def rf_model(training_df, testing_df):
    start_time = time.time()
    #Data set split.
    #Data for training.
    X_train = training_df[['recent_ave_rank','jockey_ave_rank','trainer_ave_rank']].values
    y_train = training_df[['finishing_position']].values
    #Data for testing.
    X_test = testing_df[['recent_ave_rank','jockey_ave_rank','trainer_ave_rank']].values
    y_test = testing_df[['race_id', 'horse_id', 'horse_index','jockey_ave_rank', 'trainer_ave_rank','finishing_position']].values
    #Model training and predict.
    md = 5
    rf_model = RandomForestClassifier(max_depth=md, random_state=0)
    rf_model.fit(X_train, y_train[:,0])
    test_predict = rf_model.predict(X_test)
    #Label and foramt change.
    result = testing_df[['race_id', 'horse_id', 'horse_index', 'jockey_ave_rank', 'trainer_ave_rank']].values
    result = np.append(result, test_predict[:,None], axis = 1)
    result_df = pd.DataFrame(data=result,columns=['race_id', 'horse_id', 'horse_index','jockey_ave_rank', 'trainer_ave_rank', 'finishing_position'])
    result_df = reorder(result_df)
    result_df = add_All(result_df)
    result_df.to_csv(path_or_buf = 'predictions/rf_predictions.csv')  
    #Evaluation.
    truth_df = pd.DataFrame(data=y_test,columns=['race_id', 'horse_id', 'horse_index','jockey_ave_rank', 'trainer_ave_rank', 'finishing_position'])
    truth_df = add_All(truth_df)
    print("rf_model: ")
    print("Max depth: ", md)
    prediction_Evaluation(result_df, truth_df)
    #Running time.
    running_time = time.time() - start_time
    print("Running time : %s seconds" % running_time)
    return result_df,truth_df

def convert_to_float_time(frac_str):
    if frac_str.isdigit():
        return float(frac_str)
    else:
        min, sec, subsec = frac_str.split('.')
        whole = float(min) * 60 + float(sec) + float(subsec) / 100
        return whole

def reorder(df):
    df_all = pd.DataFrame()
    raceID = df[['race_id']].drop_duplicates(subset='race_id',keep='last')
    for i in range(0, raceID.values.shape[0]):
        df_sub = df[df['race_id'].isin(raceID.values[i])]
        df_sub = df_sub.sort_values(by=['finishing_position'])
        df_sub = tie_Break(df_sub)
        frames = [df_all, df_sub]
        df_all = pd.concat(frames)
    df_all = df_all.sort_index()
    return df_all[['race_id', 'horse_id','horse_index','finishing_position']]

def tie_Break(df):
    df_all = pd.DataFrame()
    df['ave_rank'] = df['jockey_ave_rank'] + df['trainer_ave_rank']
    finishingPosition = df[['finishing_position']].drop_duplicates(subset='finishing_position',keep='last')
    for i in range(0, finishingPosition.values.shape[0]):
        df_sub = df[df['finishing_position'].isin(finishingPosition.values[i])]
        df_sub = df_sub.sort_values(by=['ave_rank'])
        frames = [df_all, df_sub]
        df_all = pd.concat(frames)
    df_all['finishing_position'] = [x for x in range(1, df_all.values.shape[0] + 1)]
    return df_all[['race_id', 'horse_id','horse_index','finishing_position']]

def add_All(df):
    df_all = pd.DataFrame()
    raceID = df[['race_id']].drop_duplicates(subset='race_id',keep='last')
    for i in range(0, raceID.values.shape[0]):
        df_sub = df[df['race_id'].isin(raceID.values[i])]
        df_sub['race_capacity'] = df_sub.values.shape[0]
        frames = [df_all, df_sub]
        df_all = pd.concat(frames)
    df_all['HorseWin'] = df_all.apply(lambda df: add_Label(df, label = 'HorseWin'), axis=1)
    df_all['HorseRankTop3'] = df_all.apply(lambda df: add_Label(df, label = 'HorseRankTop3'), axis=1)
    df_all['HorseRankTop50Percent'] = df_all.apply(lambda df: add_Label(df, label = 'HorseRankTop50Percent'), axis=1)
    df_all = df_all[['race_id', 'horse_id', 'HorseWin', 'HorseRankTop3', 'HorseRankTop50Percent']].values
    data_required = pd.DataFrame(data=df_all,columns=['RaceID', 'HorseID', 'HorseWin','HorseRankTop3', 'HorseRankTop50Percent'])
    return data_required
                
def add_Label(df, label):
    if label == 'HorseWin':
        if df['finishing_position'] == 1:
            return '1'
        else:
            return '0'
    if label == 'HorseRankTop3':
        if df['finishing_position'] <= 3:
            return '1'
        else:
            return '0'
    if label == 'HorseRankTop50Percent':
        if df['finishing_position'] <= df['race_capacity']/2:
            return '1'
        else:
            return '0'
    return -1

def prediction_Evaluation(result, truth):
    p_score = precision_score(np.asfarray(truth[['HorseWin']].values,float), np.asfarray(result[['HorseWin']].values,float))
    r_score = recall_score(np.asfarray(truth[['HorseWin']].values,float), np.asfarray(result[['HorseWin']].values,float))
    print("For Winner: ")
    print("p_score: ", p_score)
    print("r_score: ", r_score)
    p_score = precision_score(np.asfarray(truth[['HorseRankTop3']].values, float), np.asfarray(result[['HorseRankTop3']].values, float))
    r_score = recall_score(np.asfarray(truth[['HorseRankTop3']].values, float), np.asfarray(result[['HorseRankTop3']].values, float))
    print("For Top3: ")
    print("p_score: ", p_score)
    print("r_score: ", r_score)
    p_score = precision_score(np.asfarray(truth[['HorseRankTop50Percent']].values, float), np.asfarray(result[['HorseRankTop50Percent']].values, float))
    r_score = recall_score(np.asfarray(truth[['HorseRankTop50Percent']].values, float),np.asfarray(result[['HorseRankTop50Percent']].values, float))
    print("For Top50P: ")
    print("p_score: ", p_score)
    print("r_score: ", r_score)

warnings.filterwarnings('ignore')
training_df = pd.read_csv("training.csv")
training_df['finish_time'] = training_df['finish_time'].apply(lambda frac: convert_to_float_time(frac))
training_df = training_df.fillna(0)

testing_df = pd.read_csv("testing.csv")
temp_df_jockey_index = training_df[['jockey_index','jockey_ave_rank']].drop_duplicates(subset = 'jockey_index', keep='last')
temp_df_trainer_index = training_df[['trainer_index', 'trainer_ave_rank']].drop_duplicates(subset='trainer_index', keep='last')
testing_df = testing_df.merge(temp_df_jockey_index, left_on = 'jockey_index', right_on = 'jockey_index',how = 'left')
testing_df = testing_df.merge(temp_df_trainer_index, left_on='trainer_index', right_on='trainer_index', how='left')
testing_df['jockey_ave_rank'].fillna(7,inplace = True)
testing_df['trainer_ave_rank'].fillna(7, inplace=True)
    
lr_result, lr_truth = lr_model(training_df, testing_df)
nb_result, nb_truth = nb_model(training_df, testing_df)
svm_result, svm_truth = svm_model(training_df, testing_df)
rf_result, rf_truth = rf_model(training_df, testing_df)