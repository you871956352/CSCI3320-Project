import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
import matplotlib.pyplot as plt

train_data = pd.read_csv('../training.csv')

train = train_data[['actual_weight','declared_horse_weight','draw','win_odds','jockey_ave_rank','trainer_ave_rank','recent_ave_rank','race_distance']].values.astype('float')

X_train_numeric = train
y_train = train_data['finishing_position'].values.astype('int')
y_train[y_train!=1] = 0

print(X_train_numeric.shape)
print(y_train.shape)
print(train_data.actual_weight[0])
rf_model = RFC(max_depth=10, random_state=0)
rf_model.fit(X_train_numeric, y_train)
importatnce_list = rf_model.feature_importances_


features_names = ['actual\n_weight','declared_\nhorse_weight','draw','win_odds','jockey_\nave_rank','trainer\n_ave_rank','recent_\nave_rank','race_\ndistance']
sorted_features = [x for _, x in sorted(zip(importatnce_list,features_names), key=lambda pair: pair[0],reverse=True)]
sorted_importance = sorted(importatnce_list,reverse=True)

y_pos = np.arange(len(sorted_features))

plt.bar(y_pos, sorted_importance, align='center', alpha=0.5)
plt.xticks(y_pos, sorted_features,size=6,rotation='vertical')
plt.ylabel('feature importance')
plt.title('Feature Importance Bar Chart')

plt.show()
