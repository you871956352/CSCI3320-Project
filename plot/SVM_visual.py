import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


train = pd.read_csv('../training.csv')

train = train[['finishing_position','horse_number','horse_name','horse_id','jockey','trainer','actual_weight','declared_horse_weight','draw','length_behind_winner','running_position_1','running_position_2','running_position_3','running_position_4','finish_time','win_odds','running_position_5','running_position_6','race_id','recent_6_runs','recent_ave_rank','jockey_ave_rank','trainer_ave_rank','race_distance']]

X = train[['recent_ave_rank', 'jockey_ave_rank']].values

train_win3 = np.copy(train['finishing_position'].values)
X01 = []
X00 = []
X11 = []
X10 = []
y0 = []
y1 = []

for i in range(0, train_win3.shape[0]):
  race = train['race_id'][i]
  record = train.loc[train['race_id'] == race]
  horses = record.shape[0]
  if train_win3[i] <= (float(horses)/2) :
    train_win3[i] = 1
    X01 = np.append(X01, X[i][0])
    X11 = np.append(X11, X[i][1])
    y1= np.append(y1, 1)
  else:
    train_win3[i] = 0
    X00 = np.append(X00, X[i][0])
    X10 = np.append(X10, X[i][1])
    y0 = np.append(y0, 0)
train['rank_top_50_percent'] = train_win3

y = train['rank_top_50_percent']

'''
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target
print(y)
'''

model = SVC(kernel='linear')
model.fit(X,y)
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X00, X10, s=20, c='k', alpha=0.6, label='Rank Last 50 Percent')
plt.scatter(X01, X11, s=20, c='r', alpha=0.6, label='Rank Top 50 Percent')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Recent Average Rank')
plt.ylabel('Jockey Average Rank')
plt.title("SVM Visualization")
plt.legend(loc=2)
plt.show()
