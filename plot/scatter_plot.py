# CHEN Jiamin
import matplotlib.pyplot as plt
import pandas as pd


# Read data from preprocessed csv
df = pd.read_csv('../data/race-result-horse.csv')

horses = df.horse_id.unique()
horse_wins = [0] * len(horses)
horse_winrate = [0] * len(horses)
horse_label = [0] * len(horses)

jockeys = df.jockey.unique()
jockey_wins = [0] * len(jockeys)
jockey_winrate = [0] * len(jockeys)
jockey_label = [0] * len(jockeys)
for i in range (0,len(horses)):
    races = len(df.loc[df['horse_id'] == horses[i]].values)
    wins = len(df.loc[(df['horse_id'] == horses[i]) & (df['finishing_position'] == '1')].values)
    horse_wins[i] = wins
    horse_winrate[i] = wins * 1.0 / races
    if  (horse_winrate[i] > 0.5 and wins >= 4):
        horse_label[i] = 1

for i in range (0,len(jockeys)):
    races = len(df.loc[df['jockey'] == jockeys[i]].values)
    wins = len(df.loc[(df['jockey'] == jockeys[i]) & (df['finishing_position'] == '1')].values)
    jockey_wins[i] = wins
    jockey_winrate[i] = wins * 1.0 / races
    if  (jockey_winrate[i] > 0.15 and wins >= 100):
        jockey_label[i] = 1


# plot the graph


plt.subplot(211)
plt.ylabel('win rate')
plt.title('The Best Horse and The Best Jockey')
plt.scatter(horse_wins, horse_winrate, c = 'k', alpha = 0.6)
for i in range(0,len(horses)):
    if horse_label[i]:
        plt.annotate(horses[i],(horse_wins[i], horse_winrate[i]))
plt.subplot(212)
plt.ylabel('win rate')
plt.xlabel('win numbers')
plt.scatter(jockey_wins, jockey_winrate, c = 'k', alpha = 0.6)
for i in range(0,len(jockeys)):
    if jockey_label[i]:
        plt.annotate(jockeys[i],(jockey_wins[i], jockey_winrate[i]))
plt.show()
