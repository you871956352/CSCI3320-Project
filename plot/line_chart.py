# CHEN Jiamin
import matplotlib.pyplot as plt
import pandas as pd

# Get horse_id from input
horse_id = raw_input("Enter the horse id: ")

# Read data from preprocessed csv
df1 = pd.read_csv('../training.csv')
df2 = pd.read_csv('../testing.csv')
df = pd.concat([df1,df2])

# Search for the specific record
record = df.loc[df['horse_id'] == horse_id]['recent_6_runs'].values[-1]
positions = record.split("/")
races = ['0'] * len(positions)
for i in range(0,len(positions)):
    races[i] = df.loc[df['horse_id'] == horse_id]['race_id'].values[i-len(positions)]
nlist = range(1,len(positions)+1)

# plot the graph
plt.ylabel('position')
plt.xlabel('game id')
plt.xlim(0, 7)  
plt.xticks(nlist, races)
plt.title('Posiyion of Horse %s in Recent 6 Runs'%(horse_id))
plt.plot(nlist, positions)
plt.show()
