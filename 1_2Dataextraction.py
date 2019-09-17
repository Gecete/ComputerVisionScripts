
import numpy as np
import pandas as pd



data = pd.read_csv('C://Users/user1/Documents/integration/traces/20171006_1103.txt', sep="\t",header=0,skiprows=0)

#question 1
print("Total time in minutes:")
time=data.get(data.columns[0])
totalTime = (time.max()-time.min())/1000/60
print(totalTime);

#question 2 , 3

altitude = data.get(data.columns[3])
RSSI = data.get(data.columns[8])
bigger300 = RSSI[altitude > 300].dropna()
smaller260 = RSSI[altitude < 260].dropna()
print("Mean RSI when altitude >300 m:")
print(bigger300.mean())
print("Mean RSI when altitude <260 m:")
print(smaller260.mean())

#question 4
pitch =data.get(data.columns[5])
totalRows = pitch.size;
pitchPositive=pitch[pitch>0].dropna()
totalPositives=pitchPositive.size
print("Pitch greater than 0:")
print(str(int(100*totalPositives/totalRows))+" % of the time")
#question 5 & 6
print("Average difference between timestamps in seconds")
listDiff=time.diff()
listDiff.pop(0)
print(round(np.array(listDiff).mean()/1000,2))
print("Biggest difference between timestamps in seconds:");
print(listDiff.max()/1000);
