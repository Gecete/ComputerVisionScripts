

# import MongoDB driver for python
from pymongo import MongoClient
import glob
import pandas as pd
from pprint import pprint


#leo todos los archivos txt dentro del directorio y guardo los Dataframe en una lista
path =r'C://Users/user1/Documents/integration/traces' # use your path
allFiles = glob.glob(path + "/*.txt")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_, sep="\t",header=0,skiprows=0)
    list_.append(df)
frame = pd.concat(list_)

client = MongoClient('localhost', 27017)

database = client.database
#extract info and create objects for each file
for data in list_:
	time=data.get(data.columns[0])
	totalTime = (time.max()-time.min())/1000/60
	RSSI = data.get(data.columns[8])
	newDocu={
		"initiallat":data.iat[0,1],
		"initialLong":data.iat[0,2],
		"maxElevation":float(data.get([data.columns[3]]).max()),
		"minElevation": float(data.get([data.columns[3]]).min()),
		"totaltime": totalTime,
		"RSSI":RSSI.mean()

	}

	database.droneModels.insert_one(newDocu)
#lista resultantes
cursor = database.droneModels.find({})
for document in cursor:
    pprint(document)
