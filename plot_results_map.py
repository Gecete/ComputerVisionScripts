
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gmplot


data = pd.read_csv('C://Users/user1/Documents/integration/traces/20170614_1120.txt', sep="\t",header=0,skiprows=0)
data = pd.DataFrame(data)
latitude = np.array(data.get([data.columns[1]]))


#Borramos registros de longitud y latitud que difieran mas de 3 veces la desviacion típica
data=data[np.abs(latitude-latitude.mean()) <= (3*latitude.std())]
longitude = np.array(data.get([data.columns[2]]))
data=data[np.abs(longitude-longitude.mean()) <= (3*longitude.std())]
#obtenemos las columnas necesarias
latitude = np.array(data.get([data.columns[1]]))
longitude = np.array(data.get([data.columns[2]]))
altitude = np.array(data.get([data.columns[3]]))
rssi = np.array(data.get([data.columns[8]]))
time = np.array(data.get([data.columns[0]]))

lati=[];
alti=[];
longi=[];
for i in range(latitude.size):
	lati.append(latitude[i][0])
	alti.append(altitude[i][0])
	longi.append(longitude[i][0])

gmap3 = gmplot.GoogleMapPlotter(lati[0],longi[0],15)

gmap3.apikey='AIzaSyAG_PdWklpHjb0VSi7e7wOq9Zo95SRTaK8'
# scatter method of map object
# scatter points on the google map
gmap3.scatter(lati,longi, '#FF0000', size=1, marker=False)
# Plot method Draw a line in
# between given coordinates
gmap3.plot(lati,longi,'cornflowerblue', edge_width=2.5)
gmap3.draw("mapa.html")
print("Done!")

#RSSI
plt.figure()


plt.subplot(2, 1, 1)
plt.plot(time/1000/60-time.min()/1000/60,rssi, 'g')
plt.title('Results')
plt.ylabel('RSSI dB')

plt.legend(['RSSI along Time'])
plt.subplot(2, 1, 2)
plt.plot(time/1000/60-time.min()/1000/60,altitude, 'r')
plt.xlabel('time (min)')
plt.ylabel('Altitude')

plt.legend(['Elevation along Time'])
plt.show()
