import numpy as np
import math
import wave
import matplotlib.pyplot as plt
import scipy.io.wavfile 
import scipy.fftpack as fft
from scipy import signal
import csv



def readWav(fileName):
    srate,data = scipy.io.wavfile.read(fileName)
    
    temp1 = []
    temp2 = []
    for i in range(len(data)):
        temp1.append(data[i][0])
        temp2.append(data[i][1])
    
    return temp1,temp2
def generateWav(filename,data,srate = 44100):
    scipy.io.wavfile.write(filename,srate,data)

def createWindows(data,window_size = 2048):
    output = []
    lower = 0
    
    while (lower + window_size < len(data)):
        output.append(data[lower:lower+window_size])
        lower+= window_size

    output.append(data[lower:])
    return output

def findPeak(data,lower,upper):
    mx = 0
    idx = 0
    for i in range(lower,upper):
        if (data[i] > mx):
            mx = data[i]
            idx = i
    
    return idx

def getMels(windows):
    mels = [] 
    for w in windows:
        complex_spectrum = np.fft.rfft(w)
        #magnitude_spectrum = np.abs(complex_spectrum)
        temps = []
        for i in range(len(complex_spectrum)):
            f = complex_spectrum[i] * 44100/2048
            m= 2595 * np.log10(1 + f/700)
            w = 1 - np.abs( (i -( len(complex_spectrum) - 1)/2 )  / (len(complex_spectrum)/2) )
            m = m*w
            temps.append(m)
        mels.append(temps)
    
    return mels

data1,data2 = readWav("TubthumpingCut.wav")

windows1 = createWindows(data1)


#mels = getMels(windows1)

#hope = np.zeros(shape = (len(mels)-1,len(mels)-1))
#print("Total Length:", len(mels)-1)
#for i in range(len(mels)-1):
#    for j in range(len(mels)-1):
#        dot = np.dot(mels[i],mels[j])
#        hope[i][j] = dot
#    print("done with row:",i)
    
#with open("TubthumpingCut2.csv","w") as f:
#    writer = csv.writer(f)
#    writer.writerows(mels)
fig1 = []
fig2 = []
with open('TubthumpingCut.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        fig1.append(row)

f.close()

#with open("TubthumpingCut2.csv","r") as f:
#    reader = csv.reader(f)
#    for row in reader:
#        fig2.append(row)
#        
#f.close()
 
print(fig1[0][0],type(fig1[0][0]))
c = np.fromstring(fig1[0][0],dtype = complex) 
       

#for i in range (len(fig1)-1):
#    for j in range(len(fig1[0])):
#        fig1[i][j] = fig1[i][j][0]


plt.figure()
plt.imshow(fig1)

#plt.figure()
#plt.imshow(fig2)

