# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:23:10 2017

@author: pfierens
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from cmath import exp, pi
import os

# Imported from :
# https://rosettacode.org/wiki/Fast_Fourier_transform#Python:_Recursive

def fft(x):
    N = len(x)
    if N <= 1: return x
    even = fft(x[0::2])
    odd =  fft(x[1::2])
    T= [exp(-2j*pi*k/N)*odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + \
           [even[k] - T[k] for k in range(N//2)]

cap = cv2.VideoCapture('2017-09-14 21.53.59.mp4')

# cap = cv2.VideoCapture('martina-arco(25seg).mp4')


#if not cap.isOpened(): 
#    print("No lo pude abrir")
#    return

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

r = np.zeros((1,length))
g = np.zeros((1,length))
b = np.zeros((1,length))

k = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    # resize = cv2.resize(frame, (720, 1280), cv2.INTER_LINEAR)
    
    if ret == True:
        r[0,k] = np.mean(frame[330:360,610:640,0])
        g[0,k] = np.mean(frame[330:360,610:640,1])
        b[0,k] = np.mean(frame[330:360,610:640,2])
    else:
        break
    k = k + 1


cap.release()
cv2.destroyAllWindows()

n = 1024
f = np.linspace(-n/2,n/2-1,n)*fps/n

r = r[0,0:n]-np.mean(r[0,0:n])
g = g[0,0:n]-np.mean(g[0,0:n])
b = b[0,0:n]-np.mean(b[0,0:n])

R = np.abs(np.fft.fftshift(fft(r)))**2
G = np.abs(np.fft.fftshift(fft(g)))**2
B = np.abs(np.fft.fftshift(fft(b)))**2

plt.plot(60*f,R)
plt.xlim(0,200)

plt.plot(60*f,G)
plt.xlim(0,200)
plt.xlabel("frecuencia [1/minuto]")

plt.plot(60*f,B)
plt.xlim(0,200)

plt.show()

print("Frecuencia cardíaca: ", abs(f[np.argmax(G)])*60, " pulsaciones por minuto")