# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:23:10 2017

@author: pfierens
"""
import parser
import argparse

import numpy as np
import matplotlib.pyplot as plt
import cv2
from cmath import exp, pi

import os

# Imported from :
# https://rosettacode.org/wiki/Fast_Fourier_transform#Python:_Recursive

def fft(x):
    N = len(x)
    # print("Len(X)")
    # print(len(x))
    if N <= 1: return x
    even = fft(x[0::2])
    odd =  fft(x[1::2])
    T= [exp(-2j*pi*k/N)*odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + \
           [even[k] - T[k] for k in range(N//2)]

def filter(fft, frequencies):
    result = []
    for i in range(len(frequencies)):
        if 40 <= frequencies[i] <= 120:
            result.append(fft[i])
        else:
            result.append(0)

    return result

# cap = cv2.VideoCapture('2017-09-14 21.53.59.mp4')
# cap = cv2.VideoCapture('mimi.mp4')
cap = cv2.VideoCapture('noled.mp4')
# cap = cv2.VideoCapture('martina-arco(25seg).mp4')
# cap = cv2.VideoCapture('sentada.mp4')
# cap = cv2.VideoCapture('correr.mp4')
parser = argparse.ArgumentParser(description='Argumentos.')
parser.add_argument("--size",type=int, default=30)
parser.add_argument("--total",type=bool, default=False)
args = parser.parse_args()



if not cap.isOpened():
   print("No lo pude abrir")

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
size=args.size
total=args.total
print("Size:",size,"Total:",total)



r = np.zeros((1,length))
g = np.zeros((1,length))
b = np.zeros((1,length))

left= width//2 - size//2
right= width//2 + size//2
up = height//2 + size//2
down= height//2 - size//2


k = 0
while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        if total==False:

            r[0,k] = np.mean(frame[left:right, down:up,0])
            g[0,k] = np.mean(frame[left:right, down:up,1])
            b[0,k] = np.mean(frame[left:right, down:up,2])
        else:
            r[0, k] = np.mean(frame[0:width, 0:height, 0])
            g[0, k] = np.mean(frame[0:width, 0:height, 1])
            b[0, k] = np.mean(frame[0:width, 0:height, 2])
    else:
        break
    k = k + 1


cap.release()
cv2.destroyAllWindows()
n = int(2 ** np.floor(np.log2(length)))
f = np.linspace(-n/2,n/2-1,n)*fps/n

r = r[0,0:n]-np.mean(r[0,0:n])
g = g[0,0:n]-np.mean(g[0,0:n])
b = b[0,0:n]-np.mean(b[0,0:n])

R = np.abs(np.fft.fftshift(fft(r)))**2
G = np.abs(np.fft.fftshift(fft(g)))**2
B = np.abs(np.fft.fftshift(fft(b)))**2

R_filter=filter(R,f*60)
G_filter=filter(G,f*60)
B_filter=filter(B,f*60)


heartrate_R=abs(f[np.argmax(R_filter)])*60
heartrate_G=abs(f[np.argmax(G_filter)])*60
heartrate_B=abs(f[np.argmax(B_filter)])*60

plt.suptitle("FFT")

plt.plot(60*f,R)
plt.xlim(0,200)

plt.plot(60*f,G)
plt.xlim(0,200)
plt.xlabel("frecuencia [1/minuto]")

plt.plot(60*f,B)
plt.xlim(0,200)

plt.show()

print("Frecuencia cardíaca R: ", heartrate_R, " pulsaciones por minuto")
print("Frecuencia cardíaca G: ", heartrate_G, " pulsaciones por minuto")
print("Frecuencia cardíaca B: ", heartrate_B, " pulsaciones por minuto")



