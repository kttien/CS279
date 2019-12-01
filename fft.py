'''Contains functions for Fast Fourier Transform'''
from cmath import exp, pi
import time
import numpy as np
 

def dft1D(x):	
	x = np.asarray(x, dtype=float)
	N = x.shape[0]
	n = np.arange(N)
	k = n.reshape((N, 1))
	M = np.exp(-2j * np.pi * k * n / N)
	return np.dot(M, x)

def fft1D_1(x):
  N = len(x)
  if N <= 1: return x
  even = fft1D_1(x[0::2])
  odd =  fft1D_1(x[1::2])
  T= [exp(-2j*pi*k/N)*odd[k] for k in range(N//2)]
  return [even[k] + T[k] for k in range(N//2)] + [even[k] - T[k] for k in range(N//2)]

def fft1D_2(x):
  N = len(x)
  if N <= 32: return dft1D(x)
  even = fft1D_2(x[0::2])
  odd =  fft1D_2(x[1::2])
  T= [exp(-2j*pi*k/N)*odd[k] for k in range(N//2)]
  return [even[k] + T[k] for k in range(N//2)] + [even[k] - T[k] for k in range(N//2)]

x = np.random.random(1024)
start_time = time.time()
dft1D(x)
end_time = time.time() - start_time
print(end_time)

start_time = time.time()
fft1D_1(x)
end_time = time.time() - start_time
print(end_time)

start_time = time.time()
fft1D_2(x)
end_time = time.time() - start_time
print(end_time)