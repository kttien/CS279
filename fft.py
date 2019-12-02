'''Contains functions for Fast Fourier Transform'''
from cmath import exp, pi
import time
import numpy as np
 

def dft1D(x):	
	N = len(x)
	n_range = np.arange(N)
	a = np.ndarray(shape=(N, N), dtype=complex)
	# c1 = -2j*pi/N
	for n in range(N):
		# c2 = c1 * n
		for k in range(N):
			a[k][n] = exp(-2j * pi * k * n / N)
	return np.dot(a, x)

def dft1D_2(x, c1):	
	N = len(x)
	n_range = np.arange(N)
	a = np.ndarray(shape=(N, N), dtype=complex)
	for n in range(N):
		c2 = c1 * n
		for k in range(N):
			a[k][n] = exp(c2*k)
	return np.dot(a, x)

def fft1D_1(x):
  N = len(x)
  if N <= 1: return x
  even = fft1D_1(x[0::2])
  odd =  fft1D_1(x[1::2])
  T= [exp(-2j*pi*k/N)*odd[k] for k in range(N//2)]
  return [even[k] + T[k] for k in range(N//2)] + [even[k] - T[k] for k in range(N//2)]

def fft1D_2(x):
  N = len(x)
  c1 = -2j*pi/N
  if N <= 4: return dft1D_2(x, c1)
  even = fft1D_2(x[0::2])
  odd =  fft1D_2(x[1::2])
  T= [exp(-2j*pi*k/N)*odd[k] for k in range(N//2)]
  return [even[k] + T[k] for k in range(N//2)] + [even[k] - T[k] for k in range(N//2)]



def test_times():
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

	start_time = time.time()
	np.fft.fft(x)
	end_time = time.time() - start_time
	print(end_time)

test_times()