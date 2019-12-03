'''Contains functions for Fast Fourier Transform'''
from cmath import exp, pi
import time
import numpy as np
import timeit
 

def dft1D_unopt(x):	
	N = len(x)
	n_range = np.arange(N)
	a = np.ndarray(shape=(N, N), dtype=complex)
	for n in range(N):
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

def fft1D_unopt(x):
  N = len(x)
  if N <= 1: return x
  even = fft1D_unopt(x[0::2])
  odd =  fft1D_unopt(x[1::2])
  T= [exp(-2j*pi*k/N)*odd[k] for k in range(N//2)]
  return [even[k] + T[k] for k in range(N//2)] + [even[k] - T[k] for k in range(N//2)]

def fft1D_opt(x):
	N = len(x)
	c1 = -2j*pi/N
	if N <= 4: return dft1D_2(x, c1)
	even = fft1D_opt(x[0::2])
	odd =  fft1D_opt(x[1::2])
	T= [exp(-2j*pi*k/N)*odd[k] for k in range(N//2)]
	return [even[k] + T[k] for k in range(N//2)] + [even[k] - T[k] for k in range(N//2)]

def inverse_fft1D(x):
	def inverse_fft_helper(x):
		N = len(x)
		c1 = 2j*pi/N
		if N <= 4: return dft1D_2(x, c1)
		even = inverse_fft_helper(x[0::2])
		odd =  inverse_fft_helper(x[1::2])
		T= [exp(2j*pi*k/N)*odd[k] for k in range(N//2)]
		return [even[k] + T[k] for k in range(N//2)] + [even[k] - T[k] for k in range(N//2)]
	result = inverse_fft_helper(x)
	return (np.divide(result, len(x))) 

def fft2D(x):
	first_result = np.asarray([fft1D_opt(x[row]) for row in range(x.shape[0])])
	second_result = np.asarray([fft1D_opt(first_result.T[col]) for col in range(first_result.shape[1])])
	return(second_result.T)

def dft2D(x):
	first_result = np.asarray([dft1D_unopt(x[row]) for row in range(x.shape[0])])
	second_result = np.asarray([dft1D_unopt(first_result.T[col]) for col in range(first_result.shape[1])])
	return(second_result.T)

def ifft2D(x):
	first_result = np.asarray([inverse_fft1D(x[row]) for row in range(x.shape[0])])
	second_result = np.asarray([inverse_fft1D(first_result.T[col]) for col in range(first_result.shape[1])])
	return(second_result.T)


def tests():
	x = np.random.random(1024)

	start_time_1 = time.time()
	f_1_output = fft1D_opt(x)
	f_1_time = time.time() - start_time_1

	start_time_2 = time.time()
	f_2_output = dft1D_unopt(x)
	f_2_time = time.time() - start_time_2
	print("1D FFT took {} seconds, while 1D DFT took {} seconds".format(f_1_time, f_2_time))
	print('Do output of both functions match? {}'.format(np.allclose(f_1_output, f_2_output)))

	start_time_1 = time.time()
	f_1_output = fft1D_opt(x)
	f_1_time = time.time() - start_time_1

	start_time_2 = time.time()
	result = np.fft.fft(x)
	f_2_time = time.time() - start_time_2
	print("1D FFT took {} seconds, while Numpy 1D FFT took {} seconds".format(f_1_time, f_2_time))
	print('Do output of both functions match? {}'.format(np.allclose(f_1_output, result)))

	start_time_1 = time.time()
	f_1_output = inverse_fft1D(result)
	f_1_time = time.time() - start_time_1

	start_time_2 = time.time()
	f_2_output = np.fft.ifft(result)
	f_2_time = time.time() - start_time_2
	print("1D IFFT took {} seconds, while Numpy 1D IFFT took {} seconds".format(f_1_time, f_2_time))
	print('Do output of both functions match? {}'.format(np.allclose(f_1_output, f_2_output)))

	x = np.random.rand(256, 256)
	start_time_1 = time.time()
	f_1_output = fft2D(x)
	f_1_time = time.time() - start_time_1

	start_time_2 = time.time()
	f_2_output = np.fft.fft2(x)
	f_2_time = time.time() - start_time_2
	print("2D FFT took {} seconds, while Numpy 2D FFT took {} seconds".format(f_1_time, f_2_time))
	print('Do output of both functions match? {}'.format(np.allclose(f_1_output, f_2_output)))

	x = np.random.rand(256, 256)
	start_time_2 = time.time()
	f_2_output = dft2D(x)
	f_2_time = time.time() - start_time_2
	print("2D FFT took {} seconds, while 2D DFT took {} seconds".format(f_1_time, f_2_time))
	print('Do output of both functions match? {}'.format(np.allclose(f_1_output, f_2_output)))

	result = np.fft.fft2(x)
	start_time_1 = time.time()
	f_1_output = ifft2D(result)
	f_1_time = time.time() - start_time_1

	start_time_2 = time.time()
	f_2_output = np.fft.ifft2(result)
	f_2_time = time.time() - start_time_2
	print("2D IFFT took {} seconds, while Numpy 2D IFFT took {} seconds".format(f_1_time, f_2_time))
	print('Do output of both functions match? {}'.format(np.allclose(f_1_output, f_2_output)))
	print('Do output of both functions match? {}'.format(np.allclose(f_1_output, f_2_output)))


def main():
	tests()

if __name__ == '__main__':
	main()
