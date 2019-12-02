'''Contains functions for Fast Fourier Transform'''
from cmath import exp, pi
import time
import numpy as np
 

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
  if N <= 4: return x
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
		even = fft1D_opt(x[0::2])
		odd =  fft1D_opt(x[1::2])
		T= [exp(2j*pi*k/N)*odd[k] for k in range(N//2)]
		return [even[k] + T[k] for k in range(N//2)] + [even[k] - T[k] for k in range(N//2)]
	result = inverse_fft_helper(x)
	return (np.divide(result, len(x))) 

def fft2D(x):
	first_result = np.asarray([fft1D_opt(x[row]) for row in range(x.shape[0])])
	second_result = np.asarray([fft1D_opt(first_result.T[col]) for col in range(first_result.shape[1])])
	return(second_result.T)


def test(func_1, func_2, dim, inverse = False):
	if dim == 1:
		x = np.random.random(1024)
	else:
		x = np.random.rand(256, 256)
	if inverse:
		x = np.fft.fft(x)

	start_time_1 = time.time()
	f_1_output = func_1(x)
	f_1_time = time.time() - start_time_1

	start_time_2 = time.time()
	f_2_output = func_1(x)
	f_2_time = time.time() - start_time_2

	print('{} took {} seconds to run'.format(func_1.__name__, f_1_time))
	print('{} took {} seconds to run'.format(func_2.__name__, f_2_time))
	print('Do output of both functions match? {}'.format(np.allclose(f_1_output, f_2_output)))
	print()


def run_tests():
	test(dft1D_unopt, fft1D_unopt, 1)					# test our Fast Fourier Transform against 1D DFT
	test(dft1D_unopt, fft1D_opt, 1)						# test our optimized Fast Fourier Transform against 1D DFT
	test(fft1D_opt, np.fft.fft, 1)						# test our 1D FFT against the numpy 1D FFT
	test(inverse_fft1D, np.fft.ifft, 1, inverse = True)	# test our inverse 1D FFT against the numpy 1D IFFT
	test(fft2D, np.fft.fft2, 2)							# test our 2D FFT against the numpy 2D FFT


def main():
	run_tests()
	


if __name__ == '__main__':
	main()
