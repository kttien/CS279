'''Contains functions for Fast Fourier Transform'''
from cmath import exp, pi
import time
import numpy as np
import timeit
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
import cv2
import scipy.misc
 

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


def testImage(imgName):
	img = plt.imread(imgName).astype(np.float64)

	fimg = fft2D(img)
	fimg = np.fft.fftshift(fimg)
	rows, cols = img.shape
	crow, ccol = int(rows / 2), int(cols / 2)  # center

	# Circular HPF mask, center circle is 0, remaining all ones
	mask = np.ones((rows, cols), np.uint8)
	r = 80
	center = [crow, ccol]
	x, y = np.ogrid[:rows, :cols]
	mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
	mask[mask_area] = 0

	# apply mask and inverse DFT
	fimg = fimg * mask
	f_ishift = np.fft.ifftshift(fimg)
	img_back = np.abs(ifft2D(f_ishift))

	print(np.allclose(img_back, img))
	plt.imshow(np.abs(fimg)) 
	plt.savefig("filtered")
	plt.imshow(img_back)
	plt.savefig("testing3")
	plt.imshow(img) 
	plt.savefig("original")
	plt.imshow(img_back)
	plt.show


def main():
	# tests()
	testImage("A.tif")

if __name__ == '__main__':
	main()
