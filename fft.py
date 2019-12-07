'''Contains functions for Fast Fourier Transform'''
from cmath import exp, pi
import time
import numpy as np
import timeit
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import cv2
import scipy.misc
import h5py
import csv
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
 

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
	T= [exp(c1*k)*odd[k] for k in range(N//2)]
	return [even[k] + T[k] for k in range(N//2)] + [even[k] - T[k] for k in range(N//2)]

def inverse_fft1D(x):
	def inverse_fft_helper(x):
		N = len(x)
		c1 = 2j*pi/N
		if N <= 4: return dft1D_2(x, c1)
		even = inverse_fft_helper(x[0::2])
		odd =  inverse_fft_helper(x[1::2])
		T= [exp(c1*k)*odd[k] for k in range(N//2)]
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

'''
Code to test our algorithm's efficiency.
'''
def test_complexity():
	dft_res = []
	fft_res = []
	sizes = [64, 128, 256, 512, 1024]
	for size in sizes:
		x = np.random.rand(size, size)
		start_dft = time.time()
		dft2D(x)
		end_dft = time.time() - start_dft
		dft_res.append((size, end_dft))

		start_fft = time.time()
		fft2D(x)
		end_fft = time.time() - start_fft
		fft_res.append((size, end_fft))
		print("Finished image of size: {}".format(size))

	d = plt.plot(*zip(*dft_res), marker='o', color='b', label="2D DFT")
	f = plt.plot(*zip(*fft_res), marker='v', color='r', label='2D FFT')
	plt.title("Algorithmic Complexity")
	plt.xlabel(' N (Size of N x N Image)', fontsize=18)
	plt.ylabel('Time (seconds)', fontsize=16)
	plt.legend(loc='upper left')
	plt.savefig("cell_plots/complexity_comparison")
	plt.show()


'''
Run various tests on our algorithm's accuracy and efficiency
'''
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



'''
Returns circular mask for a high pass filter of FT data. Takes in the image
shape and desired radius of mask and returns mask of 0 pixel intensity. Adapted from 
https://akshaysin.github.io/fourier_transform.html#.XesPjC2ZPPB.
'''
def hpf_circular_mask(s, r):
	rows, cols = s
	crow, ccol = int(rows / 2), int(cols / 2)  # get center of image

	# Circular HPF mask, center circle is 0, remaining all ones
	mask = np.ones((rows, cols), np.uint8)
	center = [crow, ccol]
	x, y = np.ogrid[:rows, :cols]
	mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
	mask[mask_area] = 0
	return mask

'''
Plots figure of fourier transform with filter applied along with reconstructed image
and saves to appropriate folder.
'''
def plot_comparisons(img, fimg, img_back, filter_type, param, show=False):
	fig, ax = plt.subplots(nrows=1, ncols=2)
	
	ax[0].set_title("Fourier Transform (log)")
	ax0 = ax[0].imshow(np.log(np.abs(fimg)))
	divider0 = make_axes_locatable(ax[0])
	cax0 = divider0.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(ax0, cax=cax0)

	img_back[np.log(img_back) < 0] = 0
	ax[1].set_title("Filtered (log)")
	ax1 = ax[1].imshow(np.log(img_back))
	divider1 = make_axes_locatable(ax[1])
	cax1 = divider1.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(ax1, cax=cax1)

	fig.tight_layout()
	if filter_type == "hp":
		paramlabel = "r={}".format(str(param))
	else:
		paramlabel = "%={}".format(str(param))
	plt.title("{} {}".format(filter_type, paramlabel))
	output_filename = "cell_plots/keratocyte_{}_{}".format(filter_type, str(param))
	print(output_filename)
	plt.savefig(output_filename)
	if show: plt.show()


'''
Filters the image passed in by file name using filter specified by filter_type
with parameter specified by param. Supports "hp" for high pass circular mask
and "thresh" for high pass filter based on percent maximum log signal.
'''
def filterImage(imgName, filter_type, param):
	# read image
	img = plt.imread(imgName).astype(np.float64)

	# run fft, shift frequencies, apply filter
	fimg = fft2D(img)
	fimg = np.fft.fftshift(fimg)
	if filter_type == "hp":
		# For filtering based on circular mask
		mask = hpf_circular_mask(img.shape, r = param)
		fimg = fimg * mask
	elif filter_type == "thresh":
		# For filtering based on % of maximum value in frequency domain:
		max_val = np.amax(np.log(np.abs(fimg)))
		high_threshold = param*max_val/100.
		fimg[np.log(np.abs(fimg)) > high_threshold] = 0

	f_ishift = np.fft.ifftshift(fimg)
	img_back = np.abs(ifft2D(f_ishift))
	plot_comparisons(img, fimg, img_back, filter_type, param)


'''
Helper function to save original picture
'''
def show_original(filename):
	img = plt.imread(filename).astype(np.float64)
	plt.imshow(np.log(np.abs(img)))
	output_filename = "cell_plots/keratocyte_original"
	plt.savefig(output_filename)

'''
Test various parameter values for circular mask and percent maximum
high pass filters.
'''
def test_filter_paramters(filename):
	show_original(filename)
	for param in [1, 5, 10, 20, 40, 60, 70]:
		filterImage(filename, "hp", param)
	for param in [70, 75, 80, 85, 90, 95]:	# Will divide by 100 in filterImage
		filterImage(filename, "thresh", param)


'''
Plots figure of fourier transform with filter applied along with reconstructed image
and saves to appropriate folder.
'''
def plot_comparisons_mri(img, fimg, img_back, filter_type, param, show=False):
	fig, ax = plt.subplots(nrows=1, ncols=2)
	
	ax[0].set_title("Fourier Transform (log)")
	ax0 = ax[0].imshow(fimg)
	divider0 = make_axes_locatable(ax[0])
	cax0 = divider0.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(ax0, cax=cax0)

	ax[1].set_title("Filtered (log)")
	ax1 = ax[1].imshow(img_back)
	divider1 = make_axes_locatable(ax[1])
	cax1 = divider1.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(ax1, cax=cax1)

	fig.tight_layout()
	if filter_type == "hp":
		paramlabel = "r={}".format(str(param))
	else:
		paramlabel = "%={}".format(str(param))
	plt.title("{} {}".format(filter_type, paramlabel))
	output_filename = "cell_plots/mri_{}_{}".format(filter_type, str(param))
	print(output_filename)
	plt.savefig(output_filename)
	if show: plt.show()

'''
Code to take in the single file fully sampled kspace image from:
https://www.cis.rit.edu/htbooks/mri/chap-10/detail/det-h-1.html
Code written to first read in data from unique text file specific
to this raw image data.
'''
def fullly_sampled_k_space(filename):
	c_vals = []
	with open(filename, newline = '') as txt:                                                                                          
		txt_reader = csv.reader(txt, delimiter='\t')
		for line in txt_reader:
			vals = line[0].split()
			val = complex(float(vals[0]), float(vals[1]))
			c_vals.append(val)


	c_vals = np.reshape(c_vals, (256, 256)).T
	plt.imshow(np.log(np.abs(c_vals)))
	plt.savefig("cell_plots/full_k_space")

	c_shifted = np.fft.fftshift(c_vals)
	c_f = fft2D(c_shifted)
	plt.imshow(np.abs(np.fft.fftshift(c_f)))
	original = np.abs(np.fft.fftshift(c_f))
	plt.savefig("cell_plots/fully_transformed_MRI_image")

	for size in [110, 120, 125, 130, 135]:
		c_vals2 = c_vals
		max_val = np.amax(np.log(np.abs(c_vals2)))
		high_threshold = (max_val/100.)*size
		c_vals2[np.log(np.abs(c_vals2)) > high_threshold] = 0
		c_shifted = np.fft.fftshift(c_vals2)
		c_f = fft2D(c_shifted)

		plt.imshow(np.abs(np.fft.fftshift(c_f)))
		plt.savefig("cell_plots/filtered_fully_transformed_MRI_image")
		plt.colorbar()
		filt = np.abs(np.fft.fftshift(c_f))

		plt.imshow(np.log(np.abs(c_vals2)))
		plt.savefig("cell_plots/filtered_fully_transformed_MRI_image")
		plt.colorbar()
		kspace = np.log(np.abs(c_vals2))
		plot_comparisons_mri(original, kspace, filt, "thresh", size)




'''
Helper function to plot mri reconstructed images from NYU Dataset located here:
https://fastmri.med.nyu.edu. Transforms the data across the slices listed in the parameters.
Code adapted from data tutorial listed on website for how to access the raw k-space image.
'''
def show_slices(data, slice_nums, cmap=None):
	fig = plt.figure
	for i, num in enumerate(slice_nums):
		transformed = np.fft.fftshift(fft2D(np.fft.fftshift(data[num])))
		plt.subplot(1, len(slice_nums), i + 1)
		plt.imshow(np.abs(transformed), cmap=cmap)
	plt.show()

'''
Function to take in data from https://fastmri.med.nyu.edu. These contain
partial k-spaces. Some code adapted from tutorial on how to access data in HDF5
format. Later half of code adated from Sigpy tutorial on:
https://github.com/mikgroup/sigpy-mri-tutorial/blob/master/
02-parallel-imaging-compressed-sensing-reconstruction.ipynb.
'''
def nyu_dataset(filename):
	hf = h5py.File(filename)
	print('Keys:', list(hf.keys()))
	print('Attrs:', dict(hf.attrs))
	volume_kspace = hf['kspace']
	print(volume_kspace[20].shape)	# Using slice number 20 of the volume of specific MRI scans
	slice_kspace = volume_kspace
	show_slices((slice_kspace), [20], cmap='gray')
	plt.imshow(np.log(np.abs(slice_kspace[20])))
	plt.show()

	# Data Wrangling to add dimension of 1 for number of coils as needed for following methods
	test = volume_kspace[20]
	test2 = test[np.newaxis, :, :]

	# Generating sensitivity maps using ESPIRiT method
	mps = mr.app.EspiritCalib(test2).run()
	pl.ImagePlot(mps, title='Sensitivity Maps Estimated by ESPIRiT')

	# Running SENSE reconstruction algorithm on partial k-space image
	lamda = 0.01
	img_sense = mr.app.SenseRecon(test2, mps, lamda=lamda).run()
	pl.ImagePlot(img_sense, title='SENSE Reconstruction')

	# Running L1 Wavelet Regularized Reconstruction on partial k-space, images inconclusive 
	lamda = 0.005
	img_l1wav = mr.app.L1WaveletRecon(test2, mps, lamda).run()
	pl.ImagePlot(img_l1wav, title='L1 Wavelet Regularized Reconstruction')


'''
Un-Comment to run program.
'''
def main():
	# test_complexity()
	# test_filter_paramters("input_images/A.tif")
	fullly_sampled_k_space("input_images/mrimage1d.txt")
	# nyu_dataset('input_images/file1000022_v2.h5')

if __name__ == '__main__':
	main()
