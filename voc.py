'''
Code by: Eli Haynal
Supervisor: Dr. Reinhard Schulte
For: LLU Volatile Organic Compound Detector Siganl Analysis
Version: 10:50 am 6/23/2023

Modified by: Nathan Perry and Nathan Fisher
Version: 3.0.2
'''


#These statements import the libraries needed for the code to run
import numpy as np #A library with useful data storage strucutures and mathematical operations
import matplotlib.pyplot as plt #A library for generating plots
from scipy.signal import find_peaks #A function for finding peaks
import scipy.fft #A library for computing ffts
import scipy.stats #A library with statistical tools
import csv #A library for reading and writing csv files
import os #A library for loading and writing to the filesystem more easily
import pickle #A library for saving data in a python-readable format
import multiprocessing # A library for parallel processing
from tqdm import tqdm # A library for progress bars

METRIC = {
	'(us)': 1e-6,
	'(ms)': 1e-3,
	'(s)': 1
}

def mvavg(x, y, window_size):
	"""
	Calculate the moving average of an input array 'y' with a given window size.
	Parameters:
		x (array-like): The input array of x-values.
		y (array-like): The input array of y-values.
		window_size (int): The size of the moving average window.
	Returns:
		aligned_x (ndarray): The x-values aligned with the moving average.
		averages (ndarray): The calculated moving averages.
	Raises:
		ValueError: If the window size is less than 1 or greater than the length of the input array.
	"""
	if window_size < 1 or window_size > len(y):
		raise ValueError("Window size must be between 1 and the length of the input array.")
	
	# Calculate moving average
	averages = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
	
	# Align x with the moving average
	start_index = window_size - 1
	aligned_x = np.asarray(x[start_index: start_index + len(averages)])
	aligned_x -= (aligned_x[0] - x[0]) / 2
	
	return aligned_x, averages

#A function for integration by the trapezoidal rule
def trap(x,y):
	area = 0
	for i in range(0,len(x)-2):
		area += ((y[i]+y[i+1])/2) * (x[i+1]-x[i])
	return area

#An function that returns the x value such that the integral under
#the curve from 0 to x is a specified proportion of the total area
#under the curve
def area_under(x,y,proportion):
	i = 3
	target = proportion*trap(x,y)
	while trap(x[:i],y[:i]) > target:
		i += 1
	return x[i]

#A function that performs the statistical F-test
def F_test(sample1,sample2):
	F = np.var(sample1)/np.var(sample2)
	df1 = len(sample1) - 1
	df2 = len(sample2) - 1
	p = scipy.stats.f.cdf(F,df1,df2)
	return F,p

#A function that automatically performs a bank of common statistical mean tests
def tstats(sample1, sample2, evar = False):
	print('datset1 mean and std:',scipy.stats.norm.fit(sample1))
	print('datset2 mean and std:',scipy.stats.norm.fit(sample2))
	print(scipy.stats.shapiro(sample1))
	print(scipy.stats.shapiro(sample2))
	print(scipy.stats.kstest(sample1, scipy.stats.norm.cdf))
	print(scipy.stats.kstest(sample2, scipy.stats.norm.cdf))
	print('F-test',F_test(sample1,sample2))
	print(scipy.stats.ttest_ind(sample1,sample2,equal_var=evar))
	print(scipy.stats.ranksums(sample1,sample2))

class Signal():
	"""
	Class representing a signal.
	Attributes:
		units (list): List of units of the signal.
		x (ndarray): Array of x values for the signal.
		y (ndarray): Array of y values for the signal.
		name (str): Name of the signal.
		flipped (bool): True if the signal is flipped, False otherwise.
	"""

	#The function initiating each class instance from a specified .txt file
	def __init__(self, infile, name = False, flip = False, baseline_shift = 0, smooth_window=0):
		"""
		Initializes an instance of the Signal class.
		Parameters:
			infile (str): The path to the input file.
			name (str, optional): The name of the object. If not provided, the name will be extracted from the input file path.
			flip (bool, optional): Specifies whether to flip the data. Default is False.
			baseline_shift (float, optional): The amount to shift the y values of the signal. Default is 0.
			smooth_window (int, optional): The size of the window for smoothing the signal. Default is 0 (no smoothing).
		Returns:
		None
		"""

		#Flip the data only if specified
		const = 1
		if flip:
			const = -1

		#Open the specified input file and read all of its lines into a list
		with open(infile) as f:
			reader = csv.reader(f,delimiter='\t')
			data = [row for row in reader]

			#Extract the units of the signal from the header
			self.units = data[1]

			#Eliminate the three header lines
			data = data[3:]

			#Create a list of the x values for the signal
			self.x = np.asarray([float(elm[0]) for elm in data])

			#Create a list of y values for the signal, flipping each
			#if specified and moving them to zero the signal
			self.y = np.asarray([const*float(elm[1])+baseline_shift for elm in data])

		#Set a name for the object so that it can be identified
		self.name = os.path.split(infile)[1]
		if name:
			self.name = name

		# True if flipped
		self.flipped = flip

		# Smooth the signal if specified
		self.smooth(smooth_window)

	def plot(self,folder,fft = False):
		"""
		Generate and save a plot of the signal or its FFT.
		Parameters:
			folder (str): Directory to save the plot image. Created if it doesn't exist.
			fft (bool, optional): Plot FFT if True, time-domain signal if False. Default is False.
		"""
		if fft:
			plt.plot(self.xf,self.yf)
			plt.title('FFT: ' + self.name)
			plt.xlabel('Frequency (Hz)')
			plt.ylabel('Magnitude ' + self.units[1])
		else:
			if self.flipped:
				plt.ylim(-150,0)
			else:
				plt.ylim(-400,-150)
			plt.title(self.name)
			plt.xlabel('Time ' + self.units[0])
			plt.ylabel('Amplitude ' + self.units[1])
			plt.plot(self.x,self.y)	

		#Create the output folder if it does not already exist
		if not os.path.isdir(folder):
			os.mkdir(folder)

		#Generate the filename by replacing 'txt' with 'png'
		#and properly converting it into a filepath the computer will understand
		filename = self.name[0:-3] + 'png'
		path = os.path.join(folder,filename)

		#Save the figure and clear the plotting tool's buffer
		plt.savefig(path)
		plt.clf()

	#Integrate the signal with the trapezoid rule
	def integral(self):
		return(trap(self.y,self.x))

	#Find the x value such that the integral under
	#the signal from 0 to x is a specified proportion of the total area
	#under the signal
	def area_under_prop(self,prop):
		return area_under(self.x,self.y,prop)

	#Determine if the signal has multiple peaks
	#this version is unused
	def multimodal(self):
		if len(find_peaks(self.y,width = 100,distance=500)[0]) > 0:
			return True
		elif len(find_peaks([-1*y for y in self.y],width = 10,distance=500,height=100)[0]) != 1:
			return True
		else:
			return False
	
	# Checks if there exists no peak and returns true if there isn't a peak else return false
	def is_empty(self, threshold=None):
		"""
		Check if there exists no peak in the signal.
		Parameters:
			threshold (int, optional): Threshold to determine peak existence. Default is -390.
		Returns:
			bool: True if there is no peak above the threshold, False otherwise.
		"""
		if self.flipped:
			raise ValueError("Cannot check for empty signal if signal is flipped")
		
		if threshold == None:
			threshold = -390
		
		return self.y.max() < threshold

	def smooth(self, window_size = None):
		"""
		Smooth the signal using a moving average and recalculate FFT and signal statistics.
		Parameters:
			window_size (int, optional): Window size for smoothing. Default is 10.
		Returns:
			None
		"""
		if window_size == 0:
			return
		elif window_size == None:
			window_size = 10
		
		self.x, self.y = mvavg(self.x, self.y, window_size)
		self.fft()
		if self.flipped:
			if np.max(self.y) >= 0:
				raise ValueError("Cannot recalculate signal statistics, graph is above the x axis. Try changing the baseline shift")
			self.max = np.min(self.y)
			self.max_x = self.x[np.argmin(self.y)]
			self.n = []
			i = 0 
			while self.y[i] > 0.9*self.max:
				i += 1
			self.n.append(i)
			i = -1
			while self.y[i] > 0.9*self.max:
				i -= 1
			self.n.append(i)
			self.f = []
			i = 0 
			while self.y[i] > 0.5*self.max:
				i += 1
			self.f.append(i)
			i = -1
			while self.y[i] > 0.5*self.max:
				i -= 1
			self.f.append(i)
			self.t = []
			i = 0 
			while self.y[i] > 0.1*self.max:
				i += 1
			self.t.append(i)
			i = -1
			while self.y[i] > 0.1*self.max:
				i -= 1
			self.t.append(i)

			self.risetime = self.max_x-self.x[self.t[0]]
			self.falltime = self.x[self.t[1]] - self.max_x
			self.tnrise = self.x[self.n[0]]-self.x[self.t[0]]
			self.tnfall = self.x[self.t[1]]-self.x[self.n[1]]

	#Calculate the Fast-Fourier transform of the signal
	def fft(self, metric_prefix = None):
		"""
		Calculate the Fast-Fourier transform of the signal.
		Parameters:
			metric_prefix (optional): The units of the time axis of the signal,
			such as "(um)", "(ms)" or "(s)". Default is the value of self.units[0].
		Returns:
			None
		"""
		# remove dc component
		y = self.y - np.mean(self.y)
		
		# Convert signal time axis to seconds
		if metric_prefix == None:
			x = np.asarray(self.x) * METRIC[self.units[0]]
		else:
			x = np.asarray(self.x) * metric_prefix

		# Sample spacing
		T = np.mean(np.diff(x))
		n = len(x)
		
		fft_values = np.fft.fft(y)
		fft_frequncies = np.fft.fftfreq(n, d=T)
		magnitude = np.abs(fft_values) / n
		self.yf = np.asarray(magnitude[:n//2])
		self.xf = np.asarray(fft_frequncies[:n//2])

	# Plot the magnitude of the FFT results
	def show_signal(self, fft=False):
		"""
		Show a graph of the signal.
		Parameters:
			fft (bool, optional): Plot FFT if True, time-domain signal if False. Default is False.
		Returns:
			None
		"""
		# Plot the frequency vs. magnitude
		plt.plot(self.xf if fft else self.x, self.yf if fft else self.y)
		
		# Label the axes
		plt.xlabel('Frequency (Hz)' if fft else 'Time ' + self.units[0])
		plt.ylabel('Magnitude' if fft else 'Amplitude ' + self.units[1])
		
		# Add a title
		plt.title('FFT of the Signal' if fft else 'Signal')
		
		# Display the plot
		plt.show()

class Run():
	"""
	Class representing a run of signals.
	Attributes:
		name (str): Name of the run.
		signals (list): List of signal objects in the run.
		smoothed (bool): True if the signals are smoothed, False otherwise.
		smoothness (int): The size of the window for smoothing the signals.
		units (list): List of units of the signals in the run.
	"""

	def __init__(self, foldername, flip = False, cache = True, smoothness = 'default'):
		"""
		Initialize a run instance from a specified folder of .txt files.
		Parameters:
			foldername (str): Path to the folder containing .txt files.
			flip (bool, optional): If True, flip the signals. Default is False.
			cache (bool, optional): If True, save the run object to cache. Default is True.
			smoothness (int, optional): The size of the window for smoothing the signals. Default is 'default'.
		Returns:
			None
			
		"""
		
		# True if signals are smoothed
		self.smoothed = smoothness == 'default' or smoothness > 0
		self.smoothness = smoothness
		
		self.name = os.path.split(foldername)[1]

		try:
			run_cache = load()
			if cache and self.name == run_cache.name and not (not self.smoothed and run_cache.smoothed) and (run_cache.smoothness == smoothness or run_cache.smoothness == 0):
				self.signals = run_cache.signals
				self.units = run_cache.units
				if self.smoothed and not run_cache.smoothed:
					self.smooth(smoothness)
					save(self)
					print("Saved run to cache")
				print("Loaded run from cache")
				return
		except:
			pass

		
		# Get the list of files to be processed and filter out hidden files
		files = [os.path.join(foldername, filename) for filename in os.listdir(foldername) if filename[0] != '.']

		# Create a pool of worker processes
		with multiprocessing.Pool() as pool:
			# Use pool.map to parallelize the loading of signals
			results = list(tqdm(pool.imap(self.load_signal, [(f, flip) for f in files]), total=len(files), desc="Loading files"))

		# Filter out any None results (in case of errors)
		self.signals = [res for res in results if res is not None]

		# Get units from signals
		self.units = self.signals[0].units
		
		# Smooth the signals
		self.smooth(smoothness)

		# Try to save signals to cache
		try:
			if cache:
				with open("saved_run_objects.p", 'wb') as f:
					pickle.dump(self, f)
					print("Saved run to cache")
		except:
			pass

	@staticmethod
	def load_signal(args):
		f, flip = args
		try:
			return Signal(f, flip=flip)
		except ValueError:
			return None

	#A function defining how a run object is represented when printed
	#to the command line, etc. Increases readability.
	def __repr__(self):
		return(self.name)
	
	def plot(self,folder,fft = False):
		"""
		Plot every signal in the run to a specified folder.
		Parameters:
			folder (str): Directory to save the plot images.
			fft (bool, optional): Plot FFT if True, time-domain signal if False. Default is False.
		Returns:
			None
		"""
		with multiprocessing.Pool() as pool:
			# Use pool.map to parallelize the plotting of signals and us tqdm to show progress
			list(tqdm(pool.imap(self.plot_signals, [(s, folder, fft) for s in self.signals]), total=len(self.signals), desc="Plotting signals"))

	@staticmethod
	def plot_signals(args):
		s, folder, plot_fft = args
		s.plot(folder, fft=plot_fft)

	def fft(self, metric_prefix = None):
		"""
		Calculate the FFT for every signal in the run.
		Parameters:
			metric_prefix (optional): The units of the time axis of the signal,
			such as "(um)", "(ms)" or "(s)" . Default is the value of self.units[0].
		Returns:
			None
		"""
		for s in self.signals:
			s.fft(metric_prefix)

	def clean(self):
		"""
		Remove double-peaked signals from the run.
		Returns:
			None
		"""
		new = []
		for signal in self.signals:
			if not signal.multimodal():
				new.append(signal)
		self.signals = new

	def clean_empty(self, threshold=None):
		"""
		Remove signals without peaks from the run.
		Returns:
			None
		"""
		new = []
		for signal in self.signals:
			if not signal.is_empty(threshold):
				new.append(signal)
		self.signals = new
				

	#Calculates time values t for each signal in the run for which the integral
	#from 0 to t represents 90%, 50%, or 10% of the total area under the curve
	def remake_stats(self):

		#Create a list for the run object to store this data for each signal
		self.nineties = []
		self.fifties = []
		self.tens = []
		i = 0

		#Calculate each statistic for each signal and append the results to
		#the appropriate lists
		for s in self.signals:
			self.nineties.append(s.area_under_prop(0.9))
			self.fifties.append(s.area_under_prop(0.5))
			self.tens.append(s.area_under_prop(0.1))
			i += 1

			#Print a progress message
			print(f'Calculating statistics for {self.name}, {i} of {len(self.signals)} complete.',end = '\r')
		
	#Remove double-peaked signals from the run using the 'nnet_multimodal'
	#method above, this is the currently used cleaning method
	def nnet_clean(self):
		new = []
		for signal in self.signals:
			if not signal.nnet_multimodal():
				new.append(signal)
		self.signals = new

	#Calculate another panel of statistics for each signal in the run
	#area between 90% and 50% amplitude values
	#peak width at 90% and 50% amplitude
	#amplitude for each signal
	#0-100% and 10-90% rise and fall time
	def new_stats(self):

		#Create lists to store the values of each parameter for each signal
		self.narea = []
		self.farea = []
		self.nwidth = []
		self.fwidth = []
		self.amplitudes = []
		self.risetime = []
		self.falltime = []
		self.tnrise = []
		self.tnfall = []
		i = 0

		#For each signal, calculate the desired parameters using the
		#50% and 90% amplitude x-values obtained above in the "signal" class
		for s in self.signals:
			self.amplitudes.append(s.max)
			self.nwidth.append(s.x[s.n[1]]-s.x[s.n[0]])
			self.fwidth.append(s.x[s.f[1]]-s.x[s.n[0]])
			self.farea.append(trap(s.x[s.f[0]:s.f[1]],s.y[s.f[0]:s.f[1]])/trap(s.x,s.y))
			self.narea.append(trap(s.x[s.n[0]:s.n[1]],s.y[s.n[0]:s.n[1]])/trap(s.x,s.y))
			self.risetime.append(s.risetime)
			self.falltime.append(s.falltime)
			self.tnrise.append(s.tnrise)
			self.tnfall.append(s.tnfall)
			i += 1

			#Print a progress message
			print(f'Calculating statistics for {self.name}, {i} of {len(self.signals)} complete.',end = '\r')
	
	def smooth(self, smoothness = None):
		"""
		Smooth each signal in the run with a 10-point moving average.
		Parameters:
			smoothness (int, optional): The size of the window for smoothing the signals. Default is None.
		Returns:
			None
		"""
		if smoothness == 'default':
			smoothness = None
		with multiprocessing.Pool() as pool:
			# Use pool.map to parallelize the smoothing of signals and us tqdm to show progress
			self.signals = list(tqdm(pool.imap(self.smooth_signals, [(s, smoothness) for s in self.signals]), 
							total=len(self.signals), desc="Smoothing signals"))
	
	@staticmethod
	def smooth_signals(args):
		s, smoothness = args
		s.smooth(smoothness)
		return s

	def avg_signal(self, fft):
		"""
		Calculate the average signal or FFT for the run.
		Parameters:
			fft (bool): If True, calculate the average FFT. If False, calculate the average time-domain signal.
		Returns:
			tuple: Arrays of x-values and average y-values.
		"""
		# Extract the y or yf arrays
		y_arrays = [s.yf if fft else s.y for s in self.signals]
		
		# Stack the arrays along a new axis and compute the mean
		avg_y = np.mean(np.stack(y_arrays), axis=0)
		
		# Extract the corresponding x or xf array (assuming they are the same for all signals)
		x = self.signals[0].xf if fft else self.signals[0].x
		
		return x, avg_y

	def show_avg_signal(self, fft=False, ybottom=None, ytop=None, xleft=None, xright=None):
		"""
		Plot and show the average signal or FFT for the run.
		Parameters:
			fft (bool, optional): Plot FFT if True, time-domain signal if False. Default is False.
			ybottom (optional): Bottom limit for y-axis.
			ytop (optional): Top limit for y-axis.
			xleft (optional): Left limit for x-axis.
			xright (optional): Right limit for x-axis.
		Returns:
			None
		"""
		x, avg_y = self.avg_signal(fft)
		plt.plot(x, avg_y)
		plt.title('Average FFT: ' + self.name if fft else 'Average Signal: ' + self.name)
		plt.xlabel('Frequency (Hz)' if fft else 'Time ' + self.units[0])
		plt.ylabel('Magnitude' if fft else 'Amplitude ' + self.units[1])
		plt.ylim(bottom=ybottom, top=ytop)
		plt.xlim(left=xleft, right=xright)
		plt.show()
		
def plot_average_signals(A, B, filepath, fft=False, show=False):
	"""
	Plot the average signal or FFT for two runs. Useful for subjectively identifying 
	typical signal differences between two treatments.
	Parameters:
		A (run): The first run.
		B (run): The second run.
		filepath (str): The directory to save the plot image.
		fft (bool, optional): Plot FFT if True, time-domain signal if False. Default is False.
		show (bool, optional): If True, display the plot. If False, save the plot. Default is False.
	Returns:
		None
	"""
	A_x, A_y = A.avg_signal(fft)
	
	B_x, B_y = B.avg_signal(fft)

	#Clear the plotting tool
	plt.clf()

	#Plot both average signals over time
	if fft:
		plt.plot(A_x,A_y,label=A.name)
		plt.plot(B_x,B_y,label=B.name)
	else:
		plt.plot(A_x,A_y,'o',markersize=3,label=A.name)
		plt.plot(B_x,B_y,'o',markersize=3,label=B.name)

	#Add a title and axis labels to the plot
	plt.title('Average signals')
	plt.xlabel('Frequency (Hz)' if fft else 'Time ' + A.units[0])
	plt.ylabel('Magnitude' if fft else 'Amplitude ' + A.units[1])

	#Add a legend to the plot
	plt.legend()

	#Save the plot at the specified location with an auto-generated name
	#and clear the plotting tool
	if show:
		plt.show()
	else:
		plt.savefig(os.path.join(filepath,A.name+'_'+B.name+'_average_signals.png'))
	plt.clf()

#A function that plots each signal in two runs as an (x,y) datapoints
#the x and y coorinates corespond to two specified parameters
#the points are color-coded to show which run they came from
def cluster_plot(run1, run2, parameter1, parameter2, filepath, usenames=True):
	
	#Ensure that both specified parameters exist
	check1 = [elm for elm in dir(run1) if elm[0] != '_']
	check2 = [elm for elm in dir(run2) if elm[0] != '_']
	if parameter1 not in check1 or parameter1 not in check2:
		print('Unavailable parameter')
		return
	elif parameter2 not in check1 or parameter2 not in check2:
		print('Unavailable parameter')
		return

	#Load the desired parameter data from each run into local data
	else:
		dat11 = getattr(run1,parameter1)
		dat12 = getattr(run1,parameter2)
		dat21 = getattr(run2,parameter1)
		dat22 = getattr(run2,parameter2)

	#A dictionary that allows the code to generate readable figure text
	#from internal variable names
	caption_dict = {
	'nineties':'90% area time',
	'fifties':'50% area time',
	'tens':'10% area time',
	'amplitudes':'Amplitude',
	'nwidth':'Peak width at 90% amplitude',
	'fwidth':'Peak width at 50% amplitude',
	'narea':'Area between 90% amplitudes',
	'farea':'Area between 50% amplitues',
	'risetime':'Rise time (10% to peak)',
	'falltime':'Fall time (peak to 10%)',
	'tnrise':'Rise time (10% to 90%)',
	'tnfall':'Fall time (90% to 10%)'
	}

	#Load the names for the two runs
	label1 = run1.name
	label2 = run2.name
	if not usenames:
		label1 = 'Propane'
		label2 = 'Phenol'

	#Clear the plotting tool
	plt.clf()

	#Plot the each signal from the first run as a (parameter1, parameter2) datapoint
	plt.plot(dat11, dat12,'o',markersize=3,label=run1.name,alpha=0.5)
	#Repeat for the second run
	plt.plot(dat21, dat22,'o',markersize=3,label=run2.name,alpha=0.5)
	
	#Apply an auto-generated title and axis names
	plt.title('Clustering for '+caption_dict[parameter1]+' vs '+caption_dict[parameter2])
	plt.xlabel(caption_dict[parameter1])
	plt.ylabel(caption_dict[parameter2])
	#Add a legend
	plt.legend()

	#Save the figure at the specified location with an auto-generated file name
	plt.savefig(os.path.join(filepath,run1.name+'_'+run2.name+'_'+parameter1+'_v_'+parameter2+'.png'))
	plt.clf()
	return

#A function to export all of the calculated statistics from a run
#(amplitude for each signal, fall time for each signal, etc.)
#as a csv file (can be opened in excel)
#Useful for saving processed data in a familiar format
def export_csv(run, filepath):

	#A list of every parameter to be iterated over
	parameters = ['nineties','fifties','tens','amplitudes','nwidth','fwidth','narea','farea','risetime','falltime','tnrise','tnfall']
	
	#A dictionary allowing the code to generate readable text from
	#internal variable names
	caption_dict = {
	'nineties':'90% area time',
	'fifties':'50% area time',
	'tens':'10% area time',
	'amplitudes':'Amplitude',
	'nwidth':'Peak width at 90% amplitude',
	'fwidth':'Peak width at 50% amplitude',
	'narea':'Area between 90% amplitudes',
	'farea':'Area between 50% amplitues',
	'risetime':'Rise time (10% to peak)',
	'falltime':'Fall time (peak to 10%)',
	'tnrise':'Rise time (10% to 90%)',
	'tnfall':'Fall time (90% to 10%)'
	}

	#Generate a header for the csv file containing the name of each parameter
	header = [caption_dict[elm] for elm in caption_dict]

	#Open the designated output file
	with open(os.path.join(filepath,run.name+'.csv'),'w',newline='') as f:
		writer = csv.writer(f,delimiter=',')
		#Write the header to the file
		writer.writerow(header)
		#Convert the run's signals into a list of 12-element datapoints
		#(one coordinate for each parameter) using the 'run_to_poitns'
		#function (see below) and write each point as a row in the output
		#file.
		for row in run_to_points(run):
			writer.writerow(row)

#For two runs, plot a side-by-side histogram for each of the 12
#parameters using the 'histogram' function (see below)
def histograms(run1, run2, filepath):
	parameters = ['nineties','fifties','tens','amplitudes','nwidth','fwidth','narea','farea','risetime','falltime','tnrise','tnfall']
	for parameter in parameters:
		histogram(run1, run2, parameter, filepath)

#For two runs, plot a side-by-side histogram of their signals for
#a specified parameter
def histogram(run1, run2, parameter, filepath):
	
	#A dictionary allowing the code to generate readable text from
	#internal variable names
	caption_dict = {
	'nineties':'90% area time',
	'fifties':'50% area time',
	'tens':'10% area time',
	'amplitudes':'Amplitude',
	'nwidth':'Peak width at 90% amplitude',
	'fwidth':'Peak width at 50% amplitude',
	'narea':'Area between 90% amplitudes',
	'farea':'Area between 50% amplitues',
	'risetime':'Rise time (10% to peak)',
	'falltime':'Fall time (peak to 10%)',
	'tnrise':'Rise time (10% to 90%)',
	'tnfall':'Fall time (90% to 10%)'
	}

	#Clear the plotting tool
	plt.clf()
	#Plot the histogram for the first run's signals
	plt.hist(getattr(run1,parameter), bins = 30,density=True,alpha=0.5,label=run1.name)
	#Repeat for the second run
	plt.hist(getattr(run2,parameter), bins = 30,density=True,alpha=0.5,label=run2.name)
	#Generate a title for the plot and add a legend
	plt.title(caption_dict[parameter])
	plt.legend()
	#Generate a name for the file and s ave it at the specified location
	name = run1.name+'_'+run2.name+'_'+parameter+'_histogram.png'
	plt.savefig(os.path.join(filepath,name))
	plt.clf()



#A function converting a run object to a list of 12-dimensional datapoints
#with each datapoint representing a signal and each dimension
#representing one of the 12 parameters.
#Useful for feeding data to a variety of other functions here
def run_to_points(run_in):

	#A list of parameters to iterate over
	parameters = ['nineties','fifties','tens','amplitudes','nwidth','fwidth','narea','farea','risetime','falltime','tnrise','tnfall']
	
	#Instantiate the output list
	out = []
	#For each signal in the run...
	for i in range(len(run_in.nineties)):
		#Append a point to the 'out' list with 12 coordinates
		#(one coordinate per calculated parameter)  
		out.append([getattr(run_in,parameter)[i] for parameter in parameters])
	return out

#Apply the above function (run_to_points) to a list of runs
def runs_to_points(runs):
	output = []
	for run in runs:
		output = output + run_to_points(run)
	return output

#A function that saves a list (technically a python dictionary datatype) of run objects
#as the file 'saved_run_objects.p'
#This is useful because it allows for a large number of runs to be loaded
#into the code in a session of use and then saved, with their statistics,
#in a python-readable format.
#Since loading the raw data and calculating statistics takes time, this
#greatly reduces loading times if the code must be restarted but work will
#be continued on the same set of runs.
#When I was writing this code, I frequently had to restart it to make minor
#bugfixes in the code between generating figures, and reducing loading times was necessary.
def save(runs):
	with open('saved_run_objects.p','wb') as f:
		pickle.dump(runs,f)
	return

def load():
	with open('saved_run_objects.p','rb') as f:
		return pickle.load(f)