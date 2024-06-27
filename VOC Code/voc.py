import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import scipy.stats
import csv
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pickle

if __name__ == '__main__':
	with open('multi_nnet.p','rb') as f:
		multi_nnet = pickle.load(f)

def trap(x,y):
	area = 0
	for i in range(0,len(x)-2):
		area += ((y[i]+y[i+1])/2) * (x[i+1]-x[i])
	return area

def area_under(x,y,proportion):
	i = 3
	target = proportion*trap(x,y)
	while trap(x[:i],y[:i]) > target:
		i += 1
	return x[i]

def F_test(sample1,sample2):
	F = np.var(sample1)/np.var(sample2)
	df1 = len(sample1) - 1
	df2 = len(sample2) - 1
	p = scipy.stats.f.cdf(F,df1,df2)
	return F,p

def tstats(sample1, sample2, evar = False):
	print(scipy.stats.shapiro(sample1))
	print(scipy.stats.shapiro(sample2))
	print('F-test',F_test(sample1,sample2))
	print(scipy.stats.ttest_ind(sample1,sample2,equal_var=evar))
	print(scipy.stats.ranksums(sample1,sample2))

class signal():

	def __init__(self, infile, name = False, flip = False):

		with open(infile) as f:
			reader = csv.reader(f,delimiter='\t')
			self.dat = [row for row in reader]
			self.dat = self.dat[3:]

		self.name = infile
		if name:
			self.name = name
		const = 1
		if flip:
			const = -1
		self.x = [float(elm[0]) for elm in self.dat]
		self.y = [const*float(elm[1]) for elm in self.dat]
		self.max = min(self.y)
		self.max_x = self.x[self.y.index(self.max)]

	def plot(self,folder):
		plt.plot(self.x,self.y, 'o',markersize = 3)
		if folder not in os.listdir():
			os.mkdir(folder)
		filename = self.name[0:-3] + 'png'
		path = os.path.join(folder,filename)
		plt.savefig(path)
		plt.clf()

	def integral(self):
		return(trap(self.y,self.x))

	def area_under_prop(self,prop):
		return area_under(self.x,self.y,prop)

	def multimodal(self):
		if len(find_peaks(self.y,width = 100,distance=500)[0]) > 0:
			return True
		elif len(find_peaks([-1*y for y in self.y],width = 10,distance=500)[0]) != 1:
			return True
		else:
			return False

	def zero(self):
		#baseline = np.mean(self.y[:10])
		baseline = max(self.y)
		self.y = [elm-400 for elm in self.y]

	def nnet_multimodal(self):
		if multi_nnet.predict([self.y])[0] == 1:
			return True
		else:
			return False

class run():

	def __init__(self, foldername, flip = False):
		self.name = foldername
		self.signals = []
		i = 0
		l = os.listdir(foldername)
		l = [filename for filename in l if filename[0] != '.']
		for filename in l:
			f = os.path.join(foldername, filename)
			try:
				self.signals.append(signal(f,name = filename,flip = flip))
			except:
				None
			i += 1
			print(f'Loading files for {self.name}, {i} of {len(l)} complete.',end = '\r')

		#self.integrals = [s.integral() for s in self.signals]
		#self.fifties = [s.area_under_prop(0.5) for s in self.signals]
		##self.nineties = [s.area_under_prop(0.9) for s in self.signals]
		#self.risetime = [s.max_x for s in self.signals]
		#self.tens = [s.area_under_prop(0.1) for s in self.signals]

		
	def plot(self,folder):
		i = 0
		for s in self.signals:
			s.plot(folder)
			i += 1
			print(f'Plotting signals for {self.name}, {i} of {len(self.signals)} complete.',end='\r')

	def clean(self):
		new = []
		for signal in self.signals:
			if not signal.multimodal():
				new.append(signal)
		self.signals = new

	def remake_stats(self):
		self.nineties = []
		self.fifties = []
		self.tens = []
		i = 0
		stats = [1,1]
		for s in self.signals:
			self.nineties.append(s.area_under_prop(0.9))
			self.fifties.append(s.area_under_prop(0.5))
			self.tens.append(s.area_under_prop(0.1))
			i += 1
			print(f'Calculating statistics for {self.name}, {i} of {len(self.signals)} complete.',end = '\r')
			stats = scipy.stats.norm.fit(self.nineties)
		print(f'90% area stats for {self.name}: Mean: {stats[0]} Std: {stats[1]}')
		
	def zero(self):
		for s in self.signals:
			s.zero()

	def nnet_clean(self):
		new = []
		for signal in self.signals:
			if not signal.nnet_multimodal():
				new.append(signal)
		self.signals = new

#s = signal('voctest.txt')
#s.plot('testfolder')
#t = signal('multimodaltest.txt')
#t.plot('testfolder')

prop = run('Propane')
phen = run('Phenol')
##print(scipy.stats.norm.fit(prop.nineties))
##print(scipy.stats.norm.fit(phen.nineties))
prop.clean() #should eliminate 3
phen.clean() #should eliminate 2
#prop.zero()
#phen.zero()
prop.remake_stats()
phen.remake_stats()
#prop.plot('Propane_graphs')
#phen.plot('Phenol_graphs')

#tstats(prop.nineties,phen.nineties)


'''
#SAMPLE PLOTTING

plt.boxplot([prop.nineties, phen.nineties])
plt.tight_layout()
plt.savefig('boxplot.png')
plt.clf()

plt.bar(['Propane','Phenol'],[np.mean(proplarge.nineties), np.mean(phenlarge.nineties)],yerr = [scipy.stats.sem(proplarge.nineties),scipy.stats.sem(phenlarge.nineties)],capsize = 5)
plt.savefig('barplot.png')
plt.clf()

plt.plot(proplarge.nineties,proplarge.tens,'o',markersize=3)
plt.plot(phenlarge.nineties,phenlarge.tens,'o',markersize=3)
plt.savefig('nineties_v_tens_nnetclean.png')
plt.clf()

plt.hist(proplarge.nineties, bins = 100)
plt.savefig('hist.png')
plt.clf()
'''

proplarge = run('PropLarge',flip=True)
proplarge.zero()
proplarge.nnet_clean()
#proplarge.clean()
proplarge.remake_stats()

phenlarge = run('PhenLarge',flip=True)
phenlarge.zero()
phenlarge.nnet_clean()
#phenlarge.clean()
phenlarge.remake_stats()

plt.plot(proplarge.nineties,proplarge.fifties,'o',markersize=3)
plt.plot(phenlarge.nineties,phenlarge.fifties,'o',markersize=3)
plt.savefig('nineties_v_fifties_nnetclean.png')
plt.clf()

