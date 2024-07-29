from voc import *

file = signal('20240724-0001 propane and water_0001.txt', flip=True)
file.smooth()
file.plot('test')