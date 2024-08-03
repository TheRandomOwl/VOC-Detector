from voc import *

sample = run("07-24-24/20240724-0001 controls/20240724-0001 propane control", flip=False)

sample.clean_empty()
sample.smooth()
sample.fft(mp=1e-6)

sample.avg_fft()
sample.show_avg_fft()
sample.plot('test', fft=False)

""" from voc import *

W = run("8-1-24/20240801-0001 water", flip=False)
P = run("8-1-24/20240801-0001 prop control", flip=False)

plot_average_signals(W, P, "test") """