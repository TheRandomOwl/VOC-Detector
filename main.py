from voc import *

file = signal('1kHz.txt', flip=False, baseline_shift=0)
file.smooth()
file.plot('test')
file.fft()
file.plot_fft()

# Plot the signal - file.x and file.y are the x and y values of the signal
plt.plot(file.x, file.y)
plt.show()
