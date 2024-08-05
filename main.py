""" from voc import *

file = signal('20240730-0001 water_0424.txt')
for i in range(2):
    file.smooth()

file.fft()
file.plot('test', fft=True)

file.show_fft()

# Plot the signal - file.x and file.y are the x and y values of the signal
plt.plot(file.x, file.y)
plt.show()
 """
from voc import *

file = signal('1kHz.txt')

# add random noise
file.y = file.y + 0.1*np.random.randn(len(file.y))

# add 50Hz noise
file.y = file.y + 0.1*np.sin(2*np.pi*50*np.array(file.x)*1e-3)

for i in range(5):
    file.smooth()

file.plot('test')
file.fft(1e-3)
file.show_fft()

# Plot the signal - file.x and file.y are the x and y values of the signal
plt.plot(file.x, file.y)
plt.show()
