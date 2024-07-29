import numpy as np
import matplotlib.pyplot as plt

# Construct a time signal
Fs = 2000 # sampling freq
tstep = 1 / Fs # sample time interval
f0 = 100 # signal freq

N = int(10 * Fs / f0) # number of samples

t = np.linspace(0, (N-1)*tstep, N) # time steps
fstep = Fs /N
f = np.linspace(0, (N-1)*fstep, N) #freq steps

y = 1 * np.sin(2 * np.pi  *f0 * t) + 4 * np.sin(2 * np.pi * 3 * f0 * t) + 2 #the second part of equation is for multi freqs


# Proform fft
X = np.fft.fft(y)
X_mag = np.abs(X) / N

f_plot = f[0:int(N/2+1)]
x_mag_ploy = 2 * X_mag[0:int(N/2+1)]
x_mag_ploy = x_mag_ploy[0] / 2 # Note: DC component does not need to multiply by 2

# plot
fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
ax1.plot(t,y, '.-')
ax2.plot(f, X_mag, '.-')
ax1.set_xlabel("time (s)")
ax2.set_xlabel("frequency (Hz)")
ax1.grid()

ax1.set_xlim(0, t[-1])
ax2.set_xlim(0, f_plot[-1])
plt.tight_layout()
plt.show()