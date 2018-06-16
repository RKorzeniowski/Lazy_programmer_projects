import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

from scipy.io.wavfile import write

spf = wave.open('helloworld.wav', 'r')

signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
print("numpy signal shape", signal.shape)

plt.plot(signal)
plt.title("Hello world without echo")
plt.show()

delta = np.array([1.0, 0., 0.])
noecho = np.convolve(signal, delta)
print("noecho signal:", noecho.shape)
assert(np.abs(noecho[:len(signal)] - signal).sum() < 0.0000001)


noecho = noecho.astype(np.int16)  # recast as 16 else all you can hear is noise
write('noecho.wav', 16000, noecho)

filt = np.zeros(16000)  # we have 16k sampling so 16k samples is a 1 sec od audio
filt[0] = 1
filt[4000] = 0.4
filt[8000] = 0.2
filt[15999] = 0.1
out = np.convolve(signal, filt)

out = out.astype(np.int16)
write('out.wav', 16000, out)
