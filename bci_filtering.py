import numpy as np
from matplotlib import pyplot as plt
import bci_filtering_plots as bfp
from scipy import signal

lfilter = False
fir = False

fs = 100 # Sampling frequency in Hz
T = 2 # Duration of the signal in seconds
t = np.arange(0, T, 1/fs)

# Create signal
f = 12 # alpha frequency
s_t = np.sin(2*np.pi*f*t)
s_t[int(len(t)/2) : ] = 0

x_t = s_t
x_t[int(len(t)*0.75)] = 10

# Take ft of signal
X_f = np.fft.rfft(x_t)
f = np.fft.rfftfreq(len(x_t), 1/fs)

if fir:
    # Declare our filter
    order = 5
    filter_type = 'hann'
    pass_type = 'lowpass'
    fc = 15
    # NOTE: Width = bandwidth, pass_zero? = 'bandpass'
    b = signal.firwin(numtaps=order+1, cutoff=fc, window=filter_type, pass_zero=pass_type, fs=fs)
    a=1
else:
    # Declare our filter
    order = 5
    filter_type = 'butter'
    pass_type = 'lowpass'
    fc = 15
    # NOTE: Width = bandwidth, pass_zero? = 'bandpass'
    # b = signal.firwin(numtaps=order+1, cutoff=fc, window=filter_type, pass_zero=pass_type, fs=fs)
    b, a = signal.iirfilter(N=order, Wn=fc, btype=pass_type, ftype=filter_type, fs=fs)    


impulse = np.zeros(len(t))
impulse[int(len(t)/2)] = 1
if lfilter:
    h_t = signal.lfilter(b, a, x=impulse)
else:    
    h_t = signal.filtfilt(b, a, x=impulse)

f_filter, H_f = signal.freqz(b, a, fs=fs)

# Get filter signal
if lfilter:
    y_t = signal.lfilter(b, a, x=x_t)
else:
    y_t = signal.filtfilt(b, a, x=x_t)
Y_f = np.fft.rfft(y_t)

bfp.plot_filtering(x_t,X_f,h_t,H_f,y_t,Y_f,t,f,f_filter,title='',is_plot_db = False)