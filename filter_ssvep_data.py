#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_ssvep_data.py
A module containing functions used to filter ssvep data.

Created on Mon Mar 11 18:13:43 2024
@author: eertle
@author: SkylerDY
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

# %% Part 2: Design a Filter
"""
Cell 2: 
    • low_cutoff, the lower cutoff frequency (in Hz),  
    • high_cutoff, the higher cutoff frequency (in Hz),  
    • filter_type, the filter type, (this will be passed to the “window” input of scipy.signal.firwin 
    and should be “hann” by default),  
    • filter_order, the filter order, and  
    • fs, the sampling frequency (in Hz). 
It should then create a filter matching these specifications and return the 
filter coefficients in an array called filter_coefficients (of length 
filter_order+1). Your code should plot the impulse response and frequency 
response of your filter and save the plot as a .png file with this naming 
convention: the figure for a Hann filter with pass-band 10-20 Hz and order 100 
should be saved as hann_filter_10-20Hz_order100.png. The function should then r
return the filter coefficients b. In your test_ script, call this function 
twice to get 2 filters that keep the data around 12Hz and around 15Hz. Use a 
1000-order Hann filter and a 2Hz bandwidth for each. In a comment, address the 
following questions: A) How much will 12Hz oscillations be attenuated by the
15Hz filter? How much will 15Hz oscillations be attenuated by the 12Hz 
filter?  B) Experiment with higher and lower order filters. Describe how 
changing the order changes the frequency and impulse response of the filter. 
"""
def make_bandpass_filter(low_cutoff, high_cutoff, filter_type='hann', filter_order=10, fs=1000):
    # Returns coefficients of FIR filter (fs+1 length)
    # Filter coefficients (b)
    filter_coefficients = signal.firwin(numtaps=filter_order+1, cutoff=[low_cutoff, high_cutoff], window=filter_type, pass_zero='bandpass', fs=fs)

    # Frequencies at which impulse response was computed and the frequency response as complex numbers
    # frequncy axis values corresponding to filter in freq domain, Filter magnitude response in the frequency domain
    frequencies, impulse_response = signal.freqz(filter_coefficients, a=1, fs=fs)
    
    # Top end of time range for graphing
    impulse_response_max_time = len(filter_coefficients) / fs
    # Time between samples in seconds
    sample_length = 1 / fs
    time_array = np.arange(0, impulse_response_max_time, sample_length)
    
    impulse = np.zeros(len(time_array))
    impulse[0] = 1
    filter_impulse_response = signal.lfilter(filter_coefficients, a=1, x=impulse)
    
    plt.figure(1, clear=True, figsize=(9, 6))

    ax1 = plt.subplot(2,1,1)
    
    plt.suptitle(f'{low_cutoff} - {high_cutoff} Hz {filter_order} Order {filter_type} Filter')

    ax1.set_title('Impulse Response')
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("gain")
    # plot event
    ax1.plot(time_array, filter_impulse_response)
    ax1.grid()

    # plot EEG data using channels
    ax2 = plt.subplot(2,1,2)
    
    impulse_response = np.abs(impulse_response)**2
    impulse_response_dB = 10*np.log10(impulse_response/np.max(impulse_response))

    ax2.set_title('Frequency Response')
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("amplitude (dB)")
    ax2.plot(frequencies, impulse_response_dB)
    ax2.grid()
            
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.savefig(f'Images/hann_filter_{low_cutoff}-{high_cutoff}_order{filter_order}.png')
    return filter_coefficients

# %% Part 3: Filter the EEG Signals
"""
Cell 3: In your filter_ module, write a function called filter_data() that 
takes 2 inputs:  
    • data, the raw data dictionary 
    • b, the filter coefficients that you produced in the last part.  
This function should apply the filter forwards and backwards in time to each
channel in the raw data. It should return the filtered data (in uV) in a 
variable called filtered_data. In your test_ script, call filter_data() twice
to filter the data with each of your two band-pass filters (the ones designed 
to capture 12Hz and 15Hz oscillations) and store the results in separate 
arrays. 
"""

def filter_data(data,b, a=1):
    return signal.filtfilt(b, a, (data['eeg'] * 1e6))

# %% Part 4: Calculate the Envelope
"""
Cell 4: In your filter_ module, write a function called get_envelope() that 
takes the following inputs:  
    • data, the raw data dictionary,  
    • filtered_data, the filtered data (one of the outputs from the last 
    part),  
    • channel_to_plot, an optional string indicating which channel you’d like 
    to plot 
    • ssvep_frequency, the SSVEP frequency being isolated (this is for the title).  
If channel_to_plot is not None (which should be the default), the function 
should create a new figure and plot the band-pass filtered data on the given 
channel with its envelope on top. (If your computer slows or freezes when you 
try to do this, it’s ok to plot every 10th sample instead of every sample.) The 
function should return the amplitude of oscillations (on every channel at every 
time point) in an array called envelope. In your test_ script, call this 
function twice to get the 12Hz and 15Hz envelopes. In each case, choose 
electrode Oz to plot. 

Update from Teams:
    
Cell 4: In your filter_ module, write a function called get_envelope() that
takes the following inputs:
        • data, the raw data dictionary,
        • filtered_data, the filtered data (one of the outputs from the last 
          part),
        • channel_to_plot, an optional string indicating which channel you’d 
          like to plot(default is None)
        • ssvep_frequency, the SSVEP frequency being isolated (this is for the 
          title – default is None).
If channel_to_plot is not None (which should be the default), the function 
should create a new figure and plot the band-pass filtered data on the given 
channel with its envelope on top. (If your computer slows or freezes when you 
try to do this, it’s ok to plot every 10th sample instead of every sample.) If 
the ssvep_frequency input is None, your plot's title should state that the 
frequency being isolated is unknown. The function should return the amplitude 
of oscillations (on every channel at every time point) in an array called 
envelope. 
"""

def get_envelope(data, filtered_data, channel_to_plot=None, ssvep_frequency=None):
    envelope = np.abs(signal.hilbert(filtered_data))
    
    if channel_to_plot != None:        
        channel_index = np.where(data['channels'] == channel_to_plot)[0][0]
        sample_duration = 1 / data['fs'] # Find sampling rate using 'fs' field of data
        times = np.arange(0, len(filtered_data[channel_index]) * sample_duration, sample_duration)
        # Plot filtered EEG data of target channel with envelope
        plt.figure(1, clear=True, figsize=(9, 6))
        plt.plot(times, filtered_data[channel_index])
        plt.plot(times,envelope[channel_index])
        plt.legend(['Filtered Data','Envelope'])
        if ssvep_frequency == None:
            plt.title('Unknown Frequency Isolated')
        else:
            plt.title(f'Data Filtered for {ssvep_frequency} Hz')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (uV)')
        plt.grid()
        plt.tight_layout()
        plt.show()
    
    plt.savefig(f'Images/{ssvep_frequency}Hz_filtered_data_with_envelope_at_channel_{channel_to_plot}.png')
    return envelope

# %% Part 5: Plot the Amplitudes
"""
Cell 5: In your filter_ module, create a function called plot_ssvep_amplitudes
() that takes the following inputs:  
    • data, the raw data dictionary, 
    • envelope_a, the envelope of oscillations at the first SSVEP frequency,  
    • envelope_b, the envelope of oscillations at the second frequency,  
    • channel_to_plot, an optional string indicating which channel you’d like to plot,  
    • ssvep_freq_a, the SSVEP frequency being isolated in the first envelope (12 in our case),  
    • ssvep_freq_b, the SSVEP frequency being isolated in the second envelope (15 in our case), 
    • subject, the subject number.  
The last 3 inputs are for the legend and plot title. The function should plot 
two things in two subplots (in a single column): 1) the event start & end 
times, as you did in the previous lab, and (2) the envelopes of the two 
filtered signals. Be sure to link your x axes so you can zoom around and 
investigate different times in the task. In your test_ script, call this
function to produce the plot for channel Oz. In a comment in the 
test_ script, describe what you see. What do the two envelopes do when the 
stimulation frequency changes? How large and consistent are those changes? Are 
the brain signals responding to the events in the way you’d expect? Check some 
other electrodes – which electrodes respond in the same way and why? 
"""

def plot_ssvep_amplitudes(data, envelope_a, envelope_b, channel_to_plot, ssvep_freq_a, ssvep_freq_b, subject):
    1

# %% Part 6: Examine the Spectra
"""
Cell 5: In your filter_ module, create a function called plot_filtered_spectra()
that takes the following inputs: 
    • data, the raw data dictionary, 
    • filtered_data, the filtered data 
    • envelope, the envelope of oscillations at the first SSVEP frequency,  
It should produce and save a 2x3 set of subplots where each row is a channel
and each column is a stage of analysis (raw, filtered, envelope). The power 
spectra should be normalized and converted to dB as in your previous lab. 
In your test_ script, call this function. In a comment, describe how the 
spectra change at each stage and why. Changes you should address include (but 
are not limited to) the following:  
    1. Why does the overall shape of the spectrum change after filtering? 
    2. In the filtered data on Oz, why do 15Hz trials appear to have less power 
    than 12Hz trials at most frequencies?  
    3. In the envelope on Oz, why do we no longer see any peaks at 15Hz? 
"""
