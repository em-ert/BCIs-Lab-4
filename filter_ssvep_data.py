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

#%% Part 2: Design a Filter


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
def make_bandpass_filter(low_cutoff, high_cutoff, filter_type='hann', filter_order=1000, fs=1000):
    """
    This function will use the scipy.signal.firwin function to create filter 
    coefficients and use scipy.signal.freqz to help create plots used to 
    visualize the frequency response and impulse response, saving those plots
    in the Images folder.

    Parameters
    ----------
    low_cutoff : int
        Set the low end of the frequencies to cut off when filtering
    high_cutoff : int
        Set the high end of the frequencies to cut off when filtering
    filter_type : str, optional with default of 'hann'
        Set which type of filter to use. See the scipy.signal.firwin function
        description for more options
    filter_order : non-negative int, optional with default of 1000
        Set the filter order (number of coefficients -1).
    fs : non-negative int, optional with default of 1000
        Sampling rate of the eeg data.

    Returns
    -------
    filter_coefficients : Array of float64 size C where C = number of 
        coefficients (order +1)
        Returns coefficients to be used in filtering EEG data.
    
    """
    # Returns coefficients of FIR filter (order+1 length)
    # Filter coefficients (b)
    filter_coefficients = signal.firwin(numtaps=filter_order+1, cutoff=[low_cutoff, high_cutoff], window=filter_type, pass_zero='bandpass', fs=fs)

    # Frequencies at which impulse response was computed and the frequency response as complex numbers
    # frequncy axis values corresponding to filter in freq domain, Filter magnitude response in the frequency domain
    frequencies, impulse_response = signal.freqz(filter_coefficients, a=1, fs=fs)
    
    # Top end of time range for graphing
    impulse_response_max_time = len(filter_coefficients) / fs
    # Time between samples in seconds
    sample_length = 1 / fs
    # Create the length of time used to filter EEG data for graphing
    time_array = np.arange(0, impulse_response_max_time, sample_length)
    
    # Find the filter's impulse response values using lfilter
    impulse = np.zeros(len(time_array))
    impulse[0] = 1
    filter_impulse_response = signal.lfilter(filter_coefficients, a=1, x=impulse)
    
    # Create figure
    plt.figure(1, clear=True, figsize=(9, 6))
    
    # Create subplot
    ax1 = plt.subplot(2,1,1)
    
    # Plot impulse response
    plt.suptitle(f'{low_cutoff} - {high_cutoff} Hz {filter_order} Order {filter_type} Filter')

    ax1.set_title('Impulse Response')
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("gain")
    # plot event
    ax1.plot(time_array, filter_impulse_response)
    ax1.grid()

    # Plot frequency response
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
    
    #Save subplot
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
    """
    Uses the scipy.signal.filtfilt to filter EEG data.

    Parameters
    ----------
    data : dict size D where D = number of domains
        The raw EEG data dictionary
    Array of float64 size C where C = number of coefficients (order +1)
        Coefficients to be used in filtering EEG data.
    a : int, optional with default value of 1.
        Useful only if using an iir filter, but must be included as "1" for fir
        filters

    Returns
    -------
    filtered_data : Array of float 64 size ChxS where Ch = number of channels 
    and S = number of samples
        Contains filtered EEG data

    """
    
    # Use scipy.signal.filtfilt to filter the raw EEG data which is converted
    # at this step to uV instead of V
    filtered_data = signal.filtfilt(b, a, (data['eeg'] * 1e6))
    return filtered_data


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

def get_envelope(data, filtered_data, channel_to_plot=None, ssvep_frequency=None,subject=1):
    """
    Find the amplitude of the filtered EEG data at each time point for each 
    channel
    

    Parameters
    ----------
    data : dict size D where D = number of domains
        The raw EEG data dictionary
    filtered_data : Array of float 64 size ChxS where Ch = number of channels 
    and S = number of samples
        Contains filtered EEG data
    channel_to_plot : str, optional with default value of None
        If value does not equal None, this function will plot the envelope on
        top of the filtered EEG data for the listed channel
    ssvep_frequency : int, optional with default of None
        If value does not equal None, this will add the value of the target
        frequency to the title of the plot.
    subject : int, optional with default value of 1
        The ID number of the subject

    Returns
    -------
    envelope : Array of float64, size ChxS where Ch = number of channels 
    and S = number of samples
        Contains the value of the EEG amplitude at each time point.

    """
    # Use scipy.signal.hilbert to extract the EEG's envelope. The absolute 
    # value is used to only return positive values as amplitude cannot be negative
    envelope = np.abs(signal.hilbert(filtered_data))
    
    # If channel_to_plot does not = None, plot the envelope and filtered EEG
    # data of the listed channel
    if channel_to_plot != None: 
        # Find the index value of the listed channel
        channel_index = np.where(data['channels'] == channel_to_plot)[0][0]
        sample_duration = 1 / data['fs'] # Find sampling rate using 'fs' field of data
        times = np.arange(0, len(filtered_data[channel_index]) * sample_duration, sample_duration)
        # Plot filtered EEG data of target channel with envelope
        plt.figure(1, clear=True, figsize=(9, 6))
        plt.plot(times, filtered_data[channel_index])
        plt.plot(times,envelope[channel_index])
        plt.legend(['Filtered Data','Envelope'])
        # If the target ssvep frequency is listed, add the number to the title
        if ssvep_frequency == None:
            plt.title('Unknown Frequency Isolated')
        else:
            plt.title(f'Data Filtered for {ssvep_frequency} Hz')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (uV)')
        plt.grid()
        plt.tight_layout()
        plt.show()
    
    # Save the plot
    plt.savefig(f'Images/subject_{subject}_{ssvep_frequency}Hz_filtered_data_with_envelope_at_channel_{channel_to_plot}.png')
    return envelope

#%% Part 5: Plot the Amplitudes
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
    """
    Create a plot to compare two different envelopes for a given channel.

    Parameters
    ----------
    data : dict size D where D = number of domains
        The raw EEG data dictionary
    envelope_a : Array of float64, size ChxS where Ch = number of channels 
    and S = number of samples
        Contains the value of the EEG amplitude at each time point for 
        envelope a.
    envelope_b : Array of float64, size ChxS where Ch = number of channels 
    and S = number of samples
        Contains the value of the EEG amplitude at each time point for 
        envelope b.
    channel_to_plot : str
        Selects which channel will be used to compare envelopes.
    ssvep_freq_a : int
        The target frequency for envelope a.
    ssvep_freq_b : int
        The target frequency for envelope b.
    subject : int, optional with default value of 1
        The ID number of the subject

    Returns
    -------
    None.

    """
    # Create the subplot
    fig, ax = plt.subplots(nrows=2, sharex=True, layout='constrained')
    sample_count = data['eeg'].shape[1] # Number of samples
    sample_duration = 1 / data['fs'] # Find sampling rate using 'fs' field of data
    # Create range of times samples were collected to be used later as x-axis
    times = np.arange(0, sample_count * sample_duration, sample_duration)
    # Find the start and end of each stimulus
    event_beginnings = data['event_samples'] * sample_duration
    event_ends = event_beginnings + data['event_durations'] * sample_duration
    # Extract types of stimulus from event_types
    types = data['event_types']
    duration_color = 'tab:blue' # Set color for stimulus duration lines
    
    # For plot one, plot horizontal line from event_beginnings[x] to 
    # event_ends[x] using types to set y-axis value
    ax[0].hlines(types, event_beginnings, event_ends, color=duration_color)
    # Plot dots at event_beginnings and event_ends
    ax[0].scatter(event_beginnings, types, color=duration_color)
    ax[0].scatter(event_ends, types, color=duration_color)
    ax[0].set_ylabel('Flash frequency') # Label Y axis of plot 1
    
    # For plot two, plot the envelopes for the channel in channel_to_plot 
    
    # Find the index of the target channel in data['channels']
    channel_index = np.where(data['channels'] == channel_to_plot)[0][0]
    # Plot the envelopes
    ax[1].plot(times, envelope_a[channel_index])
    ax[1].plot(times, envelope_b[channel_index])
    ax[1].set_xlabel('time (s)') # Label x-axis
    ax[1].set_ylabel('Voltage (μV)') # Label y-axis
    ax[1].legend([f'Amplitude at {ssvep_freq_a} Hz',f'Amplitude at {ssvep_freq_b} Hz']) # Create legend for plot 2 in subplot
    fig.suptitle(f'Subject {subject} Filtered Amplitudes') # name subplot
    # Save the subplot
    plt.savefig(f'Images/SSVEP_S{subject}_filtered_amplitudes_{ssvep_freq_a}Hz_{ssvep_freq_b}Hz_at_channel_{channel_to_plot}.png') # Save subplot

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
