#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_filter_ssvep_data.py
A file used for testing the filter_ssvep_data module.

Created on Mon Mar 11 18:41:15 2024
@author: eertle
@author: SkylerDY
"""
import import_ssvep_data, filter_ssvep_data


# %% Part 1: Load the Data

# Load the data from subject 1 into a python dictionary called data
subject = 1
data_directory = 'SsvepData'
data = import_ssvep_data.load_ssvep_data(subject, data_directory)

fs = data['fs']
eeg = data['eeg'] * 1e6 # initially in volts, scaled to microvolts to match example
channels = data['channels']
event_samples = data['event_samples']
event_durations = data['event_durations']
event_types = data['event_types']

# %% Part 2: Design a Filter
"""
Cell 2: 
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

filter_coefficients = filter_ssvep_data.make_bandpass_filter(low_cutoff=11, high_cutoff=13, filter_type='hann', filter_order=10000, fs=fs)

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
"""

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
