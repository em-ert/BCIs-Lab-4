"""
Name:        import_ssvep_data.py
Authors:     Emily Ertle, Nick Hanna
Description: loads, plots, and epochs raw data, then performs frequency analysis on data with plots that
             match instructor description
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp

def load_ssvep_data(subject: int, data_directory: str = 'SsvepData'):
    """
    Loads and returns SSVEP data for a given subject.

    Parameters
    ----------
    subject : int
        The identification number of the subject whose data should be loaded.
    data_directory : str, optional
        The root directory where SSVEP data is located.. The default is 'SsvepData'.

    Returns
    -------
    data_dict : lib.npyio.NpzFile (dictionary/struct, size: 6)
        The dictionary of information about the SSVEP dataset, as described in the dataset's README file.

    """
    # # extract variables from dictionary
    # eeg = data_dict['eeg']
    # channels = data_dict['channels']
    # fs = data_dict['fs']
    # Validate correct subject
    if subject < 1 or subject > 2:
        print("Please select either subject 1 or 2")
        return 

    # Load dictionary from f'{data_directory}/SSVEP_S{subject}.npz'
    # teams chat
    data_dict = np.load(f'{data_directory}/SSVEP_S{subject}.npz', allow_pickle=True)
    
    return data_dict

def plot_raw_data(data, subject: int = 1, channels_to_plot: list = ['Fz', 'Oz']):
    """
    Takes a dictionary of a subject's data and plots the subject's raw EEG data 
    (voltages) over time as well as flash frequency for each time.

    Parameters
    ----------
    data : lib.npyio.NpzFile (dictionary/struct, size: 6)
        The dictionary of information about the SSVEP dataset, as described in the dataset's README file.
    subject : int, optional
        The identification number of the subject whose data is found in the `data` parameter. The default is 1.
    channels_to_plot : list (type: string), optional
        A list of channels to plot. The default is ['Fz', 'Oz'].

    Returns
    -------
    None.

    """
    # frequency
    fs              = data['fs']
    eeg             = data['eeg'] * 1e6 # initially in volts, scaled to microvolts to match example
    channels        = data['channels']
    event_samples   = data['event_samples']
    event_durations = data['event_durations']
    event_types     = data['event_types']
    
    # time, total samples divided by the sample rate
    T     = len(eeg[0,:]) / fs

    # steps within time interval for plotting
    steps = 1/fs
    x     = np.arange(0,T,steps)
    
    plt.figure(1, clear=True, figsize=(9, 6))
    ax1 = plt.subplot(2,1,1)
    plt.title(f'SSVEP subject {subject} Raw Data')
    plt.ylabel("Flash Frequency")

    # plot events
    for es_index, event_sample in enumerate(event_samples):     
        start_time = event_sample / fs
        end_time = (event_sample + event_durations[es_index]) / fs
        plt.plot([start_time, end_time], [event_types[es_index], event_types[es_index]])
    plt.ylabel("Flash Frequency")

    # plot EEG data using channels
    plt.subplot(2,1,2, sharex=ax1)
    plt.title(f'SSVEP subject {subject} Raw Data')
    plt.ylabel("Voltage (V)")
    plt.xlabel("time (s)")
    for ee_channel, channel in enumerate(channels_to_plot):
        if channel in channels:
            # finds the index within the channel array
            channel_index = list(channels).index(channel) 
            # alternatively
            # idx = np.where(channel == channel)[0][0]
            
            # plot the channel-indexed eeg data over the time of the overall acquisition period
            plt.plot(x, eeg[channel_index, :], label=channel)
            
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.savefig(f'Images/S{subject}_SSVEP_Raw_Data.png')


def epoch_ssvep_data(data_dict, epoch_start_time=0, epoch_end_time=20):    
    """
    Epochs the data into user-defined time windows and returns the epochs as
    well as additional arrays--one containing time data for samples in each
    epoch and the other information about light flashing frequency for each
    event/epoch.

    Parameters
    ----------
    data_dict : lib.npyio.NpzFile (dictionary/struct, size: 6)
        The dictionary of information about the SSVEP dataset, as described in the dataset's README file.
    epoch_start_time : int, optional
        Desired start time of epochs relative to events, in seconds. The default is 0.
    epoch_end_time : int, optional
        Desired end time of epochs relative to events, in seconds. The default is 20.

    Returns
    -------
    eeg_epochs : numpy array (type: float, size: number of events x number of channels x samples per epoch)
        EEG epochs, or sample blocks, surrounding each event recorded.
    epoch_times : numpy array (type: float, size: samples per epoch)
        The time in seconds (relative to the event) of each time point in eeg_epochs.
    is_trial_15Hz : numpy array (type: bool, size: number of events)
        An array in which is_trial_15Hz[i] is True if the light was flashing at 15Hz during trial/epoch i.

    """
    # Extract the relevant variables from the dictionary
    fs              = data_dict['fs']
    eeg             = data_dict['eeg'] * 1e6 # initially in volts, scaled to microvolts to match example
    event_samples   = data_dict['event_samples']
    event_types     = data_dict['event_types']

    # Use eeg_time in combination with user input to determine temporal details
    # of each epoch
    samples_per_second = fs
    sample_rate = 1/fs
    seconds_per_epoch = epoch_end_time - epoch_start_time
    samples_per_epoch = int(samples_per_second * seconds_per_epoch)
    epoch_count = np.size(event_samples)
    channel_count = np.size(eeg, axis=0)
    # Shaping the 3D eeg_epochs array
    eeg_epochs = np.zeros((epoch_count, channel_count, samples_per_epoch))
    # Shaping the is_trial_15hz array
    is_trial_15Hz = np.zeros(epoch_count, dtype='bool')
    
    # Loop through all the event samples and fill each epoch with the 
    # corresponding input from eeg_data
    for (sample_index,), sample in np.ndenumerate(event_samples):
        start_index = sample + int(epoch_start_time * samples_per_second)
        end_index = start_index + samples_per_epoch
        eeg_epochs[sample_index, :, :] = eeg[:, start_index:end_index]
        is_trial_15Hz[sample_index] = (event_types[sample_index] == '15hz')
    # The user-defined start time is incremented by the sample rate until the
    # user-defined end time
    epoch_times = np.arange(epoch_start_time, epoch_end_time, sample_rate)
    
    return eeg_epochs, epoch_times, is_trial_15Hz



def get_frequency_spectrum(eeg_epochs: np.ndarray, fs: float):
    """
    Calculates the Fast Fourier Transform on each channel in each epoch. Also
    returns an array containing frequency information for the spectra.

    Parameters
    ----------
    eeg_epochs : numpy array (type: float, size: number of events x number of channels x samples per epoch)
        EEG epochs, or sample blocks, surrounding each event recorded.
    fs : float
        The sampling frequency.

    Returns
    -------
    eeg_epochs_fft : numpy array (type: float, size: number of events x number of channels x number of frequencies)
        Fast Fourier Transforms of epoch data; frequency content of epoched signals.
    fft_frequencies : numpy array (type: float, size: number of frequencies)
        The frequency corresponding to each column in the FFT; eeg_epochs_fft[:,:,i] is the energy at frequency fft_frequencies[i] Hz.

    """
    epoch_count, channel_count, samples_per_epoch = eeg_epochs.shape
    
    # Get frequencies in spectrum
    fft_frequencies = np.fft.rfftfreq(samples_per_epoch, 1/fs)
    
    eeg_epochs_fft = np.zeros((epoch_count, channel_count, len(fft_frequencies)), dtype=np.complex64)
    
    # Iterate through all epochs and take their FFTs
    for epoch_idx in range(epoch_count):
        for channel_idx in range(channel_count):
            eeg_epochs_fft[epoch_idx, channel_idx, :] = np.fft.rfft(eeg_epochs[epoch_idx, channel_idx, :])
    
    print(np.shape(eeg_epochs_fft))

    # returns the frequency spectrums for all epochs and the freq bins
    return eeg_epochs_fft, fft_frequencies


def plot_power_spectrum(eeg_epochs_fft: np.ndarray, fft_frequencies: np.ndarray, is_trial_15Hz: np.ndarray, channels: list, channels_to_plot: list, subject: int):
    """
    Plots the mean frequency spectrum across trials for a given subject. Plots
    can be generated for any number of channels in the data (the max is 32).
    Spectra are plotted separately for 12Hz and 15Hz trials.

    Parameters
    ----------
    eeg_epochs_fft : numpy array (type: float, size: number of events x number of channels x number of frequencies)
        Fast Fourier Transforms of epoch data; frequency content of epoched signals.
    fft_frequencies : numpy array (type: float, size: number of frequencies)
        The frequency corresponding to each column in the FFT; eeg_epochs_fft[:,:,i] is the energy at frequency fft_frequencies[i] Hz.
    is_trial_15Hz : numpy array (type: bool, size: number of events)
        An array in which is_trial_15Hz[i] is True if the light was flashing at 15Hz during trial/epoch i.
    channels : list (type: string, size: number of channels)
        The list of the names of each channel found in the original dataset
    channels_to_plot : list (type: string, size: number of channels to be plotted)
        A list of channels to plot.
    subject : int
        The identification number of the subject whose data should be loaded.

    Returns
    -------
    spectrum_db_12Hz : numpy array (type: float, size: number of channels x number of frequencies)
        The mean power spectrum of 12Hz trials in dB.
    spectrum_db_15Hz : numpy array (type: float, size: number of channels x number of frequencies)
        The mean power spectrum of 15Hz trials in dB.

    """
    # Calculate the absolute value of the spectrum for each channel and trial, separately for 12Hz and 15Hz trials (absolute value is there to remove phase information encoded in the complex numbers).
    eeg_epochs_spectra_12Hz = np.abs(eeg_epochs_fft[~is_trial_15Hz])
    eeg_epochs_spectra_15Hz = np.abs(eeg_epochs_fft[is_trial_15Hz])
    
    # Convert the spectra to units of power by multiplying each value by its complex conjugate (or, because we’re working with real numbers now, just squaring the spectra)
    eeg_epochs_spectra_12Hz = eeg_epochs_spectra_12Hz ** 2
    eeg_epochs_spectra_15Hz = eeg_epochs_spectra_15Hz ** 2
    
    # Take the mean across trials.
    spectrum_12Hz = np.mean(eeg_epochs_spectra_12Hz, axis=0)
    spectrum_15Hz = np.mean(eeg_epochs_spectra_15Hz, axis=0)

    # Normalize the power spectra by dividing by the max value
    # (Makes it so that peak is at 0dB)
    channel_maxima_12Hz = np.max(spectrum_12Hz, axis=1)
    channel_maxima_15Hz = np.max(spectrum_15Hz, axis=1)
    spectrum_12Hz = spectrum_12Hz / channel_maxima_12Hz[:, None]
    spectrum_15Hz = spectrum_15Hz / channel_maxima_15Hz[:, None]
    
    # Convert to decibel units (power_in_db = 10 * log10(power))
    spectrum_db_12Hz = 10 * np.log10(spectrum_12Hz)
    spectrum_db_15Hz = 10 * np.log10(spectrum_15Hz)
    
    channel_count = np.size(channels_to_plot)
    
    # Set number of max rows and axis dimensions
    MAX_ROW_COUNT = 8
    AXIS_LENGTH = 8
    AXIS_HEIGHT = 3
    
    if channel_count <= MAX_ROW_COUNT:
        row_count = channel_count
        column_count = 1
    else:
        row_count = MAX_ROW_COUNT
        column_count = int(np.ceil(channel_count / row_count))
    fig = plt.figure(2, figsize=(AXIS_LENGTH * column_count, AXIS_HEIGHT * row_count), clear=True)
    x = fft_frequencies
    
    # Plot each subplot
    for plot_index in range(channel_count):
        
        channel = channels_to_plot[plot_index]
        
        # Find channel index
        channel_index = np.where(channels==channel)
        channel_index = int(channel_index[0][0])
        
        # Mean power spectra to be plotted
        y1 = spectrum_db_12Hz[channel_index, :]
        y2 = spectrum_db_15Hz[channel_index, :]

        # Set up axes to maximize use of plot space
        if (plot_index == 0):
            ax = plt.subplot(row_count, column_count, plot_index + 1)
            ax.tick_params('x', labelbottom=False)
        elif ((plot_index + 1) > channel_count - column_count):
            ax = plt.subplot(row_count, column_count, plot_index + 1, sharex=fig.axes[0])
            ax.tick_params('x', labelbottom=True)
            # ax.set_xlabel('Voltage (uV)', fontsize=8)
        else:
            ax = plt.subplot(row_count, column_count, plot_index + 1, sharex=fig.axes[0])
            ax.tick_params('x', labelbottom=False)
            
        # Plot target ERP
        line_12Hz, = ax.plot(x, y1, label='12Hz')

        # Plot nontarget ERP
        line_15Hz, = ax.plot(x, y2, label='15Hz')
        
        # Add a dotted line at x=12 and x=15
        ax.axvline(x=12, linestyle='--', color='blue', linewidth=1)
        ax.axvline(x=15, linestyle='--', color='orange', linewidth=1)
        
        # Set x-axis range
        plt.xlim(fft_frequencies[0], fft_frequencies[-1])
        
        # Set subplot title and labels
        ax.set_title(f'Channel {channel}', fontsize=10)
        # ax.set_ylabel('Frequency (Hz)', fontsize=8)
        
        ax.legend(handles=[line_12Hz, line_15Hz], loc='upper right', fontsize=8)

    # Adjust layout for better spacing
    fig.set_tight_layout(True)   
    plt.suptitle(f'Frequency Content for SSVEP S{subject}')    
    fig.supxlabel('Voltage (uV)')
    fig.supylabel('Frequency (Hz)')

    # Show the plot
    plt.show()
    
    # Save the resulting plots as images (format: .png)
    plt.savefig(f'Images/S{subject}_SSVEP_Frequencies_for_{channel_count}_Channels.png')

    
    return spectrum_db_12Hz, spectrum_db_15Hz