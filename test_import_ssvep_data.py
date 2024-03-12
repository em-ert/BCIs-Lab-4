"""
Name:        test_import_ssvep_data.py
Authors:     Emily Ertle, Nick Hanna
Description: serves as driver for import_ssvep_data.py and contains answers to assignment
"""

# %% Cell 1: Load the data
import import_ssvep_data
import numpy as np

subject = 1
data_dict = import_ssvep_data.load_ssvep_data(subject)
print(np.shape((data_dict)))

# %% Cell 2: Plot the data
import_ssvep_data.plot_raw_data(data_dict, subject)

# %% Cell 3: Extract the Epochs
# call this function to extract 20-second epochs (starting at the event onset) from your data
eeg_epochs, epoch_times, is_trial_15Hz = import_ssvep_data.epoch_ssvep_data(data_dict, epoch_start_time=0, epoch_end_time=20)

#%% Cell 4: Take the Fourier Transform
fs = data_dict['fs'] 
eeg_epochs_fft, freqs = import_ssvep_data.get_frequency_spectrum(eeg_epochs, fs)


# %% Cell 5: Plot the Power Spectra
spectrum_db_12Hz, spectrum_db_15Hz = import_ssvep_data.plot_power_spectrum(eeg_epochs_fft, freqs, is_trial_15Hz, data_dict['channels'], ['Fz', 'Oz'], subject)

# %% Cell 6:
'''
1. On some electrodes, there are peaks in the spectra at 12Hz for 12Hz trials and 15Hz for 
15Hz trials. What is the name for the brain signal that leads to these peaks? Where in the 
brain do they originate and why (i.e., what function does this brain region serve)?

    The peaks in the spectra at 12 Hz for 12 Hz trials and 15 Hz 
    for 15Hz trials are the Steady-State Visually Evoked Potentials 
    (SSVEPs) themselves. These signals originate in the visual cortex, 
    in the occipital lobe in the back of the brain. When the brain is 
    exposed to the frequency of stimulus we observed via the input 
    signal of the checkerboard pattern, it causes neurons to synchronously 
    (with some caveats) fire at that same frequency. 
    This signal can then be monitored via EEG.
    
    Beverina F, Palmas G, Silvoni S, Piccione F, Giove S (2003). 
    "User adaptive BCIs: SSVEP and P300 based interfaces". PsychNol. J. 1: 331-54. 

2. There are smaller peaks at integer multiples of these peak frequencies. What do you call 
these integer multiples? Why might we see them in the data?

    These smaller peaks at integer multiples of these peak frequencies are called harmonics. 
    We discussed these briefly in class. For example, if we are looking at a 12Hz signal, 
    we should see smaller peaks at 24Hz, 36Hz, etc. We can see these for several reasons. 

    1) There can be muliple temporal frequencies, or because the system is nonlinear. 
    Linear response means that the brain responds directly to the system, 
    but as the brain is a non-linear system, its not always proportional to the stimulus. 
    For example, there are refractory periods and other fundemental mechanics 
    that are just not linear in processing.
    
    2) Part of this can also be artifacts from a Discrete Fourier 
    Transform as it can display harmonic artifacts due to the finite 
    nature of the data and other characteristics in its calculations.
    
    Norcia, A. M., Appelbaum, L. G., Ales, J. M., Cottereau, B. R., & Rossion, B. (2015). 
    The steady-state visual evoked potential in vision research: 
    A review. Journal of Vision, 15(6), 4. https://doi.org/10.1167/15.6.4


3. There's a peak in the spectra at 50Hz. What is the most likely cause for this peak?
    
    This is most likely noise from the power line to the BCI. Such noise is very common
    around 50 or 60 Hz (depending on the country in which the study occurred).
    
    Leske, S., & Dalal, S. S. (2019). Reducing power line noise in EEG and MEG data via spectrum 
    interpolation. NeuroImage, 189, 763–776. https://doi.org/10.1016/j.neuroimage.2019.01.026


4. Besides the sharp peaks just described, there's a frequency spectrum where the power is 
roughly proportional to 1 over the frequency. This is a common pattern in biological 
signals and is sometimes called “1/f” or “neural noise” in EEG. But on top of that, there’s 
a slight upward bump around 10 Hz on some electrodes. What brain signal might cause 
this extra power at about 10Hz? Which channels is it most prominently observed on, and 
why might this be?

    This extra power is likely caused by alpha waves, which are commonly around 10Hz. The source
    of these signals is primarily the occipital lobe, though they are also found in the 
    fronto-central region during REM sleep. Evidence suggests occipital alpha waves may
    originate in the thalamus.
    
    Source: Class powerpoint 13 (SSVEPs).
    


'''