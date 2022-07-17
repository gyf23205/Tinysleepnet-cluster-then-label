from multitaper_spectrogram import multitaper_spectrogram  # import multitaper_spectrogram function from the multitaper_spectrogram_python.py file
import numpy as np  # import numpy
from scipy.signal import chirp  # import chirp generation function
import glob
import os
from data import load_data, my_get_subject_files
import scipy.stats as stats
from bandpower import*
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper


def extract_feature(eeg, fs):
    feature = []
    for file in eeg:
        file = np.squeeze(file)
        temp = []
        for i in range(file.shape[0]):
            temp.append(compute_features_each_seg(file[i], fs))
        feature.extend(np.array(temp))
    return feature

def compute_features_each_seg(eeg_seg, fs):
    band_freq = [[0.5, 4], [4, 8], [8, 12], [12, 20]]  # [Hz]
    band_num = len(band_freq)
    total_freq_range = [0.5, 20]  # [Hz]
    window_length = 2
    window_step = 1
    psd, freqs = psd_array_multitaper(eeg_seg, fs, adaptive=True,
                                      normalization='full', verbose=0)
    power = []
    freq_res = freqs[1] - freqs[0]
    for bi in range(band_num):
        band = band_freq[bi]
        band = np.asarray(band)
        low, high = band
        # Find index of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        bp = simps(psd[idx_band], dx=freq_res)
        total_band = np.logical_and(freqs >= total_freq_range[0], freqs <= total_freq_range[1])
        bp /= simps(psd[total_band], dx=freq_res)
        power.append(bp)
    return power


subject_files = glob.glob(os.path.join("./my_data/my_eeg_fpz_cz", "*.npz"))
test_files = my_get_subject_files(files=subject_files)
x, y, fs, _ = load_data(test_files, data_from_cluster=False)
feature = extract_feature(x[:2], fs)
print()