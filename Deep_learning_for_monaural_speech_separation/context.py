import keras
import librosa
import numpy as np
import os, time
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import librosa
import librosa.display
import seaborn as sns
import pandas as pd

root = '/home/brightLLer/jupyter_notebook/speech/180403/'
time_steps = 105
n_features = 513
n_epochs = 16
n_samples = 59216
batch_size = 512
n_batch = n_samples // batch_size 
val_set_root = os.path.join(root, 'val_set')
spectra_data_root = os.path.join(root, 'complex_spectra_data')


def shift_one_step(mono_signal, interval=2):
    first = mono_signal[:-interval]
    last = mono_signal[-interval:]
    shift_mono_signal = np.hstack((last, first))
    return shift_mono_signal

def circul_shift(mono_signal, interval=1, verbose=False, output_dir=None):
    assert len(mono_signal.shape) == 1, 'the audio signal must be monarual'
    signal = mono_signal.copy()
    signals_list = [mono_signal]
    steps = len(mono_signal) // interval
    for step in range(steps):
        signal = shift_one_step(signal, interval)
        signals_list.append(signal.copy())
        if verbose:
            print("step {:d}:".format(step + 1), signal)
        if output_dir is not None:
            output_path = os.path.join(output_dir, "M{:04d}.wav".format(step + 1))
            librosa.output.write_wav(output_path, signal, sr=16000)
    return signals_list

def load_val_data():
#     x = pad_sequences([librosa.load(os.path.join(root, 'val_set/mix_val.wav'), sr=16000)[0]], maxlen=53040, dtype='float32', padding='post')[0]
    y1 = pad_sequences([librosa.load(os.path.join(root, 'val_set/SI590.wav'), sr=16000)[0]], maxlen=53040, dtype='float32', padding='post')[0]
    y2 = pad_sequences([librosa.load(os.path.join(root, 'val_set/SI649.wav'), sr=16000)[0]], maxlen=53040, dtype='float32', padding='post')[0]
#     x_val = librosa.util.fix_length(x, 53040 + 1024 // 2)
    y1_val = librosa.util.fix_length(y1, 53040 + 1024 // 2)
    y2_val = librosa.util.fix_length(y2, 53040 + 1024 // 2)
    x_val = y1_val + y2_val
    X_val = librosa.stft(x_val, n_fft=1024, hop_length=512).T[np.newaxis, :, :]
    Y1_val = librosa.stft(y1_val, n_fft=1024, hop_length=512).T[np.newaxis, :, :]
    Y2_val = librosa.stft(y2_val, n_fft=1024, hop_length=512).T[np.newaxis, :, :]
    return X_val, Y1_val, Y2_val, x_val, y1, y2


def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


def load_test_data():
    y1 = pad_sequences([librosa.load(os.path.join(root, 'test_set/SA2.wav'), sr=16000)[0]], maxlen=53040, dtype='float32', padding='post')[0]
    y2 = pad_sequences([librosa.load(os.path.join(root, 'test_set/SA1.wav'), sr=16000)[0]], maxlen=53040, dtype='float32', padding='post')[0]
#     x_val = librosa.util.fix_length(x, 53040 + 1024 // 2)
    y1_val = librosa.util.fix_length(y1, 53040 + 1024 // 2)
    y2_val = librosa.util.fix_length(y2, 53040 + 1024 // 2)
    x_val_raw = y1_val + y2_val
#     noise = wgn(x_val_raw, 0)
    noise = 0
    x_val = x_val_raw + noise
    X_val = librosa.stft(x_val, n_fft=1024, hop_length=512).T[np.newaxis, :, :]
    Y1_val = librosa.stft(y1_val, n_fft=1024, hop_length=512).T[np.newaxis, :, :]
    Y2_val = librosa.stft(y2_val, n_fft=1024, hop_length=512).T[np.newaxis, :, :]
    return X_val, Y1_val, Y2_val, x_val, y1, y2

def create_def_est_sources(y1, y2, y1_estimated, y2_estimated):
    reference_sources = np.concatenate([y1[np.newaxis, :], y2[np.newaxis, :]], axis=0)
    estimated_sources = np.concatenate([y1_estimated[np.newaxis, :], y2_estimated[np.newaxis, :]], axis=0)
    return reference_sources, estimated_sources

def plot_bss_eval(sdr, sir, sar):
    mean_metrics = [np.mean(sdr), np.mean(sir), np.mean(sar)]
    male_metrics = [sdr[0], sir[0], sar[0]]
    female_metrics = [sdr[1], sir[1], sar[1]]
    metrics_name = ['SDR', 'SIR', 'SAR']
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 3))
    plt.subplot(131)
    avg_pd = pd.DataFrame({'metrics': metrics_name, 'value': mean_metrics})
    g_avg = sns.barplot(x='metrics', y='value', data=avg_pd, label='avg')
    for index, row in avg_pd.iterrows():
        g_avg.text(row.name, row.value + 0.5, '{:.2f}dB'.format(row.value), color='black', ha="center")
    plt.legend(loc='upper left')
    plt.subplot(132)
    male_pd = pd.DataFrame({'metrics': metrics_name, 'value': male_metrics})
    g_male = sns.barplot(x='metrics', y='value', data=male_pd, label='male')
    for index, row in male_pd.iterrows():
        g_male.text(row.name, row.value + 0.5, '{:.2f}dB'.format(row.value), color='black', ha="center")
    plt.legend(loc='upper left')
    plt.subplot(133)
    female_pd = pd.DataFrame({'metrics': metrics_name, 'value': female_metrics})
    g_female = sns.barplot(x='metrics', y='value', data=female_pd, label='female')
    for index, row in female_pd.iterrows():
        g_female.text(row.name, row.value + 0.5, '{:.2f}dB'.format(row.value), color='black', ha="center")
    plt.legend(loc='upper left')

def plot_spectra(X, Y1, Y2, Y1_estimated, Y2_estimated):
    plt.figure(figsize=(22, 3))
    plt.subplot(151)
    librosa.display.specshow(librosa.amplitude_to_db(X), x_axis='time', y_axis='log', sr=16000)
    plt.colorbar()
    plt.title('mixture source')
    plt.subplot(152)
    librosa.display.specshow(librosa.amplitude_to_db(Y1), x_axis='time', y_axis='log', sr=16000)
    plt.colorbar()
    plt.title('male source ground truth')
    plt.subplot(153)
    librosa.display.specshow(librosa.amplitude_to_db(Y2), x_axis='time', y_axis='log', sr=16000)
    plt.colorbar()
    plt.title('female source ground truth')
    plt.subplot(154)
    librosa.display.specshow(librosa.amplitude_to_db(Y1_estimated), x_axis='time', y_axis='log', sr=16000)
    plt.colorbar()
    plt.title('male source estmated')
    plt.subplot(155)
    librosa.display.specshow(librosa.amplitude_to_db(Y2_estimated), x_axis='time', y_axis='log', sr=16000)
    plt.colorbar()
    plt.title('female source estimated')

def extract_bss_eval_statistic(logfilename):
    with open(logfilename) as f:
        logs = f.readlines()
    SDRs = [float(log[11: 16].strip()) for i, log in enumerate(logs) if i % 26 == 25]
    SIRs = [float(log[25: 30].strip()) for i, log in enumerate(logs) if i % 26 == 25]
    SARs = [float(log[-7: -1].strip()) for i, log in enumerate(logs) if i % 26 == 25]
    return SDRs, SIRs, SARs


def flow_from_directory(start, end):
    mix_root = os.path.join(root, 'complex_spectra_data/mixture_v1')
    male_root = os.path.join(root, 'complex_spectra_data/male_v1')
    female_root = os.path.join(root, 'complex_spectra_data/female_v1')
    mix_data_batch = []
    male_data_batch = []
    female_data_batch = []
    for i in range(start, end):
        mix_data_file = os.path.join(mix_root, 'mixture{:05d}.npy'.format(i + 1))
        male_data_file = os.path.join(male_root, 'male{:05d}.npy'.format(i + 1))
        female_data_file = os.path.join(female_root, 'female{:05d}.npy'.format(i + 1))
        mix_data_batch.append(np.load(mix_data_file)[np.newaxis, :, :])
        male_data_batch.append(np.load(male_data_file)[np.newaxis, :, :])
        female_data_batch.append(np.load(female_data_file)[np.newaxis, :, :])
    mix_batch = np.concatenate(mix_data_batch, axis=0)
    male_batch = np.concatenate(male_data_batch, axis=0)
    female_batch = np.concatenate(female_data_batch, axis=0)
    return mix_batch, male_batch, female_batch


