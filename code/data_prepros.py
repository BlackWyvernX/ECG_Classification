import wfdb
import os
import numpy as np
from scipy.signal import butter, filtfilt

if os.path.exists('./mitdb'):
    print('Database already exists')
else:
    wfdb.dl_database('mitdb', './mitdb')

def bandpass_filter(signal, low=0.5, high=40.0, fs=360, order=3):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def gendataset(recordpath, X, Y):

  record = wfdb.rdrecord(recordpath, channel_names=["MLII"])
  data = record.p_signal.flatten()
  dendata = bandpass_filter(data)

  annotation = wfdb.rdann(recordpath, "atr")
  rpeakclass = annotation.symbol
  rpeak = annotation.sample

  label_map = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
    'V': 1, 'E': 1,                         # Ventricular
    'A': 2, 'a': 2, 'J': 2, 'S': 2,         # Supraventricular
    'F': 3,                                 # Fusion
    '/': 4, 'f': 4, 'Q': 4, '|': 4          # Unknown
  }

  labels = [label_map.get(sym, -1) for sym in rpeakclass]

  for i in range(len(rpeak)):
    if labels[i] == -1:
      continue
    start_index = max(0, rpeak[i] - 99)
    end_index = min(len(dendata), rpeak[i] + 201)
    if end_index - start_index != 300:
      continue
    x = dendata[start_index:end_index]
    x_data = (x - np.mean(x)) / np.std(x)
    X.append(x_data)
    Y.append(labels[i])

  return X, Y

def load_data():
  signalset = []
  labelset = []
  for i in range(100, 234):
    record_path = f"./mitdb/{i}"
    header_file = f"{record_path}.hea"
    if os.path.exists(header_file):
        try:
            signalset, labelset = gendataset(record_path, signalset, labelset)
            #print(f"Record {i} loaded successfully")
        except Exception as e:
            print(f"Could not read or plot record {i}: {e}")
    #else:
        #print(f"Record {i} header file not found: {header_file}")
  return signalset, labelset
