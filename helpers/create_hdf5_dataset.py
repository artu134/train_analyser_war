import h5py
import pandas as pd
import numpy as np
import csv
import os
import av
import io


# Stolen from EfficientAT
def decode_mp3(mp3_arr):
    """
    decodes an array if uint8 representing an mp3 file
    :rtype: np.array
    """
    container = av.open(io.BytesIO(mp3_arr.tobytes()))
    stream = next(s for s in container.streams if s.type == 'audio')
    a = []
    for i, packet in enumerate(container.demux(stream)):
        for frame in packet.decode():
            a.append(frame.to_ndarray().reshape(-1))
    waveform = np.concatenate(a)
    if waveform.dtype != 'float32':
        raise RuntimeError("Unexpected wave type")
    return waveform


def collect_files(root):
    folders = []
    files = []
    for folder in os.listdir(root):
        for file in os.listdir(os.path.join(root, folder)):
            folders.append(folder)
            files.append(file)
    return folders, files


source_path = "C:/Users/IronTony/Projects/data/dataset/train_mp3"
dest_path = "C:/Users/IronTony/Projects/data/dataset/dataset.hdf"

labels = os.listdir(source_path)

folders, files = collect_files(source_path)
# PaSST and EfficientAT perform multilabel classification which means that on one audio there might be multiple targets
# (airplane and drone). Binary Cross Entropy is used as a loss function: [0.1, 0.6, 0.2, 0.05, 0.05] vs [0, 1, 0, 0, 0]
# For some reason they use bit representation for the label (perhaps hdf5 cannot store arrays).
# In they dataset they do np.unpackbits to get an array from the int (e.g. 8 -> 00001000 -> [0, 0, 0, 0, 1, 0, 0, 0])
# That is why here we do this weird 2 ** index
y = [2 ** labels.index(folder) for folder in folders]

dt = h5py.vlen_dtype(np.dtype('uint8'))
with h5py.File(dest_path, 'w') as hf:
    audio_name = hf.create_dataset('audio_name', shape=((len(files),)), dtype=h5py.string_dtype())
    waveform = hf.create_dataset('mp3', shape=((len(files),)), dtype=dt)
    target = hf.create_dataset('target', shape=((len(files),)), dtype='uint8')
    for i in range(len(files)):
        file_path = os.path.join(source_path, folders[i], files[i])
        a = np.fromfile(file_path, dtype='uint8')
        audio_name[i] = files[i]
        waveform[i] = a
        target[i] = y[i]

# Validate the created hdf5 file
# Make sure to do it, otherwise you will get errors during training
with h5py.File(dest_path, "r") as f:
    # Print everything
    # for key in f.keys():
    #     for i in range(len(f[key])):
    #         print(f"{i}: {f[key][i]}")

    all_keys = list(f.keys())
    name = all_keys[0]
    waveform = all_keys[1]
    label = all_keys[2]
    for i in range(len(f[waveform])):
        try:
            decode_mp3(f[waveform][i])
        except:
            print(f"{f[name][i]} - {f[label][i]}")
