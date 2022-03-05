#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, MutableSequence
from pathlib import Path
from pickle import dump
from librosa.core import load as lb_load, stft
from librosa.filters import mel
from torchaudio.datasets import GTZAN
import numpy as np

from copy import deepcopy

__all__ = ['extract_and_serialize_features']


def extract_and_serialize_features(n_mels=40):
    parent_folder = './genres/'
    sub_folders = list(Path(parent_folder).iterdir())
    n_fft = 2048
    
    audio_folders = list(filter(lambda name: '.mf' not in str(name), sub_folders))
    
    def get_genre_from(path):
        return str(path).split("\\")[1]
    
    genres = [get_genre_from(dirname) for dirname in audio_folders]
    
    for subdir in audio_folders:
        files = list(Path(subdir).iterdir())
        
        for i, file in enumerate(files):
            data, sr = lb_load(path=file, sr=None, mono=True)
            # Minimum length of found file was 645 samples,
            # therefore we will cut all clips to that length
            features = extract_mel_band_energies(data, sr, n_fft, n_mels)[:,:645]
            genre = get_genre_from(subdir)
            print(f"Shape of the features {features.shape} of genre {genre}")
            genre_one_hot = create_one_hot_encoding(genre, genres)
            features_and_classes = {'features': features, 
                                    'class': genre_one_hot}
            data_purpose = '/training/' if i < 80 else '/testing/'
            out_dir = Path('mel_features_n_' + str(n_mels) + data_purpose)
            f_name = genre + '_' + str(i)
            serialize(f_name, features_and_classes, out_dir)

            if 'training' in data_purpose:
                data_noised = add_noise(data)
                features_noised = extract_mel_band_energies(data_noised, sr, n_fft, n_mels)[:,:645]
                features_noised = spec_augment(features_noised, n_mels)
                features_and_classes_noised = {
                    'features': features_noised, 
                    'class': genre_one_hot
                }
                f_name_noised = genre + '_noised_' + str(i)
                serialize(f_name_noised, features_and_classes_noised, out_dir)
                
    print(f"Serialised features with {n_mels} Mel-bands")
            

def create_one_hot_encoding(word: str,
                            unique_words: MutableSequence[str]) \
        -> np.ndarray:
            
    encoded = np.zeros((len(unique_words)))
    encoded[unique_words.index(word)] = 1
    return encoded


def serialize(
        f_name: Path,
        features_and_classes,
        output_directory: Path) \
        -> None:

    f_path = output_directory.joinpath(f_name)
    output_directory.mkdir(parents=True, exist_ok=True)
    with f_path.open('wb') as f:
        dump(features_and_classes, f)
        
        
def extract_mel_band_energies(data: np.ndarray,
                              sr: Optional[int] = 44100,
                              n_fft: Optional[int] = 1024,
                              n_mels: Optional[int] = 40,
                              hop_length: Optional[int] = 1024,
                              window: Optional[str] = 'hamming') \
        -> np.ndarray:
    spec = stft(y=data,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window)
    mel_filters = mel(sr=sr, 
                      n_fft=n_fft, 
                      n_mels=n_mels)
    MBEs = np.dot(mel_filters, np.abs(spec) ** 2)
    return MBEs


def add_noise(audio_signal: np.ndarray) -> np.ndarray:
    samples = audio_signal.shape[0]
    # Stdev. of noise is 1/10th of max in signal
    noise = np.random.normal(0, audio_signal.max()/10, size=samples)
    return audio_signal + noise


def spec_augment(mel_spectrogram: np.ndarray):
    spectrogram = deepcopy(mel_spectrogram)
    # Making possible mask size max 10 bands
    f = np.random.randint(low=0, high=10)
    f0 = np.random.randint(low=0, high=40-f)
    min, max = f0, f+f0
    for i in range(min, max):
        spectrogram[i,:] = 0
    return spectrogram



if __name__ == '__main__':
    download = False
    if download:
        GTZAN(root=".", download=download)
    extract_and_serialize_features(80)
