#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, MutableSequence
from pathlib import Path
from pickle import dump
from librosa.core import load as lb_load, stft
from librosa.filters import mel
import numpy as np

__all__ = ['extract_and_serialize_features']

def extract_and_serialize_features(n_mels):
    parent_folder = './genres/'
    sub_folders = list(Path(parent_folder).iterdir())
    n_fft = 2048
    hop_length = 1024
    window = 'hamming'
    
    audio_folders = list(filter(lambda name: '.mf' not in str(name), sub_folders))
    
    def get_genre_from(path):
        return str(path).split("\\")[1]
    
    genres = [ get_genre_from(dirname) for dirname in audio_folders ]
    
    for subdir in audio_folders:
        files = list(Path(subdir).iterdir())
        
        for i, file in enumerate(files):
            data, sr = lb_load(path=file, sr=None, mono=True)
            spec = stft(
                y=data,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window)
            
            features = extract_mel_band_energies(spec, sr, n_fft, n_mels)
            genre = get_genre_from(subdir)
            genre_one_hot = create_one_hot_encoding(genre, genres)
            features_and_classes = {'features': features, 
                                    'class': genre_one_hot}
            data_purpose = '/training/' if i < 70 else '/testing/'
            out_dir = Path('mel_features_n_' + str(n_mels) + data_purpose)
            f_name = genre + '_' + str(i)
            serialize(f_name, features_and_classes, out_dir)
                
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
        
        
def extract_mel_band_energies(spec: np.ndarray,
                              sr: Optional[int] = 44100,
                              n_fft: Optional[int] = 1024,
                              n_mels: Optional[int] = 40) \
        -> np.ndarray:
    mel_filters = mel(sr=sr, 
                      n_fft=n_fft, 
                      n_mels=n_mels)
    MBEs = np.dot(mel_filters, np.abs(spec) ** 2)
    return MBEs


if __name__ == '__main__':
    extract_and_serialize_features()