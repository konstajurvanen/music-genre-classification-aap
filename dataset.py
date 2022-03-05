#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Optional, Union, Dict
from pickle import load as pickle_load
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np

from feature_extraction import extract_and_serialize_features

__docformat__ = 'reStructuredText'
__all__ = ['GenreDataset']


class GenreDataset(Dataset):

    def __init__(self,
                 data_dir: Union[str, Path] = 'training',
                 n_mels: Optional[int] = 40,
                 load_into_memory: Optional[bool] = True) \
            -> None:
        """

        :param data_dir: Directory to read data from.
        :type data_dir: str
        :param data_parent_dir: Parent directory of the data, defaults\
                                to ``.
        :type data_parent_dir: str
        :param key_features: Key to use for getting the features,\
                             defaults to `features`.
        :type key_features: str
        :param key_class: Key to use for getting the class, defaults\
                          to `class`.
        :type key_class: str
        :param load_into_memory: Load the data into memory? Default to True
        :type load_into_memory: bool
        """
        super().__init__()
        data_parent_dir = 'mel_features_n_' + str(n_mels)
        data_path = Path(data_parent_dir, data_dir)
        
        # Extract and serialize features if not done already for the number of 
        # Mel frequency bands
        if not data_path.is_dir():  
            extract_and_serialize_features(n_mels)
            
        self.files = list(data_path.iterdir())
        self.load_into_memory = load_into_memory
        if self.load_into_memory:
            for i, a_file in enumerate(self.files):
                self.files[i] = self._load_file(a_file)

    @staticmethod
    def _load_file(file_path: Path) \
            -> Dict[str, Union[int, np.ndarray]]:
        """Loads a file using pathlib.Path

        :param file_path: File path.
        :type file_path: pathlib.Path
        :return: The file.
        :rtype: dict[str, int|numpy.ndarray]
        """
        with file_path.open('rb') as f:
            return pickle_load(f)

    def __len__(self) \
            -> int:
        """Returns the length of the dataset.

        :return: Length of the dataset.
        :rtype: int
        """
        return len(self.files)

    def __getitem__(self,
                    item: int) \
            -> Tuple[np.ndarray, int]:
        """Returns an item from the dataset.

        :param item: Index of the item.
        :type item: int
        :return: Features and class of the item.
        :rtype: (numpy.ndarray, int)
        """
        if self.load_into_memory:
            the_item: Dict[str, Union[int, np.ndarray]] = self.files[item]
        else:
            the_item = self._load_file(self.files[item])

        return the_item['features'], the_item['class']


# EOF
