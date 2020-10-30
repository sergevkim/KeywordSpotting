import string
from pathlib import Path
from PIL import Image

import einops
import torch
import torchaudio
from torch import Tensor
from torch.nn import ZeroPad2d
from torch.utils.data import Dataset, DataLoader

from peach.utils import TokenConverter


class SpeechCommandsDataset(Dataset):
    def __init__(
            self,
            filenames,
            targets,
            max_waveform_length=17000,
        ):
        self.filenames = filenames
        self.targets = targets
        self.max_waveform_length = max_waveform_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        waveform, sample_rate = torchaudio.load(filename)
        waveform = einops.rearrange(waveform, 'b x -> (b x)')

        padding = (0, self.max_waveform_length - len(waveform), 0, 0)
        zero_padder = ZeroPad2d(padding=padding)
        padded_waveform = zero_padder(waveform)

        result = (
            padded_waveform,
            padded_target,
        )

        return result


class SpeechCommandsDataModule:
    def __init__(
            self,
            data_dir: Path,
            batch_size: int,
            num_workers: int,
        ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        filenames = list()
        targets = list()

        data = dict(
            filenames=wav_filenames,
            targets=targets,
        )

        return data

    def setup(
            self,
            val_ratio,
        ):
        data = self.prepare_data()
        wav_filenames = data['filenames']
        targets = data['targets']

        full_dataset = LJSpeechDataset(
            filenames=wav_filenames,
            targets=targets,
        )

        full_size = len(full_dataset)
        val_size = int(val_ratio * full_size)
        train_size = full_size - val_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset=full_dataset,
            lengths=[train_size, val_size],
        )

    def train_dataloader(self):
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return val_dataloader

    def test_dataloader(self):
        pass

