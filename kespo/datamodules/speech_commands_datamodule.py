from pathlib import Path
from typing import Dict, List, Union

import einops
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from peach.utils import TokenConverter


class SpeechCommandsDataset(Dataset):
    def __init__(
            self,
            wav_paths: List[Path],
            targets: List[int],
            max_waveform_length: int=16000,
        ):
        self.wav_paths = wav_paths
        self.targets = targets
        self.max_waveform_length = max_waveform_length

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = einops.rearrange(waveform, 'b x -> (b x)')

        padded_waveform = F.pad(
            input=waveform,
            pad=(0, self.max_waveform_length - len(waveform)),
            mode='constant',
            value=0,
        )
        target = torch.tensor(self.targets[idx])

        result = (
            padded_waveform,
            target,
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

    @staticmethod
    def prepare_data(
            data_dir: Path,
            keywords: List[str],
        ) -> Dict[str, List[Union[Path, str]]]:
        wav_paths = list(f for f in data_dir.glob('**/*.wav'))
        targets = list()
        for p in wav_paths:
            flag = p.parents[0].name in keywords
            targets.append(int(flag))
        #TODO target: float?
        data = dict(
            wav_paths=wav_paths,
            targets=targets,
        )

        return data

    def setup(
            self,
            val_ratio,
        ):
        data = self.prepare_data(
            data_dir=self.data_dir,
            keywords=['right', 'marvin'],
        )
        wav_paths = data['wav_paths']
        targets = data['targets']

        full_dataset = SpeechCommandsDataset(
            wav_paths=wav_paths,
            targets=targets,
        )

        full_size = len(full_dataset)
        val_size = int(val_ratio * full_size)
        train_size = full_size - val_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset=full_dataset,
            lengths=[train_size, val_size],
        )

    def train_dataloader(self) -> DataLoader:
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return val_dataloader

    def test_dataloader(self):
        pass

