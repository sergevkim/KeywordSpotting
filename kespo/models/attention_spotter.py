from collections import OrderedDict

import torch
from torch import Tensor
from torch.nn import (
    BCELoss,
    Dropout,
    Module,
    ReLU,
    Sequential,
)
from torch.optim import Adam
from torch.optim.optimizers import Optimizer
from torchaudio.transforms import MelSpectrogram


class Encoder(Module):
    def __init__(
            self,
        ):
        super().__init__()
        self.cnn = None
        self.rnn = GRU(
            input_size=0,
            hidden_size=0,
            num_layers=0,
        )

    def forward(
            self,
            x: Tensor,
            hidden: Tensor,
        ) -> Tensor:
        x_1 = self.cnn(x)

        output, hidden = self.rnn(
            input=x_1,
            h_0=hidden,
        )

        return output, hidden


class AttentionSpotter(Module):
    def __init__(
            self,
            learning_rate: float=3e-4,
            device=torch.device('cpu'),
        ):
        super().__init__()
        self.device = device
        self.learning_rate = learning_rate
        self.criterion = BCELoss()
        self.mel_spectrogramer = MelSpectrogram(
            n_fft=1024,
            sample_rate=22000,
            win_length=1024,
            hop_length=256,
            f_min=0,
            f_max=800,
            n_mels=self.in_channels,
        ).to(self.device)

        self.encoder = None
        self.attention = None

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        x_1 = self.encoder(x)
        x_2 = self.attention(x_1)

        return x_2

    def training_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        waveforms, targets, waveform_lengths, target_lengths = batch
        waveforms = waveforms.to(self.device)
        targets = targets.to(self.device)
        mel_spectrograms = self.mel_spectrogramer(waveforms)

        predictions = self(mel_spectrograms)

        loss = self.criterion(
            log_probs=log_probs,
            targets=targets,
        )

        return loss

    def training_step_end(self):
        pass

    def training_epoch_end(self):
        print("Training epoch is over!")

    def validation_step(self, batch, batch_idx):
        pass

    def validation_step_end(self):
        pass

    def validation_epoch_end(self):
        print("Validation epoch is over!")

    def configure_optimizers(self) -> Optimizer:
        optimizer = Adam(
            params=self.parameters(),
            lr=self.learning_rate,
        )

        return optimizer

