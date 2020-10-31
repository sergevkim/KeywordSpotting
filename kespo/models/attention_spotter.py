from collections import OrderedDict

import torch
from torch import Tensor
from torch.nn import (
    BCELoss,
    Module,
    Sequential,
    Softmax,
    Tanh,
)
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchaudio.transforms import MelSpectrogram


class Encoder(Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
        ):
        super().__init__()
        self.cnn = None
        self.rnn = GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

    def forward(
            self,
            x: Tensor,
            hidden: Tensor,
        ) -> Tensor:
        x_1 = self.cnn(x)

        output, _ = self.rnn(
            input=x_1,
            h_0=hidden,
        )

        return output


class AverageAttention(Module):
    def __init__(
            self,
            T: int,
        ):
        super().__init__()
        self.T = T
        self.alpha = torch.full(
            size=(1, T),
            fill_value=1 / T,
        )

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        return self.alpha


class SoftAttention(Module):
    def __init__(
            self,
        ):
        super().__init__()
        self.blocks_ordered_dict = OrderedDict(
            Wb=Linear(
                in_channels=None,
                out_channels=None,
            ),
            tanh=Tanh(),
            v=Linear(
                in_channels=None,
                out_channels=None,
                biased=False,
            ),
            softmax=Softmax(),
        )
        self.alpher = Sequential(self.blocks)

    def forward(
            self,
            x: Tensor,
        ):
        alpha = self.alpher(x)

        return alpha


class AttentionSpotter(Module):
    def __init__(
            self,
            T: int,
            learning_rate: float=3e-4,
            device=torch.device('cpu'),
        ):
        super().__init__()
        self.T = T

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

        self.encoder = Encoder(
            input_size=None,
            hidden_size=None,
            num_layers=None,
        )
        self.attention = AverageAttention(
            T=self.T,
        )
        self.epilog_ordered_dict = OrderedDict(
            U=Linear(
                in_channels=None,
                out_channels=None,
                biased=False
            ),
            softmax=Softmax(),
        )
        self.epilog = Sequential(self.epilog_ordered_dict)

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        h = self.encoder(x)
        alpha = self.attention(h)

        c = alpha * h
        p = self.epilog(c)

        return p

    def training_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        waveforms, targets = batch
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

