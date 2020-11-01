from collections import OrderedDict

import einops
import torch
from torch import Tensor
from torch.nn import (
    CrossEntropyLoss,
    GRU,
    Module,
    Linear,
    Sequential,
    Tanh,
)
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchaudio.transforms import MelSpectrogram


class Encoder(Module):
    def __init__(
            self,
            input_size: int=40,
            hidden_size: int=128,
            num_layers: int=1,
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
        ) -> Tensor:
        #x_1 = self.cnn(x)
        x_1 = x

        output, hidden = self.rnn(
            input=x_1,
        )

        return output


class AverageAttention(Module):
    def __init__(
            self,
            T: int,
        ):
        super().__init__()
        self.T = T

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        alpha = torch.full(
            size=(x.shape[0], self.T),
            fill_value=1 / self.T,
        )
        alpha = einops.rearrange(alpha, 'h (w 1) -> h w 1')

        return alpha


class SoftAttention(Module):
    def __init__(
            self,
        ):
        super().__init__()
        self.blocks_ordered_dict = OrderedDict(
            Wb=Linear(#TODO
                in_channels=None,
                out_channels=None,
            ),
            tanh=Tanh(),
            v=Linear(
                in_features=None,
                out_features=None,
                bias=False,
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
            in_channels: int=40,
            hidden_size: int=128,
            learning_rate: float=3e-4,
            device=torch.device('cpu'),
        ):
        super().__init__()
        self.T = T

        self.device = device
        self.learning_rate = learning_rate
        self.criterion = CrossEntropyLoss()
        self.mel_spectrogramer = MelSpectrogram(
            #n_fft=1024,
            sample_rate=16000,
            #win_length=1024,
            #hop_length=256,
            #f_min=0,
            #f_max=800,
            n_mels=in_channels,
        ).to(self.device)

        self.encoder = Encoder(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=1,
        )
        self.attention = AverageAttention(
            T=self.T,
        )
        self.epilog_ordered_dict = OrderedDict(
            U=Linear(
                in_features=hidden_size,
                out_features=3,
                bias=False,
            ),
            #softmax=Softmax(), #TODO remove
        )
        self.epilog = Sequential(self.epilog_ordered_dict)

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        h = self.encoder(x)
        alpha = self.attention(h)

        c_0 = alpha * h

        c = (alpha * h).sum(dim=1)
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
        mel_spec = self.mel_spectrogramer(waveforms)
        transposed_mel_spec = einops.rearrange(mel_spec, 'bs w h -> bs h w')

        predictions = self(torch.log(transposed_mel_spec))

        loss = self.criterion(
            input=predictions,
            target=targets,
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

