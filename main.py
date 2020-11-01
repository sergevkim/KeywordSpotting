from argparse import ArgumentParser
from pathlib import Path

import torch

from kespo.datamodules import SpeechCommandsDataModule
from kespo.loggers import NeptuneLogger
from kespo.models import AttentionSpotter
from kespo.trainer import Trainer


def main(args):
    model = AttentionSpotter(T=81)
    datamodule = SpeechCommandsDataModule(
        data_dir=args['data_dir'],
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
    )
    datamodule.setup(val_ratio=args['val_ratio'])

    train_dataloader = datamodule.train_dataloader()
    #logger = NeptuneLogger(
    #    api_key=None,
    #    project_name=None,
    #)
    trainer = Trainer(
    #    logger=logger,
        max_epoch=args['max_epoch'],
        verbose=args['verbose'],
        version=args['version'],
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    args = dict(
        batch_size=16,
        data_dir=Path("data/raw"),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        learning_rate=3e-4,
        max_epoch=1,
        num_workers=4,
        val_ratio=0.1,
        verbose=False,
        version='0.1.0',
    )

    main(args)

