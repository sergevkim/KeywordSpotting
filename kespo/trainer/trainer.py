from pathlib import Path

import torch
import tqdm


class Trainer:
    def __init__(
            self,
            logger,
            max_epoch,
            verbose,
            version,
        ):
        self.logger = logger
        self.max_epoch = max_epoch
        self.verbose = verbose
        self.version = version

    def save_checkpoint(
            self,
            model,
            optimizer,
            epoch: int,
            checkpoints_dir: Path,
        ):
        checkpoint = {
            'model': model,
            'optimizer': optimizer,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        checkpoint_path = checkpoints_dir / f"v{self.version}-e{epoch}.hdf5"
        torch.save(checkpoint, checkpoint_path)

    @torch.enable_grad()
    def training_epoch(
            self,
            model,
            train_dataloader,
            optimizer,
        ):
        model.train()

        for batch_idx, batch in enumerate(tqdm.tqdm(train_dataloader)):
            for b in batch:
                print(b.shape)
            loss = model.training_step(batch, batch_idx)
            loss.backward()
            clip_grad_norm_(parameters=model.parameters(), max_norm=10)
            optimizer.step()
            optimizer.zero_grad()
            model.training_step_end()

        model.training_epoch_end()

    @torch.no_grad()
    def validation_epoch(
            self,
            model,
            val_dataloader,
        ):
        model.eval()

        for batch_idx, batch in enumerate(tqdm.tqdm(val_dataloader)):
            loss = model.validation_step(batch, batch_idx)
            model.validation_step_end()

        model.validation_epoch_end()

    def fit(
            self,
            model,
            datamodule,
        ):
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        optimizer = model.configure_optimizers()

        for epoch in range(1, self.max_epoch + 1):
            self.training_epoch(
                model=model,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
            )
            self.validation_epoch(
                model=model,
                val_dataloader=val_dataloader,
            )
            self.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                checkpoints_dir=Path.cwd() / "models",
            )

        return model

    def predict(
            self,
            model,
            datamodule,
        ):
        pass

