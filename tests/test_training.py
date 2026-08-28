"""Tests for epoch loops, checkpoints, early stopping, and trainer history."""

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from urban_sound_robustness.training import (
    EarlyStopping,
    Trainer,
    create_loss_function,
    create_optimizer,
    load_checkpoint,
    run_epoch,
    save_checkpoint,
)


def _samples() -> list[dict]:
    samples = []
    for index in range(16):
        label = index % 2
        features = torch.zeros(1, 2, 4)
        features[:, :, label::2] = 1.0
        samples.append(
            {
                "features": features,
                "label": torch.tensor(label, dtype=torch.long),
            }
        )
    return samples


def _model() -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(8, 2))


def _dropout_model() -> nn.Module:
    """Include stochastic model behavior so RNG restoration is exercised."""
    return nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(8, 2))


def _loaders(seed: int) -> tuple[DataLoader, DataLoader]:
    train_generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(
        _samples(),
        batch_size=4,
        shuffle=True,
        generator=train_generator,
    )
    validation_loader = DataLoader(
        _samples(),
        batch_size=4,
        shuffle=False,
        generator=validation_generator,
    )
    return train_loader, validation_loader


def _training_settings() -> dict:
    return {
        "epochs": 2,
        "mixed_precision": False,
        "gradient_accumulation_steps": 1,
        "loss": {"name": "cross_entropy"},
        "optimizer": {
            "name": "adamw",
            "learning_rate": 0.01,
            "weight_decay": 0.0,
        },
        "scheduler": {
            "name": "reduce_on_plateau",
            "mode": "max",
            "factor": 0.5,
            "patience": 1,
        },
        "early_stopping": {
            "enabled": False,
            "monitor": "macro_f1",
            "mode": "max",
            "patience": 2,
        },
        "checkpointing": {
            "save_best": True,
            "save_last": True,
            "monitor": "macro_f1",
            "mode": "max",
        },
        "logging": {"tensorboard": False},
    }


def test_training_epoch_updates_parameters() -> None:
    """A training pass should execute loss, backward, and optimizer steps."""
    model = _model()
    optimizer = create_optimizer(
        model.parameters(),
        _training_settings()["optimizer"],
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]

    result = run_epoch(
        model,
        DataLoader(_samples(), batch_size=4),
        create_loss_function({"name": "cross_entropy"}),
        torch.device("cpu"),
        ["zero", "one"],
        optimizer=optimizer,
    )

    assert result.num_batches == 4
    assert result.loss > 0
    assert any(
        not torch.equal(original, updated)
        for original, updated in zip(before, model.parameters())
    )


def test_validation_epoch_does_not_update_parameters() -> None:
    """Evaluation should collect metrics without gradient updates."""
    model = _model()
    before = [parameter.detach().clone() for parameter in model.parameters()]

    result = run_epoch(
        model,
        DataLoader(_samples(), batch_size=4),
        create_loss_function({"name": "cross_entropy"}),
        torch.device("cpu"),
        ["zero", "one"],
    )

    assert set(result.metrics) >= {"accuracy", "macro_f1"}
    for original, current in zip(before, model.parameters()):
        torch.testing.assert_close(original, current)


def test_early_stopping_resets_on_improvement() -> None:
    """Patience should count consecutive non-improving epochs."""
    stopping = EarlyStopping(mode="max", patience=2)

    assert stopping.update(0.5) is False
    assert stopping.update(0.4) is False
    assert stopping.update(0.6) is False
    assert stopping.update(0.5) is False
    assert stopping.update(0.5) is True


def test_trainer_saves_history_and_best_last_checkpoints(tmp_path: Path) -> None:
    """A short fit should persist inspectable experiment state."""
    trainer = Trainer(
        _model(),
        ["zero", "one"],
        _training_settings(),
        device=torch.device("cpu"),
        checkpoint_directory=tmp_path / "checkpoints",
        history_path=tmp_path / "history.csv",
    )
    loader = DataLoader(_samples(), batch_size=4)

    outcome = trainer.fit(loader, loader, epochs=2)

    assert outcome.epochs_completed == 2
    assert outcome.best_checkpoint is not None
    assert outcome.last_checkpoint is not None
    assert outcome.best_checkpoint.is_file()
    assert outcome.last_checkpoint.is_file()
    assert outcome.history_path.is_file()
    saved_history = pd.read_csv(outcome.history_path)
    assert list(saved_history["epoch"]) == [1, 2]
    assert "validation_macro_f1" in saved_history


def test_checkpoint_can_restore_model_and_optimizer(tmp_path: Path) -> None:
    """Saved model and optimizer states should support resuming."""
    model = _model()
    optimizer = create_optimizer(
        model.parameters(),
        _training_settings()["optimizer"],
    )
    checkpoint_path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=3,
        metrics={"macro_f1": 0.7},
    )
    expected = [parameter.detach().clone() for parameter in model.parameters()]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    checkpoint = load_checkpoint(checkpoint_path, model, optimizer=optimizer)

    assert checkpoint["epoch"] == 3
    for restored, target in zip(model.parameters(), expected):
        torch.testing.assert_close(restored, target)


def test_resumed_training_matches_uninterrupted_training(tmp_path: Path) -> None:
    """An epoch-boundary restart must reproduce uninterrupted model state."""
    torch.manual_seed(123)
    uninterrupted_model = _dropout_model()
    uninterrupted_train, uninterrupted_validation = _loaders(900)
    uninterrupted_trainer = Trainer(
        uninterrupted_model,
        ["zero", "one"],
        _training_settings(),
        device=torch.device("cpu"),
        checkpoint_directory=tmp_path / "uninterrupted" / "checkpoints",
        history_path=tmp_path / "uninterrupted" / "history.csv",
    )
    uninterrupted_outcome = uninterrupted_trainer.fit(
        uninterrupted_train,
        uninterrupted_validation,
        epochs=4,
    )
    expected_parameters = [
        parameter.detach().clone() for parameter in uninterrupted_model.parameters()
    ]

    torch.manual_seed(123)
    interrupted_model = _dropout_model()
    interrupted_train, interrupted_validation = _loaders(900)
    interrupted_directory = tmp_path / "interrupted"
    interrupted_trainer = Trainer(
        interrupted_model,
        ["zero", "one"],
        _training_settings(),
        device=torch.device("cpu"),
        checkpoint_directory=interrupted_directory / "checkpoints",
        history_path=interrupted_directory / "history.csv",
    )
    partial_outcome = interrupted_trainer.fit(
        interrupted_train,
        interrupted_validation,
        epochs=2,
    )
    checkpoint = torch.load(
        partial_outcome.last_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["epoch"] == 2
    assert checkpoint["scheduler_state_dict"] is not None
    assert len(checkpoint["history"]) == 2
    assert checkpoint["random_states"]["torch"] is not None
    assert checkpoint["data_loader_generator_states"]["train"] is not None

    # Simulate unrelated process activity between a power outage and restart.
    torch.manual_seed(999)
    resumed_model = _dropout_model()
    resumed_train, resumed_validation = _loaders(900)
    resumed_trainer = Trainer(
        resumed_model,
        ["zero", "one"],
        _training_settings(),
        device=torch.device("cpu"),
        checkpoint_directory=interrupted_directory / "checkpoints",
        history_path=interrupted_directory / "history.csv",
    )
    resumed_outcome = resumed_trainer.fit(
        resumed_train,
        resumed_validation,
        epochs=4,
        resume_from=partial_outcome.last_checkpoint,
    )

    for resumed_parameter, expected_parameter in zip(
        resumed_model.parameters(),
        expected_parameters,
    ):
        torch.testing.assert_close(
            resumed_parameter,
            expected_parameter,
            rtol=0,
            atol=0,
        )
    pd.testing.assert_frame_equal(
        resumed_outcome.history.reset_index(drop=True),
        uninterrupted_outcome.history.reset_index(drop=True),
    )
    assert resumed_outcome.epochs_completed == 4
