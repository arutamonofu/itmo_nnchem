from __future__ import annotations

import argparse
import sys
import os
import requests
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_io import project_path
from src.metrics import compute_regression_metrics
from src.project_data import make_train_val_test_dataframes
from src.result_schema import make_result_row, ordered_result_frame

try:
    import torch
    import lightning as L
    from lightning.pytorch.loggers import CSVLogger

    import matgl
    from matgl.config import DEFAULT_ELEMENTS
    from matgl.ext.pymatgen import Structure2Graph
    from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_graph
    from matgl.utils.training import ModelLightningModule
    from matgl.models import TransformedTargetModel
except ImportError:
    matgl = None
    MGLDataset = None
    TransformedTargetModel = None


ALLOWED_TRAIN_FRACTIONS = [0.025, 0.25, 1.0]
MODEL_NAME = "matgl_megnet"
MODEL_FAMILY = "matgl"
RESULT_FILE = "matgl.csv"
NOTES = "Fine-tuning of pre-trained MEGNet (2 phases: frozen backbone -> full unfreeze)."


def fraction_to_name(train_fraction: float) -> str:
    return str(float(train_fraction)).replace(".", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MatGL MEGNet fine-tuning script.")
    parser.add_argument("--train-fraction", type=float, required=True, choices=ALLOWED_TRAIN_FRACTIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs-1", type=int, default=15, help="Epochs for Phase 1 (frozen backbone)")
    parser.add_argument("--epochs-2", type=int, default=50, help="Epochs for Phase 2 (unfrozen)")
    parser.add_argument("--lr-1", type=float, default=1e-3, help="Learning rate for Phase 1")
    parser.add_argument("--lr-2", type=float, default=1e-5, help="Learning rate for Phase 2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained-model", default="MEGNet-MP-2018.6.1-Eform")
    return parser.parse_args()


def load_data(train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        return make_train_val_test_dataframes(train_fraction=train_fraction)
    except FileNotFoundError as exc:
        message = str(exc)
        if "dataset.pkl" in message:
            raise FileNotFoundError(
                "data/processed/dataset.pkl not found. Run python scripts/01_load_dataset.py first."
            ) from exc
        raise FileNotFoundError("Split files not found. Run python scripts/02_make_splits.py first.") from exc


def prepare_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple["MGLDataset", "MGLDataset", "MGLDataset"]:
    y_train = train_df['target'].to_numpy()
    y_val = val_df['target'].to_numpy()
    y_test = test_df['target'].to_numpy()
    
    converter = Structure2Graph(element_types=DEFAULT_ELEMENTS, cutoff=4.0)

    train_dataset = MGLDataset(
        structures=train_df['structure'].tolist(),
        converter=converter,
        labels={'labels': y_train}
    )

    val_dataset = MGLDataset(
        structures=val_df['structure'].tolist(),
        converter=converter,
        labels={'labels': y_val}
    )

    test_dataset = MGLDataset(
        structures=test_df['structure'].tolist(),
        converter=converter,
        labels={'labels': y_test}
    )

    return train_dataset, val_dataset, test_dataset


def download_matgl_model_from_git(model_name: str, model_dir: Path) -> Path | None:
    base_url = f"https://raw.githubusercontent.com/materialyzeai/matgl/main/pretrained_models/{model_name}"
    files =["model.json", "model.pt", "state.pt"]

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model {model_name}...")

    for file in files:
        file_url = f"{base_url}/{file}"
        response = requests.get(file_url)

        if response.status_code == 200:
            with open(model_dir / file, "wb") as f:
                f.write(response.content)
            print(f" Successfully installed: {file}")
        else:
            print(f" Error while downloading {file}. Status: {response.status_code}")
            if response.status_code == 404:
                print(" Check model name or path")
            return None

    print(f"Model saved in folder: {model_dir.absolute()}")
    return model_dir


def train_model(
    args: argparse.Namespace, 
    train_dataset: "MGLDataset", 
    val_dataset: "MGLDataset", 
    test_dataset: "MGLDataset"
) -> "TransformedTargetModel":
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_dataset,
        val_data=val_dataset,
        test_data=test_dataset,
        collate_fn=collate_fn_graph,
        batch_size=args.batch_size,
        num_workers=0
    )

    original_model_path = project_path('artifacts', 'models', 'pretrained', args.pretrained_model)
    download_path = download_matgl_model_from_git(args.pretrained_model, original_model_path)
    if download_path is None:
        raise RuntimeError("Failed to fetch pre-trained model.")

    loaded_model = matgl.load_model(download_path)
    megnet_model = loaded_model.model
    data_mean = loaded_model.transformer.mean
    data_std = loaded_model.transformer.std

    print(f'Model fine-tuning with {args.train_fraction} train fraction.')
    print(f'\nPhase 1: Training output layer (frozen backbone), lr={args.lr_1}, epochs={args.epochs_1}...')
    for param in megnet_model.parameters():
        param.requires_grad = False
    for param in megnet_model.output_proj.parameters():
        param.requires_grad = True

    model_phase_1 = ModelLightningModule(model=megnet_model, data_mean=data_mean, data_std=data_std, lr=args.lr_1)
    logger_1 = CSVLogger(project_path('logs'), name=f'MEGNet_phase_1_train_fraction_{fraction_to_name(args.train_fraction)}')
    
    trainer_phase_1 = L.Trainer(max_epochs=args.epochs_1, accelerator='auto', devices=1, logger=logger_1)
    trainer_phase_1.fit(model=model_phase_1, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f'\nPhase 2: Unfreezing all layers and fine-tuning, lr={args.lr_2}, epochs={args.epochs_2}...')
    for param in megnet_model.parameters():
        param.requires_grad = True

    model_phase_2 = ModelLightningModule(model=megnet_model, data_mean=data_mean, data_std=data_std, lr=args.lr_2)
    logger_2 = CSVLogger(project_path('logs'), name=f'MEGNet_phase_2_train_fraction_{fraction_to_name(args.train_fraction)}')
    
    trainer_phase_2 = L.Trainer(max_epochs=args.epochs_2, accelerator='auto', devices=1, logger=logger_2)
    trainer_phase_2.fit(model=model_phase_2, train_dataloaders=train_loader, val_dataloaders=val_loader)

    final_model = TransformedTargetModel(model=megnet_model, target_transformer=loaded_model.transformer)
    finetuned_model_path = project_path('artifacts', 'models', 'finetuned', f'MEGNet_train_fraction_{fraction_to_name(args.train_fraction)}')
    finetuned_model_path.mkdir(parents=True, exist_ok=True)
    final_model.save(finetuned_model_path)
    print(f"Model saved to {finetuned_model_path}")

    return final_model


def evaluate_model(model: "TransformedTargetModel", df: pd.DataFrame, device_name: str) -> np.ndarray:
    model.eval()
    device = torch.device(device_name)
    model = model.to(device)
    
    predictions = df['structure'].apply(
        lambda x: float(model.predict_structure(x).detach().cpu())
    ).to_numpy()
    
    return predictions


def save_predictions(
    *,
    sample_ids: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_fraction: float,
    seed: int,
) -> str:
    prediction_rel_path = f"results/predictions/{MODEL_NAME}_{fraction_to_name(train_fraction)}_seed{seed}.csv"
    prediction_path = project_path(*prediction_rel_path.split("/"))
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_id": sample_ids.astype(int).to_numpy(),
            "split": "test",
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).to_csv(prediction_path, index=False)
    return prediction_rel_path


def save_result_row(row: dict[str, object]) -> Path:
    result_path = project_path("results", "raw", RESULT_FILE)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame([row])
    if result_path.exists():
        old_df = pd.read_csv(result_path)
        duplicate = (
            (old_df["model"] == row["model"])
            & (old_df["train_fraction"].astype(float) == float(row["train_fraction"]))
            & (old_df["seed"].astype(int) == int(row["seed"]))
            & (old_df["split_id"] == row["split_id"])
        )
        new_df = pd.concat([old_df.loc[~duplicate], new_df], ignore_index=True)

    ordered_result_frame(new_df).to_csv(result_path, index=False)
    return result_path


def main() -> None:
    args = parse_args()

    if matgl is None:
        raise ImportError("MatGL, PyTorch and Lightning dependencies are required. Install them first.")
    
    train_df, val_df, test_df = load_data(args.train_fraction)

    train_dataset, val_dataset, test_dataset = prepare_datasets(train_df, val_df, test_df)

    final_model = train_model(args, train_dataset, val_dataset, test_dataset)

    print("\nRunning inference...")
    y_val_pred = evaluate_model(final_model, val_df, args.device)
    y_test_pred = evaluate_model(final_model, test_df, args.device)
    
    y_true_test = test_df["target"].to_numpy()
    val_metrics = compute_regression_metrics(val_df["target"].to_numpy(), y_val_pred)
    test_metrics = compute_regression_metrics(y_true_test, y_test_pred)

    predictions_path = save_predictions(
        sample_ids=test_df["sample_id"],
        y_true=y_true_test,
        y_pred=y_test_pred,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    
    result_row = make_result_row(
        model=MODEL_NAME,
        model_family=MODEL_FAMILY,
        train_fraction=args.train_fraction,
        seed=args.seed,
        mae=test_metrics["mae"],
        rmse=test_metrics["rmse"],
        r2=test_metrics["r2"],
        predictions_path=predictions_path,
        notes=NOTES
    )
    result_path = save_result_row(result_row)

    print(f"Loaded dataset: {len(train_df) + len(val_df) + len(test_df)} samples")
    print(f"Train fraction: {args.train_fraction}")
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Model: {MODEL_NAME}")
    print(f"Validation MAE: {val_metrics['mae']:.6f}")
    print(f"MAE: {test_metrics['mae']:.6f}")
    print(f"RMSE: {test_metrics['rmse']:.6f}")
    print(f"R2: {test_metrics['r2']:.6f}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved result row to: {result_path}")


if __name__ == "__main__":
    main()
