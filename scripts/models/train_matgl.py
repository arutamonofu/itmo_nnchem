from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data_io import project_path
from src.metrics import compute_regression_metrics
from src.project_data import make_train_val_test_dataframes_for_budget
from src.result_schema import make_result_row, ordered_result_frame
from src.training_cli import add_budget_arguments, requested_budget_names

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
    torch = None
    matgl = None
    MGLDataset = None
    TransformedTargetModel = None


ALLOWED_SPLIT_STRATEGIES = ["random_iid", "element_set"]
MODEL_NAME = "matgl_megnet"
MODEL_FAMILY = "matgl"
RESULT_FILE = "matgl.csv"
NOTES = "Fine-tuning of pre-trained MEGNet (2 phases: frozen backbone -> full unfreeze)."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MatGL MEGNet fine-tuning script.")
    add_budget_arguments(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=None, help="Smoke-test alias: sets both training phases")
    parser.add_argument("--epochs-1", type=int, default=15, help="Epochs for Phase 1 (frozen backbone)")
    parser.add_argument("--epochs-2", type=int, default=50, help="Epochs for Phase 2 (unfrozen)")
    parser.add_argument("--lr-1", type=float, default=1e-3, help="Learning rate for Phase 1")
    parser.add_argument("--lr-2", type=float, default=1e-5, help="Learning rate for Phase 2")
    default_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--pretrained-model", default="MEGNet-Eform-MP-2018.6.1")
    parser.add_argument("--split-strategy", choices=ALLOWED_SPLIT_STRATEGIES, default="random_iid")
    args = parser.parse_args()
    if args.epochs is not None:
        args.epochs_1 = args.epochs
        args.epochs_2 = args.epochs
    return args


def load_data(budget_name: str, split_strategy: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        return make_train_val_test_dataframes_for_budget(
            budget_name=budget_name,
            split_strategy=split_strategy,
        )
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


def train_model(
    args: argparse.Namespace, 
    train_dataset: "MGLDataset", 
    val_dataset: "MGLDataset", 
    test_dataset: "MGLDataset",
    budget_name: str,
) -> "TransformedTargetModel":
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_dataset,
        val_data=val_dataset,
        test_data=test_dataset,
        collate_fn=collate_fn_graph,
        batch_size=args.batch_size,
        num_workers=0
    )

    loaded_model = matgl.load_model(args.pretrained_model)
    megnet_model = loaded_model.model
    data_mean = loaded_model.transformer.mean
    data_std = loaded_model.transformer.std

    print(f'Model fine-tuning with {budget_name} training budget.')
    print(f'\nPhase 1: Training output layer (frozen backbone), lr={args.lr_1}, epochs={args.epochs_1}...')
    for param in megnet_model.parameters():
        param.requires_grad = False
    for param in megnet_model.output_proj.parameters():
        param.requires_grad = True

    model_phase_1 = ModelLightningModule(model=megnet_model, data_mean=data_mean, data_std=data_std, lr=args.lr_1)
    logger_1 = CSVLogger(project_path('logs'), name=f'MEGNet_phase_1_budget_{budget_name}')
    
    trainer_phase_1 = L.Trainer(max_epochs=args.epochs_1, accelerator='auto', devices=1, logger=logger_1)
    trainer_phase_1.fit(model=model_phase_1, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f'\nPhase 2: Unfreezing all layers and fine-tuning, lr={args.lr_2}, epochs={args.epochs_2}...')
    for param in megnet_model.parameters():
        param.requires_grad = True

    model_phase_2 = ModelLightningModule(model=megnet_model, data_mean=data_mean, data_std=data_std, lr=args.lr_2)
    logger_2 = CSVLogger(project_path('logs'), name=f'MEGNet_phase_2_budget_{budget_name}')
    
    trainer_phase_2 = L.Trainer(max_epochs=args.epochs_2, accelerator='auto', devices=1, logger=logger_2)
    trainer_phase_2.fit(model=model_phase_2, train_dataloaders=train_loader, val_dataloaders=val_loader)

    final_model = TransformedTargetModel(model=megnet_model, target_transformer=loaded_model.transformer)
    finetuned_model_path = project_path('artifacts', 'models', 'finetuned', f'MEGNet_budget_{budget_name}')
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
    budget_name: str,
    seed: int,
    split_strategy: str,
) -> str:
    prediction_rel_path = (
        f"results/predictions/{split_strategy}_{MODEL_NAME}_{budget_name}_seed{seed}.csv"
    )
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
            (old_df["model_name"] == row["model_name"])
            & (old_df["budget_name"] == row["budget_name"])
            & (old_df["model_seed"].astype(int) == int(row["model_seed"]))
            & (old_df["split_id"] == row["split_id"])
        )
        new_df = pd.concat([old_df.loc[~duplicate], new_df], ignore_index=True)

    ordered_result_frame(new_df).to_csv(result_path, index=False)
    return result_path


def run_one_budget(args: argparse.Namespace, budget_name: str) -> None:
    train_df, val_df, test_df = load_data(budget_name, args.split_strategy)

    train_dataset, val_dataset, test_dataset = prepare_datasets(train_df, val_df, test_df)

    final_model = train_model(args, train_dataset, val_dataset, test_dataset, budget_name)

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
        budget_name=budget_name,
        seed=args.seed,
        split_strategy=args.split_strategy,
    )
    
    result_row = make_result_row(
        model_name=MODEL_NAME,
        model_family=MODEL_FAMILY,
        budget_name=budget_name,
        model_seed=args.seed,
        mae=test_metrics["mae"],
        rmse=test_metrics["rmse"],
        r2=test_metrics["r2"],
        predictions_path=predictions_path,
        notes=NOTES,
        split_strategy=args.split_strategy,
    )
    result_path = save_result_row(result_row)

    print(f"Loaded dataset: {len(train_df) + len(val_df) + len(test_df)} samples")
    print(f"Budget: {budget_name}")
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


def main() -> None:
    args = parse_args()

    if matgl is None:
        raise ImportError("MatGL, PyTorch and Lightning dependencies are required. Install them first.")

    for budget_name in requested_budget_names(args):
        run_one_budget(args, budget_name)


if __name__ == "__main__":
    main()
