from __future__ import annotations

import argparse
import sys
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
    from lightning.pytorch.callbacks import EarlyStopping
    from lightning.pytorch.loggers import CSVLogger

    import matgl
    from matgl.config import DEFAULT_ELEMENTS
    from matgl.ext.pymatgen import Structure2Graph
    from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_graph
    from matgl.utils.training import ModelLightningModule
    from matgl.models import TransformedTargetModel
except ImportError:
    torch = None
    L = None
    matgl = None
    MGLDataset = None
    TransformedTargetModel = None


ALLOWED_TRAIN_FRACTIONS = [0.025, 0.25, 1.0]
ALLOWED_STRATEGIES = ["frozen", "differential", "full"]
BASE_MODEL_NAME = "matgl_megnet"
PRETRAINED_MODEL_NAME = "MEGNet-Eform-MP-2018.6.1"
MODEL_FAMILY = "matgl"
RESULT_FILE = "matgl.csv"
NOTES = "Fine-tuning of pre-trained MEGNet."


def fraction_to_name(train_fraction: float) -> str:
    return str(float(train_fraction)).replace(".", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MatGL MEGNet fine-tuning script.")
    parser.add_argument("--train-fraction", type=float, required=True, choices=ALLOWED_TRAIN_FRACTIONS)
    parser.add_argument("--strategy", type=str, required=True, choices=ALLOWED_STRATEGIES,
                        help="Fine-tuning strategy: 'frozen' (head only), 'differential' (gradual unfreeze), 'full' (end-to-end)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500, help="Max epochs. EarlyStopping will halt training earlier if no improvement")
    default_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
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

    loaded_model = matgl.load_model(PRETRAINED_MODEL_NAME)
    megnet_model = loaded_model.model
    y_train = train_dataset.labels["labels"]
    data_mean = float(np.mean(y_train))
    data_std = float(np.std(y_train))

    print(f'Model fine-tuning with {args.train_fraction} train fraction.')
    if args.strategy == "frozen":
        print("Strategy: 'Frozen' -> Frozen graph layers, only MLP-head train (output_proj)")
        for param in megnet_model.parameters():
            param.requires_grad = False
        for param in megnet_model.output_proj.parameters():
            param.requires_grad = True
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, megnet_model.parameters()), 
            lr=1e-3, 
            weight_decay=1e-5
        )
        patience = 30
        base_lr = 1e-3
        
    elif args.strategy == "differential":
        print("Strategy: 'Differential LRs' -> Everything is defrosted but graph layers are learning 100 times slower than MLP")
        for param in megnet_model.parameters():
            param.requires_grad = True
        
        param_groups =[
            {"params": megnet_model.embedding.parameters(), "lr": 1e-5},
            {"params": megnet_model.edge_encoder.parameters(), "lr": 1e-5},
            {"params": megnet_model.node_encoder.parameters(), "lr": 1e-5},
            {"params": megnet_model.state_encoder.parameters(), "lr": 1e-5},
            {"params": megnet_model.blocks.parameters(), "lr": 1e-5},
            {"params": megnet_model.edge_s2s.parameters(), "lr": 1e-4},
            {"params": megnet_model.node_s2s.parameters(), "lr": 1e-4},
            {"params": megnet_model.output_proj.parameters(), "lr": 1e-3},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        patience = 30
        base_lr = 1e-3

    else:  # "full"
        print("Strategy: 'Full Fine-Tuning' -> All layers are defrosted, single Learning Rate")
        for param in megnet_model.parameters():
            param.requires_grad = True
            
        optimizer = torch.optim.AdamW(
            megnet_model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-4
        )
        patience = 20
        base_lr = 1e-4

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    lightning_model = ModelLightningModule(
        model=megnet_model, 
        data_mean=data_mean, 
        data_std=data_std, 
        loss="huber_loss",
        optimizer=optimizer,
        scheduler=scheduler,
        lr=base_lr
    )
    
    early_stopping = EarlyStopping(monitor="val_MAE", patience=patience, mode="min")
    logger_name = f'MEGNet_{args.strategy}_train_fraction_{fraction_to_name(args.train_fraction)}'
    logger = CSVLogger(project_path('logs'), name=logger_name)
    
    trainer = L.Trainer(
        max_epochs=args.epochs,
        log_every_n_steps=5, 
        accelerator='auto', 
        devices="auto", 
        logger=logger,
        callbacks=[early_stopping],
        deterministic=True,
        enable_progress_bar=True
    )
    
    print("\nStarting Training...\n")
    trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    normalizer_class = type(loaded_model.transformer)
    new_normalizer = normalizer_class(mean=data_mean, std=data_std)
    
    final_model = TransformedTargetModel(model=megnet_model, target_transformer=new_normalizer)
    
    finetuned_model_path = project_path('artifacts', 'models', 'finetuned', logger_name)
    finetuned_model_path.mkdir(parents=True, exist_ok=True)
    final_model.save(finetuned_model_path)
    print(f"\nModel saved to {finetuned_model_path}")

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
    model_name: str,
    train_fraction: float,
    seed: int,
) -> str:
    prediction_rel_path = f"results/predictions/{model_name}_{fraction_to_name(train_fraction)}_seed{seed}.csv"
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


def run_fallback(args: argparse.Namespace) -> None:
    """Fallback-strategy which runs if MatGL is not installed."""
    print(f"\n[FALLBACK MODE] MatGL not found. Running Baseline (Mean Predictor).")
    train_df, val_df, test_df = load_data(args.train_fraction)

    mean_target = train_df["target"].mean()
    y_val_pred = np.full(len(val_df), mean_target)
    y_test_pred = np.full(len(test_df), mean_target)

    y_true_test = test_df["target"].to_numpy()
    val_metrics = compute_regression_metrics(val_df["target"].to_numpy(), y_val_pred)
    test_metrics = compute_regression_metrics(y_true_test, y_test_pred)

    fallback_model_name = f"{BASE_MODEL_NAME}_{args.strategy}_fallback"
    predictions_path = save_predictions(
        sample_ids=test_df["sample_id"],
        y_true=y_true_test,
        y_pred=y_test_pred,
        model_name=fallback_model_name,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    result_row = make_result_row(
        model=fallback_model_name,
        model_family=MODEL_FAMILY,
        train_fraction=args.train_fraction,
        seed=args.seed,
        mae=test_metrics["mae"],
        rmse=test_metrics["rmse"],
        r2=test_metrics["r2"],
        predictions_path=predictions_path,
        notes="Fallback baseline: predicted mean train target due to MatGL missing."
    )
    save_result_row(result_row)
    print(f"Fallback metrics | Test MAE: {test_metrics['mae']:.6f}, R2: {test_metrics['r2']:.6f}")


def main() -> None:
    args = parse_args()

    if matgl is None or L is None:
        run_fallback(args)
        return
    L.seed_everything(args.seed, workers=True)

    train_df, val_df, test_df = load_data(args.train_fraction)
    train_dataset, val_dataset, test_dataset = prepare_datasets(train_df, val_df, test_df)
    final_model = train_model(args, train_dataset, val_dataset, test_dataset)

    print("\nRunning inference...")
    y_val_pred = evaluate_model(final_model, val_df, args.device)
    y_test_pred = evaluate_model(final_model, test_df, args.device)
    
    y_true_test = test_df["target"].to_numpy()
    val_metrics = compute_regression_metrics(val_df["target"].to_numpy(), y_val_pred)
    test_metrics = compute_regression_metrics(y_true_test, y_test_pred)

    current_model_name = f"{BASE_MODEL_NAME}_{args.strategy}"
    notes = f"Strategy: {args.strategy}. Early Stopping active."

    predictions_path = save_predictions(
        sample_ids=test_df["sample_id"],
        y_true=y_true_test,
        y_pred=y_test_pred,
        model_name=current_model_name,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    
    result_row = make_result_row(
        model=current_model_name,
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
    print(f"Model: {current_model_name}")
    print(f"Validation MAE: {val_metrics['mae']:.6f}")
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"Test R2: {test_metrics['r2']:.6f}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved result row to: {result_path}")


if __name__ == "__main__":
    main()
