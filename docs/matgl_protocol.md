# MatGL Final Protocol

The final shortened protocol runs MatGL / MEGNet as a compute-light transfer-learning baseline.

The experiment config `configs/experiments/matgl.yaml` fixes the pretrained model to
`MEGNet-Eform-MP-2018.6.1` and uses `strategy: frozen`. In this mode the pretrained
MEGNet graph layers are frozen and only the output projection is trainable. This keeps
the MatGL run reproducible and suitable for the final reduced budget suite.

`differential` and `full` fine-tuning are supported by the code for exploratory work,
but they are not part of the final shortened protocol because they require heavier
optimization of the pretrained graph model.

Run a smoke experiment with:

```bash
perovskite-screening run \
  --config configs/experiments/matgl.yaml \
  --split-strategy random_iid \
  --budget B500 \
  --seed 42
```

The first run may need internet access to download the pretrained MatGL model through
MatGL's model loader. Later runs can use the local MatGL cache if the model is already
available.
