# Flashmind PDM on TIME

Evaluate [Femto, Pico, or Nano](https://www.flashmind.com/models/pdm) from
[Flashmind UAE](https://www.flashmind.com/) through the
[flashpdm client](https://www.flashmind.com/docs/python) and
[public API](https://www.flashmind.com/docs/api). Request an evaluator key from
[Flashmind](https://www.flashmind.com/contact); no local GPU is required.

```bash
export FLASHMIND_API_KEY=your_evaluator_key
bash scripts/run_flashpdm.sh --model pdm-pico
```

Use `pdm-femto` or `pdm-nano` for the other models. The runner downloads TIME to
the Hugging Face cache, or uses the existing directory named by `TIME_DATASET`.
Use `--concurrency` to match your account's allowance (default: 32).

For a small check covering univariate and multivariate forecasting:

```bash
bash scripts/run_flashpdm.sh --model pdm-pico \
  --dataset Auto_Production_SF/M Uncertainty_1M/M --terms short --concurrency 4
```

`flashpdm` batches evaluation windows and saves validated forecasts locally.
Transient batch failures get at most three attempts. A lost response can mean
an attempt already ran, so recovery can consume additional API allowance.
TIME scores them using its official `save_window_predictions` function. Completed
tasks are resumed only when their settings and file checksums match.

Each task under `output/results/PDM-Pico/` contains the official `predictions.npz`,
`metrics.npz`, and `config.json`. Only these three files per task belong in the
Hugging Face submission; `complete.json` and the separate forecast cache are local
recovery records. Use TIME's `scripts/compute_local_leaderboard.py` to calculate
normalized MASE and CRPS against Seasonal Naive.

Protocol: all targets in a series are forecast jointly; context uses the observed
history up to the model's advertised maximum (16,336 steps). Model revisions are
pinned in the runner. The API applies flip averaging. Nonnegative histories get
nonnegative quantiles; histories containing at least eight observed nonnegative
integers get integer quantiles. TIME's nine deciles are selected from the model's
21 native levels. TIME supplies no covariates. All 98 horizons fit within the
model's native 960-step forecast.
