# Spiking Neural Network Based Fall Detection Using Wearable Sensors

Code for the ISCAS 2027 submission *"Spiking Neural Network Based Fall Detection Using Wearable Sensors"*
(S. Shorey, H. Rahimian Kalatehbali, R. Genov, A. Amirsoleimani).

A three-pathway spiking convolutional neural network (SCNN) classifies 4 s windows of SisFall
accelerometer + gyroscope data as **fall** or **activity of daily living (ADL)**. Each window is
encoded three ways (Segmented TTFS, learnable Level-Crossing, and TTFS on FS1 statistical
features); each spike train goes through its own STDP-pretrained spiking convolutional pathway,
and the fused features feed a spiking fully connected classifier.

## Layout

```
configs/
  default.yaml      method as described in the paper (use for reported results)
  legacy.yaml       reproduces the original scripts' data pipeline, for comparison only
src/snn_fall/
  config.py         typed config, YAML loading, --set overrides
  data.py           SisFall parsing, high-pass filter, windowing, subject-wise split, z-scoring
  encoding.py       Segmented TTFS, FS1 TTFS, learnable Level-Crossing encoders
  model.py          SCNN: per-encoding spiking conv pathways, heads, classifier
  stdp.py           unsupervised STDP pretraining of the conv layers
  train.py          BPTT training loop (AdamW, ReduceLROnPlateau, early stopping on val F1)
  evaluate.py       metrics, threshold sweep / selection
  plotting.py       training curves, threshold sweep, confusion matrix
  pipeline.py       end-to-end experiment and evaluation of a run
scripts/
  preprocess.py     build data/processed/*.npz
  train.py          STDP -> supervised training -> val/test evaluation
  evaluate.py       re-evaluate an existing run directory
```

## Setup

```bash
pip install -r requirements.txt
```

Download SisFall and place it so that recordings live at `data/SisFall_dataset/SA01/D01_SA01_R01.txt`, etc.
Run all commands from the repository root.

## Usage

```bash
python scripts/preprocess.py
```

```bash
python scripts/train.py --name baseline
```

`train.py` builds the dataset cache automatically if it is missing, so `preprocess.py` is optional.
Each combination of data settings gets its own cache file (`data/processed/sisfall_4s-<hash>.npz`). Any config value can be overridden:

```bash
python scripts/train.py --set seed=1 --set train.epochs=100 --set stdp.enabled=false --name no_stdp
```

### Changing settings

Every tunable value lives in [configs/default.yaml](configs/default.yaml), with a comment on each line.
Either edit that file, copy it (a new file can start with `base: default.yaml` and list only what
it changes, like [configs/legacy.yaml](configs/legacy.yaml)), or override on the command line with
`--set section.key=value`. Lists need quotes: `--set "train.fall_shift_s=[0.5, 2.5]"`.

Settings are checked before anything runs, and an inconsistent combination stops with a message
naming what to change. Changing any `data.*` value builds a new dataset cache automatically;
earlier caches are kept, so switching back is instant.

| To change | Edit | Keep consistent |
|---|---|---|
| Window length | `data.window_s` | `data.fall_pre_s`, `train.fall_shift_s` and `eval.fall_offsets_s` must be < `window_s` |
| Where the impact sits in aligned fall windows | `data.fall_pre_s` | |
| Random impact range during training | `train.fall_shift_s` (`null` = aligned only) | |
| Impact positions used for val/test | `eval.fall_offsets_s` (`null` = aligned only) | |
| Sensors | `data.sensor_cols` (0-2 ADXL345 acc, 3-5 gyro, 6-8 MMA8451Q acc) | |
| Subjects | `data.subject_prefixes` (`[SA]`, `[SA, SE]`) | |
| Filter | `data.highpass_hz`, `data.highpass_order`, `data.highpass_causal` | |
| SNN time steps | `encoding.t_max` | |
| Network size | `model.conv_channels`, `model.head_dim`, `model.hidden_dim` | `conv_channels` divisible by `model.gn_groups` |
| STDP on/off | `stdp.enabled` | |
| Training length | `train.epochs`, `train.patience` | |

Each run writes to `runs/<timestamp>_<name>/`:

| File | Contents |
|---|---|
| `config.yaml` | exact config used |
| `stdp_pretrained.pt` | weights after STDP pretraining |
| `best.pt` | checkpoint with the best validation F1 |
| `train_log.csv`, `training_curves.png` | per-epoch train/val metrics |
| `threshold_sweep_val.csv/.png` | decision-threshold sweep on the validation split |
| `metrics.json`, `confusion_test.png` | val and **test** metrics at the threshold chosen on val: `test` (falls at every `eval.fall_offsets_s` position) and `test_aligned` (impact at `data.fall_pre_s`) |
| `recall_by_offset_test.png` | test fall recall vs. where the impact sits in the window |

## Method summary

| Stage | Setting |
|---|---|
| Subjects | SisFall young adults (SA01–SA23); split 70/15/15 **by subject** |
| Channels | ADXL345 accelerometer (x, y, z) + ITG3200 gyroscope (x, y, z) |
| Filter | 4th-order causal (forward-only) Butterworth high-pass, 0.4 Hz; `data.highpass_causal=false` for zero-phase |
| Windows | 4 s at 200 Hz; ADLs non-overlapping; training falls re-cut each epoch with the impact at a random time in 1–3 s |
| Features | raw window + FS1 (mean, std, max, min per channel); z-scored with train statistics |
| Encoding | 20 time steps; Segmented TTFS (4 × 1 s segments, segment max), Level-Crossing (learnable per-channel thresholds), FS1 TTFS |
| Pathways | 2 × (Conv1d k=5/3 → GroupNorm → LIF), 32 channels; Linear 32→64 + LIF per pathway |
| Classifier | 192 → 128 → 2, LIF (β = 0.9, fast-sigmoid surrogate); logits = final output membrane |
| Pretraining | 5 epochs of STDP on both conv layers of every pathway |
| Training | AdamW (3e-4), CE on membrane with label smoothing 0.05, class-balanced sampling, Gaussian noise (σ = 0.01) augmentation, ReduceLROnPlateau on val F1, early stopping (patience 25) |
| Evaluation | val/test falls evaluated with the impact at 1.0, 1.5, 2.0, 2.5 and 3.0 s; threshold maximising val F1, applied unchanged to the test subjects |

The model has 49,736 trainable parameters.
