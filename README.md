<p align="center"><img src="assets/ImpactSeg.png" alt="IMPACT-Seg logo" width="240"></p>
<h1 align="center">IMPACT-Seg</h1>
<p align="center">Dense anatomical segmentation across CT, CBCT and MRI, with one model and one label set.</p>

IMPACT-Seg segments **131 anatomical structures** (organs, vessels, muscles, vertebrae and ribs by level, head-and-neck structures) on **CT, cone-beam CT and MRI** with the same network and the same labels. It was built for radiotherapy imaging, where the same patient is seen on a planning CT, daily CBCTs and MRI. The models are trained with [KonfAI](https://github.com/fideus-labs/KonfAI) and run with a plain `konfai PREDICTION`.

> **Status:** private preview. The manuscript is in preparation. Research use only, not a clinically validated device.

## Models

Two networks are released, each as five folds trained on different patient splits. Both predict the same 131 structures and are equally supported:

| Network | Input / patch | Normalization | Size / fold | Whole-body CT (A6000, fresh process) |
|---|---|---|---:|---|
| **2.5D** residual-encoder U-Net | 5 adjacent axial slices, 288 × 288 | BatchNorm | 411 MB | ≈ 28 s, 5.5 GB GPU |
| **3D** residual-encoder U-Net | 128 × 128 × 128 | InstanceNorm | 1.2 GB | ≈ 32 s, 4.3 GB GPU |

Weights, labels and ready-to-run configurations are hosted with the dataset on Hugging Face: [`HCedric/IMPACTSeg`](https://huggingface.co/datasets/HCedric/IMPACTSeg) (`models/`).

## Quick start: segment a volume

```bash
# 1. Environment: KonfAI with NIfTI support (reference: Python 3.11, KonfAI 1.8.3, PyTorch 2.14)
pip install "konfai[itk]==1.8.3" huggingface_hub

# 2. Weights (here the 3D network; use models/2_5D/** for the 2.5D network)
python -c "from huggingface_hub import snapshot_download; snapshot_download('HCedric/IMPACTSeg', repo_type='dataset', local_dir='hub', allow_patterns=['models/*', 'models/3D/**'])"

# 3. Volumes to segment (CT, CBCT or MRI, any orientation), in KonfAI's layout: Dataset/<case>/image.nii.gz
# 4. From the root of this repository (KonfAI imports impactseg_v2 from the working directory):
konfai PREDICTION --config configs/predict_3d.yml --models hub/models/3D/fold1/fold1.pt --gpu 0

# Five-fold ensemble (softmax average), recommended on new data:
konfai PREDICTION --config configs/predict_3d.yml --models hub/models/3D/fold{1,2,3,4,5}/fold*.pt --gpu 0
```

Label maps are written to `Predictions/IMPACTSeg_3D/Segmentations/<case>/mask_fine.nii.gz` in the geometry of the input image; label names are in [`configs/labels.json`](configs/labels.json). The configuration contains the whole pipeline (reorientation, z-score normalization, resampling to 1.5 × 1.5 × 2 mm, sliding window with confidence-based patch fusion, return to the native grid).

## Training

The five splits used for the released folds are in [`splits/`](splits). The dataset is on Hugging Face; [`tools/prepare_hub_dataset.py`](tools/prepare_hub_dataset.py) lays a download out in the training layout and maps the split case IDs to the Hub cases ([`splits/hub_crosswalk.csv`](splits/hub_crosswalk.csv)).

```bash
pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu126   # or scripts/bootstrap_pyenv.sh
python tools/prepare_hub_dataset.py --hub-root hub --output-root data          # raw layout + Hub-restricted splits
mkdir -p data/preprocessed
python tools/preprocess_training_dataset.py --source-root data/raw --destination-root data/preprocessed \
    --splits-root data/splits --overwrite                                       # z-score + 1.5 x 1.5 x 2 mm, once
python tools/run_training.py --config configs/train_3d.yml --dataset-root data/preprocessed \
    --splits-root data/splits --split-name fold1 --run-root runs/fold1_3d --max-iterations 2000000 --gpu 0
```

Recipe: SGD (lr 0.01, Nesterov momentum 0.99), polynomial decay over 2 M updates, cross-entropy + Dice with deep supervision, 70 % foreground-centred patches. The five folds are modality-balanced: a test case of one fold can be a training case of another, so evaluate internal cases with the fold whose test partition contains them.

## Repository

```text
impactseg_v2/      network, losses, sampling, augmentations and KonfAI output classes
configs/           training recipes (train_2p5d.yml, train_3d.yml), inference configs (predict_*.yml), labels.json
splits/            five explicit train / validation / test splits, crosswalk to the Hugging Face cases
tools/             dataset preparation, preprocessing, split validation, training launcher
```

## Data, licenses and citation

The code is released under the [Apache License 2.0](LICENSE). The training data combine public datasets with their own licenses, some of them non-commercial; see the [dataset card](https://huggingface.co/datasets/HCedric/IMPACTSeg). The reference masks were produced by automatic labelling and completion, followed by manual review and correction: they are not exhaustive manual ground truth.

A citation will be added with the manuscript; until then, see [CITATION.cff](CITATION.cff).
