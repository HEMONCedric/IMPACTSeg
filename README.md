<p align="center"><img src="assets/ImpactSeg.png" alt="IMPACT-Seg" width="220"></p>

<h1 align="center">IMPACT-Seg</h1>

<p align="center"><b>One model. One label set. CT, CBCT and MRI.</b><br>
Dense whole-body anatomy in seconds, built for radiotherapy imaging.</p>

<p align="center">
  <a href="https://huggingface.co/datasets/HCedric/IMPACTSeg"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20dataset%20%26%20models-HCedric%2FIMPACTSeg-ffcc4d" alt="Hugging Face"></a>
  <a href="https://github.com/fideus-labs/KonfAI"><img src="https://img.shields.io/badge/built%20with-KonfAI-3b82f6" alt="KonfAI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/code-Apache--2.0-22c55e" alt="Apache-2.0"></a>
  <img src="https://img.shields.io/badge/paper-in%20preparation-9ca3af" alt="Paper in preparation">
</p>

<table>
<tr>
<td width="38%" align="center"><img src="assets/anatomy_3d.png" alt="126 structures segmented on a whole-body CT"></td>
<td>

IMPACT-Seg segments **126 anatomical structures** with the same network on **planning CT, cone-beam CT and MRI**:
abdominal and thoracic organs, the five lung lobes, heart chambers and great vessels, **every vertebra and rib by
level**, pelvic and limb bones, trunk muscles, and head-and-neck organs at risk.

- **One model for three modalities.** No modality flag, no per-modality fine-tuning, the same 126 structures everywhere.
- **Robust on CBCT.** From CT to the paired CBCT, the Dice drops by about **0.02**, against **0.15–0.18** for CT-only
  whole-body tools; on real CBCT it reaches **0.84**, where the best competitor reaches 0.67.
- **Consistent across modalities.** The same patient gets the same anatomy on CT, CBCT and MRI, which is what
  adaptive radiotherapy needs.
- **Fast.** A whole-body CT in about 8 s on one GPU with less than 6 GB of memory, all 126 structures at once: 10 to 70 times faster than the compared whole-body tools.
- **A representation, not only a segmentation.** Used as the similarity of a registration, it ranks **first on both
  multimodal Learn2Reg tasks** (MRI–CT and CBCT–CT, [below](#registration-learn2reg)).
- **Open.** Data, weights and code: two networks (2.5D and 3D) × five folds, run with a plain `konfai PREDICTION`.

<sub>Left: 3D network on a held-out whole-body CT (anterior and oblique views; lungs and large muscles translucent).</sub>

</td>
</tr>
</table>

<p align="center"><img src="assets/multimodal.png" alt="IMPACT-Seg 3D on CT, simulated and real CBCT, and external whole-body MRI" width="100%"></p>
<p align="center"><sub>Out-of-fold predictions of the 3D network: whole-body CT, CBCT simulated from a whole-body CT, real thoracic CBCT,
and a whole-body MRI from an external dataset (five-fold ensemble).</sub></p>

## Quick start

```bash
# 1. KonfAI with NIfTI support (reference: Python 3.11, KonfAI 1.8.3, PyTorch 2.14)
pip install "konfai[itk]==1.8.3" huggingface_hub

# 2. Weights from Hugging Face (3D network; use models/2_5D/** for the 2.5D one)
python -c "from huggingface_hub import snapshot_download; snapshot_download('HCedric/IMPACTSeg', repo_type='dataset', local_dir='hub', allow_patterns=['models/*', 'models/3D/**'])"

# 3. Your volumes (CT, CBCT or MRI, any orientation): Dataset/<case>/image.nii.gz
# 4. From the root of this repository:
konfai PREDICTION --config configs/predict_3d.yml --models hub/models/3D/fold1/fold1.pt --gpu 0

#    or the five-fold ensemble, recommended on new data:
konfai PREDICTION --config configs/predict_3d.yml --models hub/models/3D/fold{1,2,3,4,5}/fold*.pt --gpu 0

#    coarse body regions and tissues (download models/coarse/** instead)
konfai PREDICTION --config configs/predict_coarse.yml --models hub/models/coarse/coarse.pt --gpu 0
```

You get `Predictions/IMPACTSeg_3D/Segmentations/<case>/mask_fine.nii.gz`, a label map on the grid of your image;
names are in [`configs/labels.json`](configs/labels.json). Everything happens inside the configuration:
reorientation, z-score normalization (no HU window, any modality), resampling to 1.5 × 1.5 × 2 mm, sliding-window
inference with confidence-based patch fusion, and return to the native grid.

## Models

| | Input | Patch | Normalization | Checkpoint | Whole-body CT (RTX A6000, per volume) |
|---|---|---|---|---:|---|
| **IMPACT-Seg 2.5D** | 5 adjacent axial slices | 288 × 288 | BatchNorm | 411 MB | ≈ 8 s · 5.7 GB |
| **IMPACT-Seg 3D** | volume | 128 × 128 × 128 | InstanceNorm | 1.2 GB | ≈ 15 s · 4.4 GB |
| **Coarse 3D** | volume at 3 mm | 128 × 128 × 128 | InstanceNorm | 57 MB | < 10 s · < 1 GB (RTX 8000) |

Both are six-stage residual-encoder U-Nets with 132 outputs (background + the 131 labels of the dataset; the cochleae,
optic chiasm and optic nerves, a few tens of voxels at the training resolution, were not learned and are never output,
so the models provide 126 structures), trained for 2 M
iterations on the IMPACT-Seg dataset (CT, real and simulated CBCT, MRI), as five folds on different patient splits. The 3D network is
the more accurate of the two; the 2.5D network is lighter. Any single fold is a complete model; the five-fold
ensemble is the most accurate option on new data.

The **coarse** network is a separate, light model (7 M parameters) for the 11 body-region and tissue classes of
`mask_coarse` (subcutaneous tissue, muscle, abdominal and thoracic cavities, bones, glands, pericardium, breast implant,
mediastinum, spinal cord, brain), trained on one split; on its 64 test volumes the Dice is 0.89 on CT, 0.86 on CBCT and
0.81 on MRI (indicative, partly pseudo-labelled references). It writes `Predictions/IMPACTSeg_coarse/Segmentations/<case>/mask_coarse.nii.gz`.

## Registration (Learn2Reg)

Because IMPACT-Seg gives the same dense anatomy on every modality, its predictions can serve as the similarity of a
multimodal registration: the IMPACT-Seg 3D segmentations of the two images drive an instance optimisation
(ConvexAdam or Elastix initialisation, then Adam refinement of a soft Dice between the predicted structures). On the official
Learn2Reg test leaderboards:

| Task | Entry | Dice ↑ | HD95 (mm) ↓ |
|---|---|---:|---:|
| [**AbdomenMRCT**](https://learn2reg.grand-challenge.org/evaluation/test/leaderboard/) · MRI–CT | 🥇 **IMPACT-Seg** | **0.917** | **2.24** |
| | 2nd entry | 0.915 | 2.73 |
| | corrField (organisers) | 0.876 | 4.40 |
| [**ThoraxCBCT**](https://learn2reg.grand-challenge.org/evaluation/thoraxcbct/leaderboard/) · CBCT–CT | 🥇 **IMPACT-Seg** | **0.762** | **9.10** |
| | 2nd entry | 0.732 | 26.26 |
| | 3rd entry | 0.732 | 11.25 |

Both entries rank first, with the lowest HD95 of all submissions (tied on AbdomenMRCT); compared with our previous intensity-based
submissions (IMPACT features in Elastix), the Dice increased by 0.04 and 0.11.

## Training

The five splits of the released folds are in [`splits/`](splits). [`tools/prepare_hub_dataset.py`](tools/prepare_hub_dataset.py)
turns a download of the Hugging Face dataset into the training layout, using [`splits/hub_crosswalk.csv`](splits/hub_crosswalk.csv)
to map the split case IDs to the Hub folders.

```bash
pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu126   # or scripts/bootstrap_pyenv.sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('HCedric/IMPACTSeg', repo_type='dataset', local_dir='hub', allow_patterns=['CT/**', 'CBCT/**', 'MRI/**'])"
python tools/prepare_hub_dataset.py --hub-root hub --output-root data          # training layout + Hub splits
mkdir -p data/preprocessed
python tools/preprocess_training_dataset.py --source-root data/raw --destination-root data/preprocessed \
    --splits-root data/splits --overwrite                                       # z-score + 1.5 x 1.5 x 2 mm, once
python tools/run_training.py --config configs/train_3d.yml --dataset-root data/preprocessed \
    --splits-root data/splits --split-name fold1 --run-root runs/fold1_3d --max-iterations 2000000 --gpu 0
```

Recipe: SGD (lr 0.01, Nesterov momentum 0.99), polynomial decay over 2 M iterations, cross-entropy + Dice with deep
supervision, 70 % foreground-centred patches, validation every 100 k iterations. The folds are modality-balanced: a
test case of one fold can be a training case of another, so evaluate internal cases with the fold whose test
partition contains them.

## Repository

```text
impactseg_v2/   network, losses, sampling, augmentations and KonfAI output classes
configs/        train_2p5d.yml, train_3d.yml, predict_2p5d.yml, predict_3d.yml, predict_coarse.yml, labels.json
splits/         five train / validation / test splits, crosswalk to the Hugging Face cases
tools/          dataset preparation, preprocessing, split validation, training launcher
```

## Data, license and citation

The code is released under the [Apache License 2.0](LICENSE). The dataset combines public datasets with their own
licenses, several of them non-commercial: see the [dataset card](https://huggingface.co/datasets/HCedric/IMPACTSeg).
Its reference masks come from automatic labelling, completion and manual review and correction; they are curated
references, not exhaustive manual delineations. Research use only, not a clinically validated device.

The manuscript is in preparation (C. Hémon, V. Boussot, C. Kurz, G. Landry); until then, please cite this
repository ([CITATION.cff](CITATION.cff)).
