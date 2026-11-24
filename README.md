<p align="center"><img src="assets/ImpactSeg.png" alt="IMPACT-Seg" width="220"></p>

<h1 align="center">IMPACT-Seg</h1>

<p align="center"><b>One model. One label set. CT, CBCT and MRI.</b><br>
Dense whole-body anatomy in seconds, built for radiotherapy imaging.</p>

<p align="center">
  <a href="https://huggingface.co/HCedric/IMPACTSeg"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20models-HCedric%2FIMPACTSeg-ffcc4d" alt="Hugging Face models"></a>
  <a href="https://huggingface.co/datasets/HCedric/IMPACTSeg"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20dataset-HCedric%2FIMPACTSeg-ffcc4d" alt="Hugging Face dataset"></a>
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
- **Robust on CBCT.** On real CBCT the Dice reaches **0.84**, where the best compared method reaches 0.67, and
  **0.58** on independent clinical pancreatic CBCT (0.49 for TotalSegmentator); the predictions on the planning CT and
  on the registered real CBCT of the same patient agree at **0.88** Dice, against 0.49–0.69 for the compared methods.
- **Consistent across modalities.** The same patient gets the same anatomy on CT, CBCT and MRI, which is what
  adaptive radiotherapy needs.
- **Fast.** A whole-body CT in 8.0 s (2.5D) or 14.8 s (3D) on one GPU with less than 6 GB of memory, all 126 structures at once: 10 to 70 times faster than the compared whole-body tools.
- **A representation, not only a segmentation.** Used as the similarity of a registration, its last-decoder features
  rank **first on both multimodal Learn2Reg tasks** (MRI–CT and CBCT–CT, [below](#registration-learn2reg)).
- **Open.** Data, weights and code: two networks (2.5D and 3D) × five folds, run as KonfAI apps.

<sub>Left: 3D network on a held-out whole-body CT (anterior and oblique views; lungs and large muscles translucent).</sub>

</td>
</tr>
</table>

<p align="center"><img src="assets/multimodal.png" alt="IMPACT-Seg 3D on CT, simulated and real CBCT, and external whole-body MRI" width="100%"></p>
<p align="center"><sub>Out-of-fold predictions of the 3D network: whole-body CT, CBCT simulated from a whole-body CT, real thoracic CBCT,
and a whole-body MRI from an external dataset (five-fold ensemble).</sub></p>

<p align="center"><a href="https://huggingface.co/HCedric/IMPACTSeg/resolve/main/impactseg.mp4"><img src="assets/video_poster.jpg" alt="IMPACT-Seg video (55 s)" width="80%"></a></p>
<p align="center"><sub>▶ IMPACT-Seg in 55 s (click to play): one model on CT, CBCT and MRI, 126 structures, and its features used
for multimodal registration. Images: MTBD / Multimodal-HC (Hinge et al., <i>Scientific Data</i> 2026, CC BY 4.0) and Learn2Reg AbdomenMRCT.</sub></p>

## Quick start

The three models are packaged as [KonfAI apps](https://huggingface.co/HCedric/IMPACTSeg): no configuration to
write, and the weights download on first use. They need KonfAI 1.9.0 or later.

```bash
pip install impact_seg_konfai
impact-seg-konfai segment fine-3d -i image.nii.gz -o output/ --gpu 0     # or fine-2.5d, coarse
```

- `--ensemble 1` runs one fold instead of five: five times faster, and one checkpoint to download.
- The same apps with the generic runner: `konfai-apps infer VBoussot/ImpactSeg:fine-3d -i image.nii.gz -o output/`.
- In 3D Slicer: the [IMPACT-Seg extension](https://github.com/vboussot/SlicerImpactSeg).

You get a label map on the grid of your image; names are in [`configs/labels.json`](configs/labels.json). The app
does the reorientation, z-score normalization (no HU window, any modality), resampling to 1.5 × 1.5 × 2 mm,
sliding-window inference with confidence-based patch fusion, and the return to the native grid. Its labels are those
behind the results reported here: the same voxels on a CT or CBCT in HU with the 3D networks, and at most 0.02 % of
the foreground voxels apart on a float volume.

## Models

| | Input | Patch | Normalization | Checkpoint | Whole-body CT (RTX A6000, per volume) |
|---|---|---|---|---:|---|
| **IMPACT-Seg 2.5D** | 5 adjacent axial slices | 288 × 288 | BatchNorm | 411 MB | 8.0 s · 5.7 GB |
| **IMPACT-Seg 3D** | volume | 128 × 128 × 128 | InstanceNorm | 1.2 GB | 14.8 s · 4.4 GB |
| **Coarse 3D** | volume at 3 mm | 128 × 128 × 128 | InstanceNorm | 57 MB | < 3 s · < 1 GB |

Both are six-stage residual-encoder U-Nets with 132 outputs (background + the 131 labels of the dataset; the cochleae,
optic chiasm and optic nerves, a few tens of voxels at the training resolution, were not learned and are never output,
so the models provide 126 structures), trained for 2 M
iterations on the IMPACT-Seg dataset (CT, real and simulated CBCT, MRI), as five folds on different patient splits. The 3D network is
the more accurate of the two; the 2.5D network is lighter. Any single fold is a complete model; the five-fold
ensemble is the most accurate option on new data.

The **coarse** network is a separate, light model (7 M parameters) for the 11 body-region and tissue classes of
`mask_coarse` (subcutaneous tissue, muscle, abdominal and thoracic cavities, bones, glands, pericardium, breast implant,
mediastinum, spinal cord, brain), trained on one split; on its 64 test volumes the Dice is 0.89 on CT, 0.86 on CBCT and
0.81 on MRI (indicative, partly pseudo-labelled references). It runs as the `coarse` app.

## Registration (Learn2Reg)

Because IMPACT-Seg represents the same anatomy on every modality, the features of the last decoder layer of the 3D
network can serve as the similarity of a multimodal registration:

- **AbdomenMRCT (MRI–CT)**: ConvexAdam initialised from the predicted liver, spleen and kidneys, then Adam
  refinement of the L1 distance between the last-decoder features of the fixed and warped moving images, with
  diffusion regularisation (160 iterations, then 180 on a control grid of two-voxel spacing); the segmentations are
  used for the initialisation only.
- **ThoraxCBCT (CBCT–CT)**: our previous Elastix-IMPACT pipeline for the initialisation, then refinement combining the
  last-decoder feature similarity, MIND-SSC descriptors and diffusion regularisation (60 iterations at about 4 mm,
  100 at 2 mm).

On the official Learn2Reg test leaderboards:

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
impactseg_v2/   network, losses, sampling and augmentations
configs/        train_2p5d.yml, train_3d.yml, labels.json
splits/         five train / validation / test splits, crosswalk to the Hugging Face cases
tools/          dataset preparation, preprocessing, split validation, training launcher
```

## Data, license and citation

The code is released under the [Apache License 2.0](LICENSE). The dataset combines public datasets with their own
licenses, several of them non-commercial: see the [dataset card](https://huggingface.co/datasets/HCedric/IMPACTSeg).
Its reference masks come from automatic labelling, completion and manual review and correction; they are curated
references, not exhaustive manual delineations. Research use only, not a clinically validated device.

The manuscript is in preparation (C. Hémon, V. Boussot, M. McCormick, L. Zimmermann, C. Kurz, G. Landry); until then, please cite this
repository ([CITATION.cff](CITATION.cff)).
