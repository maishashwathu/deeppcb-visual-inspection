# DeepPCB data card

| Field | Value |
|---|---|
| Source | tangsanli5201/DeepPCB on GitHub |
| Pinned revision | `08e98c4db5922613fb97176eb3d6497d48260cb1` |
| Unit | Reference/inspected grayscale PCB image pair |
| Image dimensions | 640 × 640 pixels |
| Official partitions | 1,000 training/validation pairs; 500 test pairs |
| Local partitions | 800 train; 200 validation; 500 official test |
| Supervision | Axis-aligned bounding boxes and six defect classes |
| Acquisition | Industrial line-scan imagery; cropped, aligned and binarised upstream |
| Artificial content | Defects were also added manually by the dataset authors |
| Use | Research, as stated in the upstream README |
| Local location | `data/DeepPCB/PCBData/` (Git-ignored) |

## Validation performed

The notebook checks all split-file paths, reads all 1,500 image pairs, checks image dimensions, validates label IDs and boxes, verifies unique image IDs and checks for exact inspected-image duplicates across the official partitions. These checks do not establish absence of near duplicates or related circuit layouts.

## Leakage controls

Partition images before deriving features/crops. Keep every candidate and crop from one image in its partition. Stratify the local validation partition by the source directory group. Select proposal settings and confidence cutoff on validation only. Keep model structure fixed. Evaluate the final detector on all official test images after settings are frozen; see the development-exposure qualification below.

The official split and local validation may share source-board groups. Results establish performance under this dataset protocol, **not generalisation to unseen board layouts**. See `artifacts/split_manifest.json` for the actual image IDs.

### Development exposure

An initial baseline was evaluated on the official test set before a morphology revision. A training-image difference visual and low validation proposal recall motivated adding morphological opening; its settings were selected on validation. Initial metrics remain in `artifacts/initial_baseline_results.json`. The final test result is a reused benchmark evaluation, not a pristine one-shot estimate. A new independent dataset is required for that stronger claim.

## Geometric stress test

Thirty official test images are selected deterministically using seed 42. Apply the same mild and moderate transformations to each selected image. Evaluate all methods on the same subset and keep registration failures in the denominator. This is a separate synthetic robustness experiment, not a substitute for the full clean-test evaluation.

## Publication

Publish code, measured aggregate results, split identifiers and credited research visualisations. Do not commit the raw dataset, downloaded archive, upstream Git metadata, personal course documents or locally trained binary weights. Obtain suitable rights and new validation before any non-research use.
