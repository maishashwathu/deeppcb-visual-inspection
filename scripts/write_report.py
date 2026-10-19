"""Generate the README and viva guide from the measured notebook outputs."""
from pathlib import Path
import json
import csv

root = Path(__file__).resolve().parents[1]
r = json.loads((root / 'artifacts/results.json').read_text())
m = r['test_at_50']
rob = list(csv.DictReader((root / 'artifacts/robustness.csv').open()))
def pct(x): return f'{float(x)*100:.2f}%'
rob_table = '| Condition | Method | F1 @ 0.50 | Failed registrations |\n|---|---|---:|---:|\n'
for x in rob:
    rob_table += f"| {x['condition']} | {x['method']} | {pct(x['F1'])} | {x['registration_failures']} |\n"

readme = f'''# DeepPCB Visual Inspection

**An explainable, CPU-only computer vision project for circuit-board defect detection.**

Compare an inspected PCB with its defect-free reference, localise defects, classify six defect types,
and measure how ORB/SIFT registration handles camera misalignment. Built for Computer Vision for AI coursework.

[Open the executed notebook](DeepPCB_Inspection.ipynb) · [Faculty presentation guide](FACULTY_GUIDE.md) ·
[Data card](DATA_CARD.md) · [Credits](CREDITS.md)

![Inspection examples](assets/05_predictions.png)

*Source PCB imagery and annotations: Tang et al., DeepPCB. Overlays and measurements: this project.*

## What is included

- A self-contained notebook with setup, automatic data acquisition, audit, training, evaluation, visuals and inference.
- All algorithm definitions are embedded in the notebook; `src/pipeline.py` is the corresponding reusable module.
- Local download of all **1,500** official pairs; **800 / 200 / 500** train / validation / test partition.
- Nine saved figures, per-class and per-image metrics, split IDs, prediction JSON and experiment provenance.
- A controlled comparison of no alignment, ORB + RANSAC, and SIFT + RANSAC on 30 fixed test pairs.
- A faculty guide with a 10-minute presentation plan, demo procedure and viva answers.
- Correctness tests for core box matching, proposal extraction and registration failure handling.

**No PyTorch or GPU is needed.** “DeepPCB” is the dataset name, not a requirement to use a deep model.
This implementation uses OpenCV, HOG features and a scikit-learn random forest. It does not implement YOLO
or the original DeepPCB paper's neural architecture.

## Problem and research question

Small breaks, unwanted connections and extra or missing material can make a PCB defective. A clean reference
provides useful structure: differences can reveal defects without learning to rediscover the whole board.
However, subtraction is sensitive to image alignment.

**Question:** How well does a lightweight reference-based detector work, and how much does geometric alignment
recover its performance under controlled camera rotation and translation?

## Run it

Tested with **Python {r['versions']['python']}**. Python 3.12 is recommended.

```bash
git clone https://github.com/maishashwathu/deeppcb-visual-inspection.git
cd deeppcb-visual-inspection
python -m pip install -r requirements.txt
python -m jupyterlab DeepPCB_Inspection.ipynb
```

Choose the intended Python kernel and select **Restart Kernel and Run All Cells**.
An isolated environment is recommended on your own machine, but not required by the code.
The first notebook cell installs the six core libraries if needed. The notebook can also be opened in Google
Colab using a CPU runtime; a fresh session downloads the dataset again. Outputs are saved under the current
working directory. Copy them to persistent storage before ending an ephemeral session.

For unattended execution after installing dependencies:

```bash
python scripts/execute_notebook.py
python scripts/write_report.py
python -m unittest discover -s tests
```

The execution helper skips the installation-tagged cell because dependencies are already installed.
It writes actual cell outputs back into the notebook and fails on execution errors.
Allow several minutes on a CPU; downloads and hardware affect the duration. The measured feature-extraction
and model-training time in this run was **{r['training_seconds']:.1f} seconds**. The full notebook takes longer.
Allow roughly 2 GB of free disk space for dependencies, data, model and outputs, and preferably 8 GB RAM.

### Reproducibility

- Random seed: `{r['seed']}`.
- Dataset revision: `{r['dataset_revision']}`.
- OpenCV is pinned to 4.11.0.86 because the HOG API used here is incompatible with the OpenCV 5 package tested.
- `requirements.txt` gives compatible ranges; `requirements-tested.txt` records the direct versions used in this run.
- `artifacts/results.json` records versions, timings, model settings and metrics.
- `artifacts/split_manifest.json` records every image's partition.
- `artifacts/robustness_ids.json` records the stress-test subset.
- Small numerical or timing differences across platforms are expected.

## Dataset

[DeepPCB](https://github.com/tangsanli5201/DeepPCB) contains paired 640 × 640 images and labelled boxes for
**open, short, mousebite, spur, spurious copper and pin hole** defects. The authors provide 1,000 training/validation
and 500 test pairs. Some defects were manually added. Images are already aligned and binarised.

The notebook validates all paths, dimensions, class IDs and bounding boxes and checks exact tested-image
duplicates across official splits. The 200-image validation set is stratified by upstream source group.
It is image-disjoint, **not circuit-layout-disjoint**. Shared or related layouts remain a limitation.

![Dataset statistics](assets/02_dataset_statistics.png)

Raw data stays in `data/DeepPCB/` and is excluded from Git. The upstream README restricts dataset use to
research; its repository code license should not be interpreted as overriding that restriction.
See [DATA_CARD.md](DATA_CARD.md) and [CREDITS.md](CREDITS.md).

## Pipeline

```text
Reference + inspected image
        |
        +-- Optional ORB/SIFT matching -> RANSAC homography -> aligned inspection
        |                                               -> valid-border mask
        v
Binary disagreement -> dilation -> connected components -> candidate boxes
        v
HOG on reference / inspection / disagreement
 + signed-change grids + local geometry
        v
Random forest: background + 6 defect classes
        v
Validation-selected confidence cutoff -> labelled detections
```

### 1. Candidate localisation

Threshold each image at 127, compute binary disagreement, optionally apply morphological opening to remove
thin edge artifacts, then dilate and extract connected components.
Filter components by actual changed-pixel area and pad their bounds for context. The chosen validation
configuration is `{r['proposal_config']}`. This is a class-agnostic classical baseline, evaluated with
proposal recall; it does not itself classify the six defect types.

![Proposal extraction](assets/03_proposals.png)

### 2. Feature extraction and classification

Each candidate produces **{r['feature_dimension']} features**: HOG from three 32 × 32 crops, two signed-change
8 × 8 grids, and seven geometry/occupancy values. Signed changes distinguish added from removed bright pixels;
their interpretation depends on the dataset's intensity convention.

Training uses proposals labelled by annotation overlap, jittered training annotations and background crops.
The final training matrix has **{r['training_examples']} examples**. A 160-tree random forest uses balanced
bootstrap class weighting, minimum leaf size 2 and two CPU workers. Its structure is fixed; only proposal
settings and the confidence cutoff are selected on validation. The chosen cutoff is **{r['confidence_threshold']:.2f}**.
Random-forest scores are not calibrated probabilities.

### 3. Evaluation discipline

No test labels enter model fitting or inference; settings are selected on validation. Precision/recall/F1 require correct class
and one-to-one box overlap at IoU ≥ 0.50. Duplicate detections are false positives. AP uses all non-background
scores before the operating cutoff and an all-point interpolated precision envelope.

This is **not COCO mAP and not the upstream DeepPCB evaluator**. The supplementary IoU 0.33 result uses ≥0.33;
the upstream README describes >0.33. Do not compare this project's scores directly to its headline paper result.

**Development transparency:** an initial version was evaluated on the test set and had low recall.
A training-pair visual and validation proposal recall motivated adding morphological opening; its settings
were selected on validation. Initial metrics are preserved in `artifacts/initial_baseline_results.json`.
The final score is a **reused benchmark evaluation**, not an untouched one-shot generalisation estimate.
A new independent dataset would be needed to support that stronger claim.

## Measured results

All **500 official test images** were evaluated after settings were frozen.

| Metric | Result |
|---|---:|
| Precision at IoU 0.50 | {pct(m['precision'])} |
| Recall at IoU 0.50 | {pct(m['recall'])} |
| F1 at IoU 0.50 | {pct(m['F1'])} |
| All-point mAP@0.50 | {pct(m['mAP'])} |
| True / false positives / missed defects | {m['TP']} / {m['FP']} / {m['FN']} |
| Clean inference including disk I/O | {1000*r['test_inference_seconds']/500:.1f} ms/image |

![Per-class test metrics](assets/04_test_metrics.png)

### Alignment stress test

Apply clean, mild (1° rotation and +3/-2 px translation), and moderate (2° and +6/-4 px) perturbations
to the same 30 test images. Settings stay frozen. Registration maps the inspected image into template coordinates.
Failures count as empty detections, and all ground-truth defects—including border defects—remain in evaluation.

{rob_table}

![Robustness comparison](assets/06_alignment_robustness.png)

![Before and after registration](assets/07_alignment_example.png)

The experiment measures only these controlled synthetic camera motions. It does not establish robustness
to all lighting, lens, perspective, blur or acquisition changes. Registration adds processing time; consult
`artifacts/robustness.csv` for measured runtime and reprojection error.

### Failures are part of the result

The notebook selects the lowest-F1 test image for post-hoc inspection, without retuning on it. This reveals
merged candidates, class confusion or box mismatch rather than hiding them behind an aggregate score.

![Lowest-F1 test case](assets/08_failure_case.png)

![Detection confusion matrix](assets/09_confusion_matrix.png)

## Inspect your own pair

After running the notebook, use its `inspect_custom(...)` function, or the reusable module:

```python
import joblib
from src.pipeline import inspect_custom

bundle = joblib.load('artifacts/inspection_model.joblib')  # only load trusted files
result = inspect_custom(
    'reference.jpg', 'inspected.jpg', bundle['model'],
    bundle['config'], bundle['threshold'], method='SIFT'
)
print(result['status'], result['predictions'])
```

Use a matching reference, comparable preprocessing and equal image dimensions. Predictions use **reference
coordinates**, not original camera coordinates. `registration_failed` means the pair could not be reliably
aligned. An empty successful prediction list is not a guarantee that the board is defect-free.

## Repository layout

```text
DeepPCB_Inspection.ipynb       Executed, self-contained educational notebook
README.md                     Project rationale, setup and measured results
FACULTY_GUIDE.md              Presentation narrative, demo and viva preparation
CREDITS.md / DATA_CARD.md     Provenance, research restrictions and limitations
src/pipeline.py               Reusable algorithms (also embedded in notebook)
scripts/build_notebook.py     Rebuild the notebook from reviewed sections
scripts/execute_notebook.py   Execute and save outputs using the current Python
scripts/write_report.py       Generate these documents from measured results
tests/test_pipeline.py        Core correctness checks
assets/                      Nine PNG visualisations
artifacts/                   Metrics, predictions and split manifests
data/                        Downloaded dataset (ignored)
artifacts/*.joblib            Locally trained model (ignored)
```

## Course fit and limits

The project demonstrates image features/HOG (CO1), ORB/SIFT (CO2), feature matching and RANSAC (CO3),
and segmentation/connected components (part of CO4). It does **not** demonstrate temporal object tracking
or deep-learning detection (CO5). The provided policy does not explicitly require every CO in each mini-project.

Key limitations: preprocessed images, artificial defects, source-layout overlap, dependence on a matching
reference, imperfect candidate grouping, uncalibrated confidence, and a small synthetic robustness subset.
A future study should hold out entire layouts, acquire real factory examples, test illumination changes,
and compare a deep detector if useful. These are future work, not completed results.

## Credits and AI assistance

Dataset: **Sanli Tang, Fan He, Xiaolin Huang and Jie Yang**, *Online PCB Defect Detector On A New PCB Defect
Dataset* (2019), [paper](https://arxiv.org/abs/1902.06197), [dataset](https://github.com/tangsanli5201/DeepPCB).
Source images and annotations in the derived figures are credited to these authors.

Prepared for **Shashwat Jain**, with code and documentation assistance from **OpenAI Codex**.
The student should understand, rerun and explain the implementation and disclose assistance as required.
Established algorithms and open-source packages are credited in [CREDITS.md](CREDITS.md).
This repository does not claim ownership of the dataset or authorship of the underlying algorithms.
'''
(root / 'README.md').write_text(readme, encoding='utf-8')

guide = f'''# Faculty Presentation and Viva Guide

## 1. Your project in one sentence

“I built a CPU-based PCB inspection pipeline that compares a board against a clean reference, classifies
six defect types using handcrafted features and a random forest, and evaluates how image registration
recovers detection performance when the inspected image is displaced.”

Use this as a study guide. Do not claim personal implementation work you have not understood or verified.
The code and materials were developed with OpenAI Codex assistance, which is disclosed in the repository.

## 2. What is actually complete

- All 1,500 pairs downloaded from a pinned DeepPCB revision and audited.
- 800 training images, 200 validation images and all 500 official test images.
- Difference-based candidate localisation and six-class recognition with a background category.
- Validation-selected proposal parameters and confidence threshold.
- Class-aware detection evaluation, per-class results and a confusion matrix including missed detections.
- Controlled 30-image test of no alignment, ORB + RANSAC and SIFT + RANSAC.
- A reusable inference function requiring only a template and inspected image.
- Executed notebook, nine figures, reproducible metadata and public source code.

Not implemented: YOLO, PyTorch, neural-network training, real factory deployment, unseen-layout evaluation,
real-time camera integration, electrical diagnosis or temporal tracking.

## 3. Suggested 10-minute presentation

| Time | Topic | What to show |
|---|---|---|
| 0:00–0:45 | Problem and objective | Dataset pair, `assets/01_dataset_pair.png` |
| 0:45–1:45 | Dataset, six defects and split | Statistics, `assets/02_dataset_statistics.png` |
| 1:45–3:15 | Candidate localisation | `assets/03_proposals.png` |
| 3:15–4:15 | Features and classifier | Notebook section 5 |
| 4:15–5:15 | Test protocol and measured results | `assets/04_test_metrics.png` |
| 5:15–6:45 | Alignment experiment | `assets/06_alignment_robustness.png` and `07_alignment_example.png` |
| 6:45–8:00 | Live inference demo | Notebook section 9 |
| 8:00–9:15 | Failure analysis and limits | `08_failure_case.png` and `09_confusion_matrix.png` |
| 9:15–10:00 | Course mapping and contributions | Summary below and credits |

### Opening script

“A PCB defect can be very small compared with the entire board. Since a correct reference image is available,
I started with a direct comparison method. The challenge is that image displacement also creates differences.
My project separates localisation from classification and studies registration as a way to reduce that problem.”

### Explain the data

“DeepPCB provides reference and inspected images with labelled boxes. There are six defect classes.
The official 1,000-image training/validation partition is split into 800 training and 200 validation images;
the 500-image test partition is excluded from model fitting. Crop extraction happens after splitting.
The data is already aligned and binarised, and the authors added some artificial defects.”

Mention that image partitions can still share circuit layouts. Do not describe the experiment as
proof of performance on unseen board designs.

### Explain localisation using one image

1. Threshold both images into binary maps.
2. XOR highlights disagreement; optional opening suppresses thin boundary artifacts.
3. Dilation joins nearby changed pixels.
4. Connected components identify candidate regions.
5. Minimum area removes tiny noise; padding adds context.

Point out a real defect on the original image, then show its mask and candidate box. Explain that at this stage
the method only proposes locations; it does not know whether the defect is a short or a spur.

### Explain the classifier

“HOG describes local edge orientation. I compute it on the clean crop, inspected crop and difference crop.
I also retain signed-change grids and simple shape measurements. A random forest maps the resulting
{r['feature_dimension']}-dimensional vector to background or one of six classes.”

The model has 160 trees. Training used {r['training_examples']} examples derived only from training images.
Background examples allow rejection of false candidates. The validation confidence cutoff is
{r['confidence_threshold']:.2f}. A forest score is not a calibrated likelihood of physical failure.

Explain the development history: the first baseline exposed low recall. Training imagery and validation
proposal recall motivated testing morphological opening. Settings were selected on validation, but the
official test had already been observed. The final score is therefore a reused benchmark result.

### State your measured results accurately

On **all 500 official test images**, at **IoU 0.50**:

- Precision: **{pct(m['precision'])}**.
- Recall: **{pct(m['recall'])}**.
- F1: **{pct(m['F1'])}**.
- All-point interpolated mAP@0.50: **{pct(m['mAP'])}**.
- True positives: **{m['TP']}**; false positives: **{m['FP']}**; missed defects: **{m['FN']}**.

Say “detection F1” or “mAP”, not “accuracy”. Explain the IoU threshold before quoting a result.
AP uses all non-background scores; precision/recall/F1 use the selected operating cutoff.
Our evaluator differs from the official benchmark, so do not claim to outperform the paper.

### Explain the alignment experiment

“I deliberately rotate and translate the inspected image while keeping the clean reference fixed. I compare
no registration with ORB and SIFT registration on exactly the same 30 images. Descriptor matching provides
candidate correspondences. RANSAC estimates a homography while rejecting inconsistent matches.”

The homography direction is inspected image → reference. The registered image and detections are therefore
in reference coordinates. Newly exposed borders are masked. Failed registrations count as missed detections,
not as removed examples. All methods retain the same denominator.

{rob_table}

Describe the measured trend, including whether either method fails or harms clean images. Do not say alignment
always helps. This stress test uses specific synthetic motions and is much smaller than the clean test set.

## 4. Live demo checklist

### Before the presentation

1. Run the notebook completely at least once. Verify `artifacts/results.json` and `inspection_model.joblib` exist.
2. Keep the dataset local, so the presentation does not depend on a download.
3. Open notebook sections 6–9 and the README. Leave the executed outputs visible.
4. Select one ordinary test image and the documented failure case. Learn to explain both.
5. Keep the nine PNGs ready as an offline backup. GitHub shows static outputs, not live widgets.
6. Read `register_pair`, `propose`, `describe`, `detect` and `match_predictions` yourself.

### During the demo

1. Show a reference and inspected image.
2. Show final labelled boxes and scores.
3. Use the dropdown or call `inspect_custom(...)` on a different test pair.
4. Show a deliberately displaced image and its alignment comparison.
5. Show the failure case and explain what its errors mean.

Do not retrain during a short viva. If you restart the kernel, use the reusable module and saved model:

```python
import joblib
from src.pipeline import inspect_custom
bundle = joblib.load('artifacts/inspection_model.joblib')
result = inspect_custom(template_path, inspected_path,
                        bundle['model'], bundle['config'], bundle['threshold'], 'SIFT')
```

If registration fails, say that the system rejected an unreliable alignment. If the demonstration environment
fails, show the previously executed notebook and images; identify them as saved outputs.

## 5. Questions you should be ready to answer

**Why no PyTorch or CNN?**  
A reference image makes classical difference detection a useful starting point. HOG plus a random forest is
CPU-friendly and directly demonstrates course concepts. Deep learning is a future comparison, not a necessity
for this specific implementation. The project does not claim CO5 coverage.

**What is the difference between localisation and classification?**  
Localisation finds candidate defect boxes. Classification assigns a type. A classifier cannot recover a defect
that never became a candidate; this is why proposal recall matters.

**Why HOG?**  
It describes distributions of local edge orientations. Opens, bridges, notches and protrusions have different
local geometry. It is not perfectly invariant to rotation; registration helps stabilise the image geometry.

**Why compare both template and inspected crops?**  
The same shape can have different meanings depending on the intended board structure. The reference gives
context, and signed changes distinguish directions of pixel change.

**Why dilation? Could it hurt?**  
It connects fragmented changed pixels, but it can also merge nearby defects. Its radius is selected on validation.
A merged candidate can cause missed detections or class confusion.

**What is IoU?**  
Intersection area divided by union area of two boxes. At IoU ≥ 0.50, a correct-class prediction can match one
unused ground-truth box. Each ground truth can match at most once.

**Why not report only classification accuracy?**  
Accuracy on supplied defect crops ignores whether the system can find defects. Detection precision, recall,
F1 and AP penalise missed boxes, extra boxes and incorrect classes.

**What is AP and how does it differ from F1?**  
AP summarises the precision–recall ranking curve over scores. F1 is evaluated at one chosen threshold.
Our AP is all-point interpolated at a fixed IoU; it is not COCO's average over many IoU thresholds.

**How was leakage avoided?**  
Split images before making crops. Use only training images to fit the forest and validation to select settings.
Keep test data for final evaluation. However, related layouts may exist across official partitions; that is
a remaining limitation, not something this protocol solves. Also disclose that this project's initial
baseline had already been evaluated on the official test before the morphology revision. No test crops
were used for fitting, but final benchmark performance is not a pristine one-shot estimate.

**How do ORB and SIFT differ here?**  
ORB produces binary descriptors matched with Hamming distance. SIFT produces floating-point descriptors
matched with L2 distance. Both feed correspondences into RANSAC. Use the experiment to discuss their actual
accuracy and runtime rather than assuming one always wins.

**What does RANSAC do?**  
It repeatedly samples correspondences, estimates a transformation and measures consensus. It can tolerate
outliers, but repeated PCB patterns can still support an incorrect transformation.

**Why a homography?**  
A PCB is approximately planar, and a homography models planar perspective changes. The test perturbations
are simpler rotation/translation cases contained within that model. A homography does not correct all lens
distortion or non-planar deformation.

**What happens if registration fails?**  
Inference returns an explicit failure status. In the robustness experiment it yields no detections and all
remaining ground truths count as missed. We do not silently remove the failed image or label it defect-free.

**Do the outputs say whether a board is electrically safe?**  
No. They classify visible patterns under this benchmark. Electrical functionality and industrial deployment
require additional evidence, testing and domain validation.

**What is your contribution?**  
The integrated reference-based pipeline, reproducible protocol, CPU implementation, comparison of alignment
methods, transparent failure handling and analysis. The algorithms and dataset are credited to their authors.
Be explicit about Codex assistance and the parts you have personally studied or modified.

## 6. Course mapping

| Outcome | Evidence | Boundary |
|---|---|---|
| CO1: Image features | HOG, disagreement and signed changes | Handcrafted features |
| CO2: Interest points | ORB and SIFT | No separate Harris/SURF comparison |
| CO3: Feature matching | Descriptor matching, ratio test, homography, RANSAC | Planar image registration |
| CO4: Segmentation / tracking | Binary masks, morphology, connected components | No temporal tracking |
| CO5: Deep detection | Future extension only | Not implemented in this version |

## 7. Improvements to propose honestly

1. Hold out whole circuit layouts to measure generalisation beyond shared source groups.
2. Collect real defects and harder acquisition conditions with appropriate rights.
3. Improve splitting of merged candidate components.
4. Test confidence calibration and an explicit review/abstention policy.
5. Compare a deep detector under the same split and metric definitions if resources and course scope permit.

These are proposed next steps. Do not imply they have already been completed.

## 8. Final preparation

You should be able to explain the pipeline without reading code, calculate one simple IoU example,
identify one real failure, distinguish validation from test, and explain why a high benchmark score does not
establish factory readiness. Credit Tang et al. for DeepPCB and acknowledge software and AI assistance.
'''
(root / 'FACULTY_GUIDE.md').write_text(guide, encoding='utf-8')
print('README.md and FACULTY_GUIDE.md generated from measured outputs.')
