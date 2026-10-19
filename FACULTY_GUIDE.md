# Faculty Presentation and Viva Guide

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
1107-dimensional vector to background or one of six classes.”

The model has 160 trees. Training used 23580 examples derived only from training images.
Background examples allow rejection of false candidates. The validation confidence cutoff is
0.00. A forest score is not a calibrated likelihood of physical failure.

Explain the development history: the first baseline exposed low recall. Training imagery and validation
proposal recall motivated testing morphological opening. Settings were selected on validation, but the
official test had already been observed. The final score is therefore a reused benchmark result.

### State your measured results accurately

On **all 500 official test images**, at **IoU 0.50**:

- Precision: **95.50%**.
- Recall: **89.87%**.
- F1: **92.60%**.
- All-point interpolated mAP@0.50: **89.60%**.
- True positives: **2822**; false positives: **133**; missed defects: **318**.

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

| Condition | Method | F1 @ 0.50 | Failed registrations |
|---|---|---:|---:|
| clean | No alignment | 93.07% | 0 |
| clean | ORB + RANSAC | 89.66% | 4 |
| clean | SIFT + RANSAC | 90.70% | 3 |
| mild | No alignment | 70.55% | 0 |
| mild | ORB + RANSAC | 90.29% | 4 |
| mild | SIFT + RANSAC | 85.12% | 7 |
| moderate | No alignment | 37.84% | 0 |
| moderate | ORB + RANSAC | 84.46% | 6 |
| moderate | SIFT + RANSAC | 85.03% | 7 |


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
