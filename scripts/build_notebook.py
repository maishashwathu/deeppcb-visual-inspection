"""Build an explanatory, self-contained notebook from reviewed pipeline sections."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'src/pipeline.py').read_text(encoding='utf-8')
sections = {}
for block in source.split('# %% ')[1:]:
    name, content = block.split('\n', 1)
    sections[name.strip()] = content.strip()
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))

md('''# DeepPCB: Explainable Visual Inspection on a CPU
### Defect localisation, six-class recognition, and robustness to camera misalignment

**Course:** Computer Vision for AI · **Student:** Shashwat Jain  
**Dataset:** [DeepPCB by Tang et al.](https://github.com/tangsanli5201/DeepPCB)  
**Implementation:** developed with OpenAI Codex assistance; review and understand the implementation before presenting it.

This notebook implements an original educational pipeline using OpenCV and scikit-learn.
**No PyTorch, GPU, YOLO weights, or hidden pretrained model is required.**
The dataset name “DeepPCB” does not mean that every method using it must be deep learning.

**Research question:** Can reference-image comparison and handcrafted features detect and classify PCB defects,
and how much does feature-based registration recover performance when the camera moves?

**Roadmap:** setup → data audit → exploratory visuals → validation split → proposals → HOG classifier →
held-out detection metrics → alignment stress test → failure analysis → reusable inference.

> Dataset images contain artificially added defects and are already aligned and binarised.
> This is a research prototype, not a validated factory inspection system.''')
md('''## 1. Setup and reproducibility
Use Python 3.12 (tested), or a compatible Python 3.10+ installation. Run the next cell once in a fresh environment.
It uses the current notebook kernel. Existing compatible packages are retained.
Then choose **Run All**. The full workflow downloads the data if absent, trains on CPU, and writes results.
The dataset is pinned to a source revision; outputs may vary slightly with library versions and hardware.''')
code('''%pip install "numpy>=1.26,<3" "pandas>=2.2,<4" "opencv-python-headless==4.11.0.86" "scikit-learn>=1.5,<2" "matplotlib>=3.9,<4" "joblib>=1.4,<2"''')
cells[-1].metadata['tags'] = ['installation']
code(sections['imports'])
code('''ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
ARTIFACTS = ROOT / 'artifacts'
ARTIFACTS.mkdir(exist_ok=True)
versions = {'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'opencv': cv2.__version__}
import sklearn, matplotlib
versions.update(sklearn=sklearn.__version__, matplotlib=matplotlib.__version__)
print('Runtime:', versions)
print('CPU only; random seed:', SEED)''')
md('''## 2. Download, validate, and understand the data
DeepPCB provides **1,500 image pairs** and six defect categories. The official split is 1,000 training/validation
pairs and 500 test pairs. Each pair contains a clean reference and inspected board, both 640 × 640 pixels.
The original split files mention a base `.jpg`; the actual images use `_temp.jpg` and `_test.jpg` suffixes.

Labels contain `x1 y1 x2 y2 class_id`. This implementation uses a consistent continuous/half-open box convention
for width, height and IoU (no `+1` area correction). Read the upstream description before comparing scores.

| ID | Defect | Interpretation |
|---|---|---|
| 1 | Open | Interrupted conducting path |
| 2 | Short | Unwanted bridge between paths |
| 3 | Mousebite | Notch in a track edge |
| 4 | Spur | Protrusion from a track |
| 5 | Spurious copper | Isolated unwanted material |
| 6 | Pin hole | Hole inside a conducting region |

The upstream README restricts the **dataset to research use**. Raw images are downloaded locally and excluded
from this repository. Visual examples derived from them are credited in the figures and CREDITS.md.''')
code(sections['data'])
code('''DATA_DIR = ensure_dataset(ROOT)
trainval = read_split(DATA_DIR, 'trainval')
test = read_split(DATA_DIR, 'test')
inventory = audit_dataset(trainval, test)
train, val = split_training(trainval)
assert not ({r['id'] for r in train} & {r['id'] for r in val})
assert not ({r['id'] for r in trainval} & {r['id'] for r in test})
manifest = {'source_commit': SOURCE_COMMIT, 'seed': SEED,
            'train': [r['id'] for r in train], 'validation': [r['id'] for r in val],
            'test': [r['id'] for r in test]}
(ARTIFACTS / 'split_manifest.json').write_text(json.dumps(manifest, indent=2))
print(f'Audited {len(trainval)+len(test)} image pairs and {len(inventory)} labelled defects.')
print(f'Train: {len(train)} | Validation: {len(val)} | Test: {len(test)}')
display(inventory.groupby(['split', 'defect']).size().unstack(fill_value=0))''')
md('''### Split discipline and limitations
- Training: 800 images; validation: 200 images; official test: 500 images.
- Split before extracting crops. All crops from one image remain in the same split.
- The validation split is stratified by upstream source group, **not group-disjoint**.
- Source layouts may be shared across official partitions. Exact tested-image duplicates are checked, but
  passing this check does not rule out related layouts or near duplicates.
- Hyperparameters and confidence thresholds use validation only. Test labels are used for audit,
  final evaluation and post-hoc error analysis, never feature extraction or inference.
- An unseen-layout claim would require a separate group-held-out experiment.''')
md('''### Development transparency
The initial baseline was evaluated on the official test set and exposed low recall. Inspection of a **training**
pair and validation proposal recall motivated adding morphological opening to suppress thin boundary artifacts.
The opening radius and other settings are selected on validation only. Initial results are preserved in
`artifacts/initial_baseline_results.json`. Because the official test was already observed during development,
the final result is a **reused benchmark evaluation**, not a pristine one-shot estimate of generalisation.
A new independent dataset would be needed for that stronger claim.''')
md('''## 3. Exploratory analysis and visual inspection
Inspect an actual image pair, class frequencies and defect sizes before selecting methods.
Tiny defects and repeated circuit patterns motivate local geometry features and careful image registration.''')
code(sections['visuals'])
code('''show_pair(train[0], ROOT)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
counts = inventory.groupby(['defect','split']).size().unstack(fill_value=0)
counts.plot.bar(ax=axes[0], color=['#38bdf8','#0f766e'])
axes[0].set(title='Defect frequency: official splits', ylabel='Annotated boxes', xlabel='')
axes[0].tick_params(axis='x', rotation=30)
axes[1].hist(inventory.width, bins=30, alpha=.7, label='Width', color='#0f766e')
axes[1].hist(inventory.height, bins=30, alpha=.6, label='Height', color='#f59e0b')
axes[1].set(title='Defect box dimensions', xlabel='Pixels', ylabel='Boxes'); axes[1].legend()
fig.tight_layout(); save_figure(fig, ROOT, '02_dataset_statistics.png')''')
md('''## 4. Classical defect proposals
1. Threshold both grayscale images at 127 (appropriate for this already binarised dataset).
2. Compute pixel disagreement: `template_binary XOR inspected_binary`.
3. Optionally apply morphological opening to remove thin edge artifacts, then dilate to connect defect pieces.
4. Find connected components; remove tiny components by original changed-pixel area.
5. Pad each component to provide local context and a bounding box.

This step is **class-agnostic**. It does not read annotations. Threshold 127, padding 9, feature sizes and model
structure are fixed design choices; the small proposal search and classification threshold use validation.
Proposal recall asks whether each real defect has *any* candidate with IoU ≥ 0.50; it is an upper bound,
not a full detector score (one candidate can cover multiple annotations in this diagnostic).''')
code(sections['proposals'])
code(sections['evaluation'])
code('''proposal_results = pd.DataFrame([
    proposal_recall(val, dict(min_area=a, dilation=d, pad=9, opening=o))
    for a in [4, 8, 16] for d in [1, 2] for o in [0, 1]
])
proposal_results = proposal_results.sort_values(
    ['recall_at_50', 'proposals_per_image'], ascending=[False, True])
display(proposal_results)
chosen = proposal_results.iloc[0]
CONFIG = {key: int(chosen[key]) for key in ['min_area', 'dilation', 'pad', 'opening']}
proposal_results.to_csv(ARTIFACTS / 'validation_proposals.csv', index=False)
print('Selected using validation only:', CONFIG)
t, s = load_pair(train[0])
boxes, diff, mask = propose(t, s, **CONFIG)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(diff, cmap='gray'); axes[0].set_title('Disagreement after optional opening'); axes[0].axis('off')
axes[1].imshow(mask, cmap='gray'); axes[1].set_title('After dilation'); axes[1].axis('off')
draw_boxes(axes[2], s, boxes, [0]*len(boxes), 'Class-agnostic candidates')
fig.tight_layout(); save_figure(fig, ROOT, '03_proposals.png')''')
md('''## 5. Learn defect types with HOG and a random forest
Each candidate becomes a feature vector comprising:
- HOG on 32 × 32 crops of the template, inspected image and disagreement mask;
- two 8 × 8 maps preserving the direction of intensity changes;
- seven geometry/occupancy measurements.

A random forest learns **background plus six defect classes**. Training examples come only from training
images: proposals matched to annotations at IoU ≥ 0.33, slightly jittered annotation crops, and background crops.
The 0.33 value here is a *training assignment rule*, separate from the final 0.50 evaluation criterion.
Every inference box is generated from pixels. No ground-truth crop is supplied at inference.

Random-forest vote scores are used for ranking and thresholding; they are **not calibrated probabilities**.
This is classical supervised machine learning, not deep learning. No backpropagation or GPU is needed.''')
code('''training_start = time.perf_counter()
X_train, y_train = make_training_data(train, CONFIG)
print('Training feature matrix:', X_train.shape)
display(pd.Series(y_train).map(CLASSES).value_counts().rename('training examples'))
model = fit_classifier(X_train, y_train)
training_seconds = time.perf_counter()-training_start
print(f'Feature extraction + training: {training_seconds:.1f} seconds')
val_raw, _ = predict_records(val, model, CONFIG, threshold=0.)
threshold_rows = []
for threshold in [0., .25, .35, .45, .55, .65]:
    _, result = score_predictions(val, threshold_predictions(val_raw, threshold), .5)
    threshold_rows.append(dict(threshold=threshold, **result))
threshold_table = pd.DataFrame(threshold_rows).sort_values(['F1','threshold'], ascending=[False,True])
THRESHOLD = float(threshold_table.iloc[0].threshold)
display(threshold_table)
threshold_table.to_csv(ARTIFACTS / 'validation_thresholds.csv', index=False)
print('Frozen confidence threshold:', THRESHOLD)
joblib.dump(dict(model=model, config=CONFIG, threshold=THRESHOLD,
                 classes=CLASSES, source_commit=SOURCE_COMMIT, versions=versions),
            ARTIFACTS / 'inspection_model.joblib')''')
md('''## 6. Final evaluation on all 500 official test images
**IoU** measures box overlap. A prediction is a true positive only when its class is correct and it matches
an unused ground-truth box above the IoU threshold. Duplicates count as false positives.

- Precision: fraction of predictions that are correct.
- Recall: fraction of real defects found correctly.
- F1: balance between precision and recall.
- mAP: mean all-point interpolated average precision across the six defect classes.

Report **mAP@0.50** using all non-background scores (no operating confidence cutoff), while precision/recall/F1
use the validation-selected cutoff. Also report results at **IoU 0.33** for context. Our evaluator is not the
upstream evaluation script and not COCO mAP; do not claim direct leaderboard comparability.
The official README uses a strict >0.33 match; this implementation uses ≥0.33.''')
code('''test_raw, test_seconds = predict_records(test, model, CONFIG, threshold=0.)
test_predictions = threshold_predictions(test_raw, THRESHOLD)
per_class, test_metrics = score_predictions(test, test_predictions, .5)
ap_class, ap_metrics = score_predictions(test, test_raw, .5)
per_class['AP'] = ap_class['AP']
test_metrics['mAP'] = ap_metrics['mAP']
_, contextual = score_predictions(test, test_predictions, .33)
_, contextual_ap = score_predictions(test, test_raw, .33)
contextual['mAP'] = contextual_ap['mAP']
display(pd.DataFrame([test_metrics, contextual]))
display(per_class.round(4))
print(f'Clean-image inference (including disk I/O): {1000*test_seconds/len(test):.1f} ms/image')
per_class.to_csv(ARTIFACTS / 'test_per_class.csv', index=False)
(ARTIFACTS / 'test_predictions.json').write_text(json.dumps(test_predictions))
fig, ax = plt.subplots(figsize=(10, 4))
per_class.set_index('defect')[['precision','recall','AP']].plot.bar(ax=ax, color=['#0f766e','#38bdf8','#f59e0b'])
ax.set(ylim=(0,1.05), title='Held-out test performance | IoU 0.50', ylabel='Score', xlabel='')
ax.tick_params(axis='x', rotation=25); fig.tight_layout()
save_figure(fig, ROOT, '04_test_metrics.png')
show_predictions(test[0], test_predictions, ROOT)''')
md('''## 7. Alignment with ORB / SIFT and RANSAC
The baseline assumes the pair is aligned. A camera shift can turn every track boundary into a false defect.
We estimate a homography from **inspected image to reference image** using descriptor matching, a 0.75 ratio
test and RANSAC. ORB uses Hamming distance; SIFT uses L2 distance. Weak or geometrically implausible fits are
rejected. Newly exposed borders are excluded using a warped validity mask.

Synthetic stress test: sample 30 test images with a fixed seed, then evaluate clean, mild (1° / +3,-2 px)
and moderate (2° / +6,-4 px) conditions. Compare no alignment, ORB and SIFT. No settings are selected from these
test results. Labels stay in reference coordinates because the aim is to restore the reference frame.

**Failed registration remains in the denominator:** its empty detections count as missed defects.
Border defects are retained in ground truth, so crop/visibility loss can also reduce recall.
This controlled experiment tests specific perturbations, not all real camera conditions.''')
code(sections['alignment'])
code('''robustness, robustness_ids = robustness_experiment(test, model, CONFIG, THRESHOLD, limit=30)
display(robustness[['condition','method','precision','recall','F1','registration_failures',
                    'mean_reprojection_error_px','milliseconds_per_image']].round(4))
robustness.to_csv(ARTIFACTS / 'robustness.csv', index=False)
(ARTIFACTS / 'robustness_ids.json').write_text(json.dumps(robustness_ids, indent=2))
fig, ax = plt.subplots(figsize=(10,4))
robustness.pivot(index='condition', columns='method', values='F1').reindex(['clean','mild','moderate']).plot.bar(ax=ax)
ax.set(ylim=(0,1.05), ylabel='Detection F1 at IoU 0.50', xlabel='', title='Does registration recover defect detection?')
ax.tick_params(axis='x', rotation=0); fig.tight_layout()
save_figure(fig, ROOT, '06_alignment_robustness.png')
t, s = load_pair(test[0]); moved, _ = perturb_image(s)
aligned, valid, info = register_pair(t, moved, 'SIFT')
fig, axes = plt.subplots(1, 3, figsize=(14,4))
axes[0].imshow(cv2.absdiff(t,moved), cmap='inferno'); axes[0].set_title('Difference before alignment')
if info['success']:
    axes[1].imshow(aligned, cmap='gray'); axes[1].set_title('SIFT + RANSAC aligned image')
    residual = cv2.absdiff(t,aligned); residual[~valid] = 0
    axes[2].imshow(residual, cmap='inferno'); axes[2].set_title('Difference after alignment')
else:
    axes[1].text(.1,.5,'Registration failed'); axes[2].text(.1,.5,info['reason'])
for ax in axes: ax.axis('off')
fig.tight_layout(); save_figure(fig, ROOT, '07_alignment_example.png')''')
md('''## 8. Honest error analysis
Inspect the weakest test image, selected *after* final evaluation for diagnosis only. Do not tune the model
against this image and then report the same test set as untouched.
The confusion matrix below matches boxes by geometry first and includes an explicit background/missed category.
It therefore reveals classification errors and missed/extra detections, unlike a crop-only accuracy score.''')
code('''errors = []
y_true, y_pred = [], []
for rec in test:
    gt = ground_truth(rec); pp = test_predictions[rec['id']]
    outcomes, used = match_predictions(pp, gt, .5, class_aware=False)
    used_for_labels = set()
    for pred, matched in outcomes:
        if matched:
            overlaps = box_iou([pred['box']], gt[:,:4])[0]
            eligible = [j for j in range(len(gt)) if j not in used_for_labels]
            j = max(eligible, key=lambda j: overlaps[j]); used_for_labels.add(j)
            y_true.append(int(gt[j,4])); y_pred.append(pred['label'])
        else:
            y_true.append(0); y_pred.append(pred['label'])
    for j in range(len(gt)):
        if j not in used_for_labels:
            y_true.append(int(gt[j,4])); y_pred.append(0)
    _, score = score_predictions([rec], {rec['id']:pp}, .5)
    errors.append(dict(image=rec['id'], **score))
error_table = pd.DataFrame(errors).sort_values(['F1','image'])
error_table.to_csv(ARTIFACTS / 'test_image_errors.csv', index=False)
display(error_table.head(5))
worst = next(r for r in test if r['id'] == error_table.iloc[0].image)
show_predictions(worst, test_predictions, ROOT, '08_failure_case.png')
fig, ax = plt.subplots(figsize=(9,8))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=list(range(7)),
    display_labels=['background / missed']+[CLASSES[c] for c in range(1,7)],
    ax=ax, cmap='Blues', colorbar=False, xticks_rotation=45)
ax.set_title('Detection confusion matrix | geometry matching at IoU 0.50')
fig.tight_layout(); save_figure(fig, ROOT, '09_confusion_matrix.png')''')
md('''## 9. Reusable inference and optional interactive demo
Use a matching defect-free template and inspected image. Outputs are in **template coordinates**.
Registration failure is an explicit failure state, never silently interpreted as “no defects”.
An empty successful result means “no accepted detections”, not proof of a defect-free board.
The function takes no ground-truth labels.

The dropdown is optional; GitHub displays saved static outputs, while Jupyter can run the widget.
To inspect your own pair, replace the two paths in `inspect_custom(...)` below.''')
code('''result = inspect_custom(test[0]['template'], test[0]['test'], model, CONFIG, THRESHOLD, method='SIFT')
print('Inspection status:', result['status'])
display(pd.DataFrame(result['predictions']))
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    choices = widgets.Dropdown(options=[r['id'] for r in test[:20]], description='Test image:')
    output = widgets.Output()
    def preview(change=None):
        rec = next(r for r in test if r['id'] == choices.value)
        with output:
            clear_output(wait=True)
            t, s = load_pair(rec); pp = test_predictions[rec['id']]
            fig, ax = plt.subplots(figsize=(7,7))
            draw_boxes(ax, s, [p['box'] for p in pp], [p['label'] for p in pp], rec['id'])
            plt.show(); plt.close(fig)
    choices.observe(preview, names='value')
    display(choices, output)
except ImportError:
    print('Optional interactive dropdown requires ipywidgets. Static figures remain available.')''')
md('''## 10. Export results and interpret the outcome
The JSON below records the measured results, chosen settings, runtime, versions and source revision.
Model weights stay local because retraining is part of the exercise. Only load joblib files you trust.

### What this project demonstrates
- CO1: handcrafted image features and HOG.
- CO2: ORB / SIFT interest points and descriptors.
- CO3: matching, homography and RANSAC.
- CO4: binary segmentation and connected components (not temporal object tracking).
- CO5: **not demonstrated** by this CPU-only classical model; a deep detector is a future extension.

### Limits and next experiments
1. Shared source layouts, artificial defects and preprocessed images limit deployment claims.
2. Proposal recall limits the classifier; merged components can hide multiple defects.
3. Repeated tracks can produce plausible but wrong registration. Current safeguards do not guarantee correctness.
4. New camera/lighting conditions need new data and validation; geometric stress tests alone do not cover them.
5. Future work: group-held-out evaluation, real factory images, controlled lighting tests, then a YOLO comparison
   if the research question requires deep learning. Do not present these as completed experiments.

### Credits
Dataset: Tang et al., *On-line PCB Defect Detector On A New PCB Defect Dataset*,
[paper](https://arxiv.org/abs/1902.06197), [repository](https://github.com/tangsanli5201/DeepPCB).
Methods and software: OpenCV (ORB/SIFT/RANSAC/HOG), scikit-learn (random forest), NumPy, pandas,
Matplotlib, Jupyter and joblib. See `CREDITS.md` for references and software links.
Code and explanatory materials were developed with OpenAI Codex assistance; results are measured by running
this implementation, not copied from the upstream benchmark.''')
code('''summary = dict(dataset_revision=SOURCE_COMMIT, seed=SEED, versions=versions,
    split_sizes=dict(train=len(train), validation=len(val), test=len(test)),
    proposal_config=CONFIG, confidence_threshold=THRESHOLD,
    training_seconds=training_seconds, test_inference_seconds=test_seconds,
    test_at_50=test_metrics, test_at_33=contextual,
    feature_dimension=int(X_train.shape[1]), training_examples=int(len(y_train)),
    robustness_subset_size=len(robustness_ids),
    evaluation='class-aware one-to-one, all-point AP; not official DeepPCB or COCO evaluator')
(ARTIFACTS / 'results.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print('Completed: model, metrics, split manifest, predictions and nine figures saved.')''')
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.update(kernelspec={'display_name':'Python 3', 'language':'python', 'name':'python3'},
                   language_info={'name':'python','version':'3.12'})
nbf.validate(nb)
nbf.write(nb, ROOT / 'DeepPCB_Inspection.ipynb')
print(f'Created notebook with {len(cells)} cells')
