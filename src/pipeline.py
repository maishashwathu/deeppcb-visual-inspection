"""Original educational implementation; see CREDITS.md for data and methods.

Sections are embedded directly into the notebook by scripts/build_notebook.py.
"""
# %% imports
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import time
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import joblib

SEED = 42
SOURCE_COMMIT = '08e98c4db5922613fb97176eb3d6497d48260cb1'
CLASSES = {0: 'background', 1: 'open', 2: 'short', 3: 'mousebite',
           4: 'spur', 5: 'spurious copper', 6: 'pin hole'}
COLORS = ['#64748b', '#ef4444', '#f59e0b', '#22c55e', '#06b6d4', '#a855f7', '#ec4899']
cv2.setNumThreads(2)
cv2.setRNGSeed(SEED)
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 10})

# %% data
def ensure_dataset(root):
    """Download the pinned upstream archive once; never publish raw data."""
    root = Path(root)
    target = root / 'data' / 'DeepPCB'
    if (target / 'PCBData' / 'trainval.txt').exists():
        if (target / '.git').exists():
            actual = subprocess.check_output(
                ['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
            if actual != SOURCE_COMMIT:
                raise ValueError(f'Dataset revision {actual} differs from pinned {SOURCE_COMMIT}')
        return target / 'PCBData'
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.parent / 'DeepPCB-pinned.zip'
    url = f'https://codeload.github.com/tangsanli5201/DeepPCB/zip/{SOURCE_COMMIT}'
    print('Downloading official DeepPCB archive (research use only)...')
    urllib.request.urlretrieve(url, archive)
    prefix = f'DeepPCB-{SOURCE_COMMIT}/'
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            if not entry.filename.startswith(prefix) or entry.is_dir():
                continue
            relative = Path(entry.filename[len(prefix):])
            dest = (target / relative).resolve()
            if not dest.is_relative_to(target.resolve()):
                raise ValueError('Unsafe archive path')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(entry))
    (target / 'SOURCE_COMMIT').write_text(SOURCE_COMMIT)
    return target / 'PCBData'


def read_split(data_dir, name):
    records = []
    for line in (data_dir / f'{name}.txt').read_text().splitlines():
        image_rel, label_rel = line.split()
        base = data_dir / image_rel
        rec = dict(id=base.stem, group=Path(image_rel).parts[0],
                   test=str(base.with_name(base.stem + '_test.jpg')),
                   template=str(base.with_name(base.stem + '_temp.jpg')),
                   annotation=str(data_dir / label_rel))
        for key in ['test', 'template', 'annotation']:
            if not Path(rec[key]).is_file():
                raise FileNotFoundError(rec[key])
        records.append(rec)
    return records


def load_pair(rec):
    template = cv2.imread(rec['template'], cv2.IMREAD_GRAYSCALE)
    tested = cv2.imread(rec['test'], cv2.IMREAD_GRAYSCALE)
    if template is None or tested is None or template.shape != tested.shape:
        raise ValueError(f'Invalid image pair: {rec["id"]}')
    return template, tested


def ground_truth(rec):
    a = np.loadtxt(rec['annotation'], dtype=int, ndmin=2)
    if a.shape[1] != 5 or not np.isin(a[:, 4], list(range(1, 7))).all():
        raise ValueError(f'Invalid annotation: {rec["id"]}')
    return a


def audit_dataset(trainval, test):
    assert len(trainval) == 1000 and len(test) == 500
    assert len({r['id'] for r in trainval + test}) == 1500
    rows, hashes = [], {}
    for split, records in [('trainval', trainval), ('test', test)]:
        for rec in records:
            template, tested = load_pair(rec)
            assert tested.shape == (640, 640)
            gt = ground_truth(rec)
            assert (gt[:, :2] >= 0).all() and (gt[:, 2:4] <= 640).all()
            assert (gt[:, 2:4] > gt[:, :2]).all()
            digest = hashlib.sha256(tested.tobytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise AssertionError('Exact tested-image duplicate across official splits')
            hashes[digest] = split
            for box in gt:
                rows.append(dict(split=split, image=rec['id'], group=rec['group'],
                                 class_id=int(box[4]), defect=CLASSES[int(box[4])],
                                 width=int(box[2]-box[0]), height=int(box[3]-box[1])))
    return pd.DataFrame(rows)


def split_training(trainval):
    """Image-disjoint validation, stratified by source group; NOT layout-disjoint."""
    train, val = train_test_split(trainval, test_size=0.2, random_state=SEED,
                                 stratify=[r['group'] for r in trainval])
    return sorted(train, key=lambda r:r['id']), sorted(val, key=lambda r:r['id'])

# %% proposals
def box_iou(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(high-low, 0).prod(axis=2)
    aa = np.maximum(a[:, 2:]-a[:, :2], 0).prod(axis=1)
    bb = np.maximum(b[:, 2:]-b[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(aa[:, None] + bb[None, :] - intersection, 1e-9)


def propose(template, tested, min_area=8, dilation=1, pad=9, opening=0, valid=None):
    """Binary disagreement -> morphology -> components -> padded xyxy boxes."""
    t = template > 127
    s = tested > 127
    diff = (t != s).astype(np.uint8)
    if valid is not None:
        diff[~valid] = 0
    if opening:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*opening+1, 2*opening+1))
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(diff, np.ones((2*dilation+1, 2*dilation+1), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    h, w = tested.shape
    for idx in range(1, n):
        x, y, bw, bh, _ = stats[idx]
        actual_area = int(diff[y:y+bh, x:x+bw][labels[y:y+bh, x:x+bw] == idx].sum())
        if actual_area < min_area:
            continue
        boxes.append([max(0, x-pad), max(0, y-pad),
                      min(w, x+bw+pad), min(h, y+bh+pad)])
    return np.asarray(boxes, dtype=int).reshape(-1, 4), diff, mask


HOG = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)


def describe(template, tested, box):
    """HOG captures local geometry; signed changes distinguish missing/extra material."""
    x1, y1, x2, y2 = np.asarray(box, int)
    h, w = template.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    t = template[y1:y2, x1:x2]
    s = tested[y1:y2, x1:x2]
    if not t.size:
        raise ValueError('Empty feature crop')
    tb, sb = t > 127, s > 127
    plus, minus = sb & ~tb, tb & ~sb
    channels = [t, s, (tb != sb).astype(np.uint8)*255]
    hog = [HOG.compute(cv2.resize(ch, (32, 32))).ravel() for ch in channels]
    signed_grid = [cv2.resize(ch.astype(np.float32), (8, 8),
                              interpolation=cv2.INTER_AREA).ravel() for ch in [plus, minus]]
    geom = [np.log1p(x2-x1), np.log1p(y2-y1), (x2-x1)/(y2-y1),
            tb.mean(), sb.mean(), plus.mean(), minus.mean()]
    return np.concatenate(hog + signed_grid + [geom]).astype(np.float32)


def make_training_data(records, config):
    """Only training images; proposal labels assigned by IoU, never at inference."""
    rng = np.random.default_rng(SEED)
    xx, yy = [], []
    for rec in records:
        t, s = load_pair(rec)
        gt = ground_truth(rec)
        boxes, _, _ = propose(t, s, **config)
        if len(boxes):
            overlaps = box_iou(boxes, gt[:, :4])
            for box, overlap in zip(boxes, overlaps):
                j = int(overlap.argmax())
                label = int(gt[j, 4]) if overlap[j] >= 0.33 else 0
                xx.append(describe(t, s, box)); yy.append(label)
        # Ground-truth crops with small box jitter teach proposal-boundary tolerance.
        for row in gt:
            b = row[:4].copy()
            b += rng.integers(-3, 4, size=4)
            b[:2] = np.maximum(b[:2], 0); b[2:] = np.minimum(b[2:], 640)
            if np.all(b[2:] > b[:2]):
                xx.append(describe(t, s, b)); yy.append(int(row[4]))
        for _ in range(2):
            x, y = rng.integers(0, 596, size=2)
            b = [x, y, x+44, y+44]
            if box_iou([b], gt[:, :4]).max() < 0.05:
                xx.append(describe(t, s, b)); yy.append(0)
    return np.asarray(xx), np.asarray(yy)


def fit_classifier(x, y):
    model = RandomForestClassifier(n_estimators=160, max_features='sqrt',
                                   min_samples_leaf=2, class_weight='balanced_subsample',
                                   random_state=SEED, n_jobs=2)
    return model.fit(x, y)


def detect(template, tested, model, config, threshold=0.35, valid=None):
    boxes, _, _ = propose(template, tested, valid=valid, **config)
    if len(boxes) == 0:
        return []
    features = np.stack([describe(template, tested, b) for b in boxes])
    probability = model.predict_proba(features)
    labels = model.classes_[probability.argmax(axis=1)]
    confidence = probability.max(axis=1)
    return [dict(box=b.tolist(), label=int(c), score=float(p))
            for b, c, p in zip(boxes, labels, confidence) if c != 0 and p >= threshold]

# %% evaluation
def match_predictions(predictions, gt, iou_threshold=0.5, class_aware=True):
    """Confidence-ordered one-to-one assignment; duplicate detections are false positives."""
    used, outcomes = set(), []
    for pred in sorted(predictions, key=lambda p: -p['score']):
        overlaps = box_iou([pred['box']], gt[:, :4])[0]
        eligible = [j for j in range(len(gt)) if j not in used
                    and (not class_aware or gt[j, 4] == pred['label'])]
        best = max(eligible, key=lambda j: overlaps[j], default=None)
        ok = best is not None and overlaps[best] >= iou_threshold
        if ok:
            used.add(best)
        outcomes.append((pred, bool(ok)))
    return outcomes, used


def score_predictions(records, predictions, iou_threshold=0.5):
    gt_by_id = {r['id']: ground_truth(r) for r in records}
    rows = []
    for c in range(1, 7):
        entries, count = [], 0
        for rec in records:
            gt = gt_by_id[rec['id']]
            gt = gt[gt[:, 4] == c]
            count += len(gt)
            pp = [p for p in predictions[rec['id']] if p['label'] == c]
            outcomes, _ = match_predictions(pp, gt, iou_threshold)
            entries.extend((p['score'], ok) for p, ok in outcomes)
        entries.sort(key=lambda x: -x[0])
        tp = np.cumsum([int(x[1]) for x in entries], dtype=float)
        fp = np.cumsum([int(not x[1]) for x in entries], dtype=float)
        recall = tp / max(count, 1)
        precision = tp / np.maximum(tp+fp, 1)
        # All-point interpolated AP (not COCO's 101-point / multi-IoU AP).
        mr = np.r_[0., recall, 1.]
        mp = np.r_[0., precision, 0.]
        mp = np.maximum.accumulate(mp[::-1])[::-1]
        indices = np.where(mr[1:] != mr[:-1])[0]
        ap = float(np.sum((mr[indices+1]-mr[indices])*mp[indices+1]))
        ntp = int(tp[-1]) if len(tp) else 0
        nfp = len(entries)-ntp
        nfn = count-ntp
        p, r = ntp/max(ntp+nfp, 1), ntp/max(count, 1)
        rows.append(dict(defect=CLASSES[c], class_id=c, ground_truth=count,
                         TP=ntp, FP=nfp, FN=nfn, precision=p, recall=r,
                         F1=2*p*r/max(p+r, 1e-9), AP=ap))
    frame = pd.DataFrame(rows)
    ntp, nfp, nfn = (int(frame[k].sum()) for k in ['TP', 'FP', 'FN'])
    p, r = ntp/max(ntp+nfp, 1), ntp/max(ntp+nfn, 1)
    summary = dict(IoU=iou_threshold, precision=p, recall=r,
                   F1=2*p*r/max(p+r, 1e-9), mAP=float(frame.AP.mean()),
                   TP=ntp, FP=nfp, FN=nfn, images=len(records))
    return frame, summary


def predict_records(records, model, config, threshold=0.):
    start = time.perf_counter()
    predictions = {}
    for rec in records:
        t, s = load_pair(rec)
        predictions[rec['id']] = detect(t, s, model, config, threshold)
    return predictions, time.perf_counter()-start


def threshold_predictions(predictions, threshold):
    return {k: [p for p in values if p['score'] >= threshold]
            for k, values in predictions.items()}


def proposal_recall(records, config):
    found, total, candidates = 0, 0, 0
    for rec in records:
        t, s = load_pair(rec)
        boxes, _, _ = propose(t, s, **config)
        gt = ground_truth(rec)
        total += len(gt); candidates += len(boxes)
        if len(boxes):
            found += int((box_iou(boxes, gt[:, :4]).max(axis=0) >= .5).sum())
    return dict(**config, recall_at_50=found/total,
                proposals_per_image=candidates/len(records))

# %% alignment
def register_pair(template, tested, method='ORB'):
    """Estimate tested -> template homography, reject unstable geometry, mask borders."""
    detector = (cv2.ORB_create(nfeatures=2500) if method == 'ORB'
                else cv2.SIFT_create(nfeatures=2000))
    kp_t, des_t = detector.detectAndCompute(template, None)
    kp_s, des_s = detector.detectAndCompute(tested, None)
    norm = cv2.NORM_HAMMING if method == 'ORB' else cv2.NORM_L2
    if des_t is None or des_s is None or len(des_t) < 2 or len(des_s) < 4:
        return None, None, dict(success=False, reason='too few descriptors')
    pairs = cv2.BFMatcher(norm).knnMatch(des_s, des_t, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair]
            if m.distance < .75*n.distance]
    if len(good) < 8:
        return None, None, dict(success=False, reason='too few ratio-test matches')
    src = np.float32([kp_s[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    cv2.setRNGSeed(SEED)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.)
    ratio = float(inliers.mean()) if inliers is not None else 0.
    h, w = template.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    if H is None or not np.isfinite(H).all() or ratio < .3 or inliers.sum() < 8:
        return None, None, dict(success=False, reason='weak RANSAC consensus')
    warped_corners = cv2.perspectiveTransform(corners, H)
    area_ratio = abs(cv2.contourArea(warped_corners))/(w*h)
    if not .6 < area_ratio < 1.5 or not cv2.isContourConvex(warped_corners):
        return None, None, dict(success=False, reason='implausible warp')
    aligned = cv2.warpPerspective(tested, H, (w, h))
    valid = cv2.warpPerspective(np.ones_like(tested), H, (w, h),
                                flags=cv2.INTER_NEAREST) > 0
    valid = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    metadata = dict(success=True, matches=len(good), inlier_ratio=ratio, H=H)
    return aligned, valid, metadata


def perturb_image(image, angle=2., dx=6., dy=-4.):
    h, w = image.shape
    A = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.)
    A[:, 2] += [dx, dy]
    H = np.vstack([A, [0., 0., 1.]])
    return cv2.warpPerspective(image, H, (w, h)), H


def robustness_experiment(records, model, config, threshold, limit=30):
    """Deterministic test subset, untouched selection. Full frames, including edge defects.

    All predictions are evaluated in reference coordinates. Failed registration counts
    as an empty detection set, not as a silently excluded image.
    """
    rng = np.random.default_rng(SEED)
    subset = [records[i] for i in sorted(rng.choice(len(records), min(limit, len(records)), replace=False))]
    conditions = [('clean', 0., 0., 0.), ('mild', 1., 3., -2.), ('moderate', 2., 6., -4.)]
    table, detail = [], []
    for condition, angle, dx, dy in conditions:
        preds = {method: {} for method in ['No alignment', 'ORB + RANSAC', 'SIFT + RANSAC']}
        timings = {method: 0. for method in preds}
        failures = {method: 0 for method in preds}
        errors = {method: [] for method in preds}
        for rec in subset:
            t, s = load_pair(rec)
            shifted, H_forward = perturb_image(s, angle, dx, dy)
            for method in preds:
                start = time.perf_counter()
                if method == 'No alignment':
                    pp = detect(t, shifted, model, config, threshold)
                else:
                    aligned, valid, info = register_pair(t, shifted, method.split()[0])
                    if not info['success']:
                        failures[method] += 1; pp = []
                    else:
                        pp = detect(t, aligned, model, config, threshold, valid)
                        pts = np.float32([[80,80],[560,80],[560,560],[80,560]]).reshape(-1,1,2)
                        reconstructed = cv2.perspectiveTransform(pts, info['H'] @ H_forward)
                        errors[method].append(float(np.linalg.norm(reconstructed-pts, axis=2).mean()))
                timings[method] += time.perf_counter()-start
                preds[method][rec['id']] = pp
        for method in preds:
            _, metrics = score_predictions(subset, preds[method], .5)
            table.append(dict(condition=condition, method=method, **metrics,
                              registration_failures=failures[method],
                              milliseconds_per_image=1000*timings[method]/len(subset),
                              mean_reprojection_error_px=float(np.mean(errors[method])) if errors[method] else None))
        detail.append(dict(condition=condition, predictions=preds))
    return pd.DataFrame(table), [r['id'] for r in subset]

# %% visuals
def save_figure(fig, root, filename):
    directory = Path(root) / 'assets'
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close(fig)


def draw_boxes(ax, image, boxes, labels, title, scores=None):
    ax.imshow(image, cmap='gray', vmin=0, vmax=255)
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1,y1,x2,y2 = box
        color = COLORS[int(label)]
        ax.add_patch(Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=1.5))
        caption = CLASSES[int(label)]
        if scores is not None:
            caption += f' {scores[i]:.2f}'
        ax.text(x1, max(y1-3, 8), caption, color='white', fontsize=7,
                bbox=dict(facecolor=color, alpha=.85, edgecolor='none', pad=1))
    ax.set_title(title); ax.axis('off')


def show_pair(rec, root):
    t, s = load_pair(rec); gt = ground_truth(rec)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(t, cmap='gray'); axes[0].axis('off'); axes[0].set_title('Defect-free template')
    draw_boxes(axes[1], s, gt[:,:4], gt[:,4], 'Inspected board: ground truth')
    axes[2].imshow(cv2.absdiff(t,s), cmap='inferno'); axes[2].axis('off'); axes[2].set_title('Absolute difference')
    fig.suptitle(f'DeepPCB pair {rec["id"]} | Source: Tang et al., DeepPCB', fontsize=13)
    fig.tight_layout(); save_figure(fig, root, '01_dataset_pair.png')


def show_predictions(rec, predictions, root, filename='05_predictions.png'):
    t, s = load_pair(rec); gt = ground_truth(rec)
    pp = predictions[rec['id']]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    draw_boxes(axes[0], s, gt[:,:4], gt[:,4], 'Ground truth')
    draw_boxes(axes[1], s, [p['box'] for p in pp], [p['label'] for p in pp],
               'Predictions (no labels used at inference)', [p['score'] for p in pp])
    fig.suptitle(f'Test image {rec["id"]} | Source imagery: DeepPCB'); fig.tight_layout()
    save_figure(fig, root, filename)


def inspect_custom(template_path, tested_path, model, config, threshold, method='ORB'):
    """Inference accepts only images, no annotation file; returns boxes in template coordinates."""
    t = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    s = cv2.imread(str(tested_path), cv2.IMREAD_GRAYSCALE)
    if t is None or s is None or t.shape != s.shape:
        raise ValueError('Provide readable, equally sized grayscale-compatible image pairs.')
    aligned, valid, info = register_pair(t, s, method)
    if not info['success']:
        return dict(status='registration_failed', reason=info['reason'], predictions=[])
    predictions = detect(t, aligned, model, config, threshold, valid)
    return dict(status='ok', coordinate_frame='template', predictions=predictions,
                aligned=aligned, metadata=info)
