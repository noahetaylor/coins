"""
train_date_reader_v1.py
=======================
Phase 1 baseline: read the whole date band directly, no digit boxes.

Every band in the DateBands manifest carries a known 4-digit year, so this
model trains with zero additional labeling. Its job is to answer one
question: do we need a digit detector at all?

    strong result  -> skip the detector, go straight to assembly
    weak result    -> fall back to box labeling (Phase 2)

WHAT IT PREDICTS
----------------
Four independent 10-way heads, one per digit position. Positions 0 and 1
are constant ('1','9') across 1965-1998, so they are trivially learnable
and are reported only as a sanity check - if head 0 is not ~100%, the
plumbing is broken. The real task is:

    pos2  decade digit   4 classes actually present (6,7,8,9)
    pos3  unit digit    10 classes, thinnest is '4'

Headline metrics are pos2 accuracy, pos3 accuracy, and full-year accuracy
(all four correct). Year accuracy is roughly pos2*pos3, so it is always
the lowest number - that is expected, not a bug.

THE SPLIT IS THE WHOLE BALLGAME
-------------------------------
Split is by `coin_id` (physical coin), never by image. 2026data
contributes 306 bands from just 51 coins - the same coin under different
lighting - so an image-level split would put near-duplicates in both
train and val and report accuracy that means nothing. Coins are
stratified on the unit digit so rare classes appear on both sides.

Bicentennial coins (dual date 1776-1976) do not fit the 4-digit
assumption and are excluded here; they need their own branch.

USAGE
-----
    python train_date_reader_v1.py                    # train
    python train_date_reader_v1.py --epochs 40
    python train_date_reader_v1.py --eval-only models/date_reader_v1.pth
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

IMPORTANT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = IMPORTANT_DIR.parents[1]
BANDS_ROOT = PROJECT_ROOT / "data" / "coins" / "DateBands"
MANIFEST = BANDS_ROOT / "manifest.json"
MODELS_DIR = IMPORTANT_DIR / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN = DEVICE.type == "cuda"
DUAL_DATE = "1776-1976"

# Bands are ~2.64:1 under the corrected harvest geometry (half-width 0.58,
# top 0.58, bottom 1.02). Keep that aspect so digits are not squashed.
# Overridable with --in-size WxH. A 6-point resolution sweep on 2026-08-29
# (5-fold CV, wear-aug, see models/resolution_sweep/) found 448x168 is the
# peak: +0.74% pooled per-band over 320x120, and accuracy falls off past it
# (1280x480 is -3.1%, with 3x the fold variance) - more pixels feeds the
# augmentation pipeline more room to distort, not more signal.
IN_W, IN_H = 448, 168
N_POS, N_CLS = 4, 10


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_records():
    recs = json.loads(MANIFEST.read_text())
    return [r for r in recs if r.get("digits") and r["digits"] != DUAL_DATE]


def split_by_coin(records, val_frac=0.2, seed=0):
    """
    Group-stratified split. Groups are physical coins; stratification is on
    the unit digit so thin classes ('4') land on both sides.
    """
    by_coin = defaultdict(list)
    for r in records:
        by_coin[r["coin_id"]].append(r)

    strata = defaultdict(list)
    for coin, rs in by_coin.items():
        strata[rs[0]["digits"][3]].append(coin)

    rng = random.Random(seed)
    train_coins, val_coins = set(), set()
    for digit, coins in sorted(strata.items()):
        coins = sorted(coins)
        rng.shuffle(coins)
        n_val = max(1, round(len(coins) * val_frac))
        val_coins.update(coins[:n_val])
        train_coins.update(coins[n_val:])

    train = [r for r in records if r["coin_id"] in train_coins]
    val = [r for r in records if r["coin_id"] in val_coins]
    return train, val, train_coins, val_coins


def k_fold_by_coin(records, k=5, seed=0):
    """
    K coin-level folds, stratified on the unit digit.

    Every coin lands in exactly one validation fold, so pooling the folds'
    predictions gives an out-of-fold prediction for ALL coins rather than
    the ~150 a single split can measure. That is the point: one 20% split
    resolves accuracy to about +/-0.7% per coin, which is too coarse to
    tell a real improvement from noise.
    """
    by_coin = defaultdict(list)
    for r in records:
        by_coin[r["coin_id"]].append(r)

    strata = defaultdict(list)
    for coin, rs in by_coin.items():
        strata[rs[0]["digits"][3]].append(coin)

    rng = random.Random(seed)
    assign = {}
    for digit, coins in sorted(strata.items()):
        coins = sorted(coins)
        rng.shuffle(coins)
        # Deal this stratum round-robin so each fold gets a share of it.
        for i, coin in enumerate(coins):
            assign[coin] = i % k

    folds = []
    for f in range(k):
        val_coins = {c for c, i in assign.items() if i == f}
        train = [r for r in records if r["coin_id"] not in val_coins]
        val = [r for r in records if r["coin_id"] in val_coins]
        folds.append((train, val, val_coins))
    return folds


class DynamicResize:
    """Resize to the current IN_H/IN_W.

    The Compose objects below are built at import time, so a plain
    transforms.Resize would bake in whatever the size was then and silently
    ignore --in-size. This reads the module globals at call time instead.
    """

    def __call__(self, img):
        return transforms.functional.resize(img, [IN_H, IN_W])


TRAIN_TF = transforms.Compose([
    DynamicResize(),
    # No horizontal flip - it would mirror the digits.
    transforms.RandomAffine(degrees=6, translate=(0.03, 0.06),
                            scale=(0.94, 1.06), shear=4, fill=0),
    transforms.ColorJitter(brightness=0.45, contrast=0.45,
                           saturation=0.45, hue=0.12),
    transforms.RandomGrayscale(p=0.25),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.08)),
])

# Wear / studio simulation.
#
# Cross-validation showed every error lands on 2026data (86.3% per coin)
# while pcgs, crops and supers all sit at 100%. The cause is not sampling
# frequency - 2026data is already 28.4% of training bands - it is coin
# DIVERSITY: only 51 of 751 coins come from the studio rig, and those are
# circulated pocket change while the scraped sources are mostly graded,
# uncirculated coins with crisp relief.
#
# Oversampling 2026data would just replay the same 51 coins. Instead,
# degrade the 686 pristine coins toward what the rig actually produces:
# soft relief, low contrast, and heavy colour casts from the LED sweep.
# That turns abundant pristine data into training signal for the domain
# that actually matters.
WEAR_TF = transforms.Compose([
    DynamicResize(),
    transforms.RandomAffine(degrees=6, translate=(0.03, 0.06),
                            scale=(0.94, 1.06), shear=4, fill=0),
    # Soften relief the way a worn coin and a soft rig image both do.
    transforms.RandomApply(
        [transforms.GaussianBlur(3, sigma=(0.4, 1.5))], p=0.55),
    # Contrast is skewed downward, not symmetric - worn dates lose contrast,
    # they do not gain it. hue at the limit reproduces the LED colour sweep.
    transforms.ColorJitter(brightness=0.5, contrast=(0.3, 1.15),
                           saturation=0.6, hue=0.5),
    transforms.RandomGrayscale(p=0.25),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    # Deliberately NO RandomErasing here. A patch that lands on a digit
    # hides the evidence while still demanding the right answer, which
    # trains the model to guess from context - exactly the observed failure
    # mode, where all 6 bands of a coin agree on the same wrong year.
])

EVAL_TF = transforms.Compose([
    DynamicResize(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Set from main(); read by BandDataset in worker processes after fork.
ACTIVE_TRAIN_TF = TRAIN_TF


class BandDataset(Dataset):
    def __init__(self, records, train):
        self.records = records
        self.tf = ACTIVE_TRAIN_TF if train else EVAL_TF

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(BANDS_ROOT / "bands" / r["band"]).convert("RGB")
        target = torch.tensor([int(c) for c in r["digits"]], dtype=torch.long)
        return self.tf(img), target


def unit_digit_sampler(records):
    """Oversample coins whose unit digit is rare, so '4' is not drowned out."""
    counts = Counter(r["digits"][3] for r in records)
    weights = [1.0 / counts[r["digits"][3]] for r in records]
    return WeightedRandomSampler(weights, num_samples=len(records), replacement=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def save_checkpoint(model, path):
    """Bundle in_size with the weights so a resolution change can't silently
    desync a checkpoint from the preprocessing that reads it."""
    torch.save({"state_dict": model.state_dict(), "in_size": [IN_W, IN_H]}, path)


def load_checkpoint(model, path, device=None):
    """Returns the (w, h) the checkpoint was trained at, or None for legacy
    raw-state-dict checkpoints saved before in_size was tracked."""
    ck = torch.load(path, map_location=device or DEVICE, weights_only=False)
    if isinstance(ck, dict) and "state_dict" in ck:
        model.load_state_dict(ck["state_dict"])
        return ck.get("in_size")
    model.load_state_dict(ck)
    return None


class DateReader(nn.Module):
    """
    ResNet-18 trunk with COLUMN-ALIGNED heads.

    Global average pooling would collapse the band into a single vector,
    forcing the trunk to encode "which digit is where" implicitly - a bad
    inductive bias for reading a left-to-right sequence, and one that would
    understate what this architecture can do.

    Instead the feature map is pooled to a width of N_POS columns, so head
    k reads the k-th horizontal slice of the band. Digits sit on a slight
    arc, so each column still sees some of its neighbours - that overlap is
    tolerated deliberately rather than hard-cropped.
    """

    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)
        in_f = base.fc.in_features
        # Drop avgpool + fc, keep the conv feature map.
        self.trunk = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, N_POS))
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList(nn.Linear(in_f, N_CLS) for _ in range(N_POS))

    def forward(self, x):
        f = self.pool(self.trunk(x)).squeeze(2)        # (B, C, N_POS)
        f = self.drop(f)
        return torch.stack([h(f[:, :, i]) for i, h in enumerate(self.heads)], dim=1)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def evaluate(model, loader):
    model.eval()
    pos_correct = np.zeros(N_POS, dtype=np.int64)
    year_correct = total = 0
    confusion = np.zeros((N_CLS, N_CLS), dtype=np.int64)   # unit digit
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=PIN), y.to(DEVICE, non_blocking=PIN)
            pred = model(x).argmax(dim=2)
            hit = (pred == y)
            pos_correct += hit.sum(dim=0).cpu().numpy()
            year_correct += hit.all(dim=1).sum().item()
            total += y.size(0)
            for t, p in zip(y[:, 3].tolist(), pred[:, 3].tolist()):
                confusion[t, p] += 1
    return pos_correct / total, year_correct / total, total, confusion


def print_confusion(cm):
    print("\n  unit-digit confusion (rows = true, cols = predicted)")
    print("      " + "".join(f"{c:>5}" for c in range(N_CLS)) + "     acc   n")
    for t in range(N_CLS):
        n = cm[t].sum()
        acc = cm[t, t] / n if n else float("nan")
        row = "".join(f"{cm[t, p]:>5}" if cm[t, p] else "    ." for p in range(N_CLS))
        flag = "  <-- weak" if n and acc < 0.75 else ""
        print(f"   {t}: {row}  {acc:>6.1%} {n:>4}{flag}")


def predict_records(model, records, workers=4, batch_size=32):
    """Predicted 4-digit string for each record, in order."""
    model.eval()
    loader = DataLoader(BandDataset(records, False), batch_size,
                        shuffle=False, num_workers=workers, pin_memory=PIN)
    out = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE, non_blocking=PIN)
            for row in model(x).argmax(dim=2):
                out.append("".join(str(int(d)) for d in row))
    return out


def coin_level_accuracy(records, predictions):
    """Majority vote over each coin's bands - the production read."""
    votes = defaultdict(list)
    for r, p in zip(records, predictions):
        votes[r["coin_id"]].append((r["digits"], p))
    correct, wrong = 0, []
    for coin, items in votes.items():
        truth = items[0][0]
        maj = Counter(p for _, p in items).most_common(1)[0][0]
        if maj == truth:
            correct += 1
        else:
            wrong.append((coin, truth, maj, len(items)))
    return correct / len(votes), wrong, len(votes)


def train_one(train_recs, val_recs, args, tag=""):
    """Train from scratch, return (best_year_acc, best_state, best_preds)."""
    model = DateReader().to(DEVICE)
    train_loader = DataLoader(BandDataset(train_recs, True), args.batch_size,
                              sampler=unit_digit_sampler(train_recs),
                              num_workers=args.workers, pin_memory=PIN)
    val_loader = DataLoader(BandDataset(val_recs, False), args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=PIN)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, best_state = -1.0, None
    opt = sched = None

    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        for p in model.trunk.parameters():
            p.requires_grad = not frozen
        params = [p for p in model.parameters() if p.requires_grad]
        if epoch == 1 or epoch == args.freeze_epochs + 1:
            lr = args.lr * (3 if frozen else 1)
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, (args.freeze_epochs if frozen
                                   else args.epochs - args.freeze_epochs)))

        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=PIN), y.to(DEVICE, non_blocking=PIN)
            opt.zero_grad()
            out = model(x)
            loss = sum(criterion(out[:, p], y[:, p]) for p in range(N_POS))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 2.0)
            opt.step()
            running += loss.item() * x.size(0)
        sched.step()

        pos, year, _, _ = evaluate(model, val_loader)
        if year > best:
            best = year
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if args.verbose:
            print(f"  {tag}epoch {epoch:>3}/{args.epochs} "
                  f"[{'frozen' if frozen else 'full  '}] "
                  f"loss {running/len(train_recs):.3f}  "
                  f"pos2 {pos[2]:.1%} pos3 {pos[3]:.1%}  year {year:.1%}")

    model.load_state_dict(best_state)
    return best, best_state, predict_records(model, val_recs, args.workers,
                                             args.batch_size)


def run_cross_validation(records, args):
    folds = k_fold_by_coin(records, args.folds, args.seed)
    oof_recs, oof_preds, fold_band, fold_coin = [], [], [], []

    for i, (tr, va, val_coins) in enumerate(folds, 1):
        print(f"\nfold {i}/{args.folds}: train {len(tr)} bands / "
              f"{len({r['coin_id'] for r in tr})} coins  |  "
              f"val {len(va)} bands / {len(val_coins)} coins")
        band_acc, _, preds = train_one(tr, va, args, tag=f"f{i} ")
        coin_acc, wrong, n_coins = coin_level_accuracy(va, preds)
        fold_band.append(band_acc)
        fold_coin.append(coin_acc)
        oof_recs.extend(va)
        oof_preds.extend(preds)
        print(f"  -> band {band_acc:.1%}   coin {coin_acc:.1%} "
              f"({n_coins - len(wrong)}/{n_coins})")

    band = np.array(fold_band)
    coin = np.array(fold_coin)
    print("\n" + "=" * 62)
    print(f"{args.folds}-FOLD CROSS-VALIDATION")
    print("=" * 62)
    print(f"per-band year : {band.mean():.1%}  +/- {band.std():.1%}   "
          f"folds {', '.join(f'{b:.1%}' for b in band)}")
    print(f"per-coin year : {coin.mean():.1%}  +/- {coin.std():.1%}   "
          f"folds {', '.join(f'{c:.1%}' for c in coin)}")

    # Pooled out-of-fold: every coin now has a held-out prediction.
    acc, wrong, n = coin_level_accuracy(oof_recs, oof_preds)
    band_ok = sum(1 for r, p in zip(oof_recs, oof_preds) if p == r["digits"])
    print(f"\npooled out-of-fold over ALL data:")
    print(f"  bands {band_ok}/{len(oof_recs)} = {band_ok/len(oof_recs):.1%}")
    print(f"  coins {n - len(wrong)}/{n} = {acc:.1%}")

    by_source = defaultdict(lambda: [0, 0])
    for r, p in zip(oof_recs, oof_preds):
        by_source[r["source"]][1] += 1
        by_source[r["source"]][0] += (p == r["digits"])
    print("\n  per-source band accuracy:")
    for s, (ok, tot) in sorted(by_source.items()):
        print(f"    {s:<10} {ok:>4}/{tot:<4} = {ok/tot:.1%}")

    print(f"\n  coin-level failures ({len(wrong)}):")
    for coin_id, truth, got, nb in sorted(wrong):
        print(f"    {coin_id:<40} {truth} -> {got}  ({nb} bands)")

    out = args.out.with_name("cross_val_report.json")
    out.write_text(json.dumps({
        "folds": args.folds, "seed": args.seed,
        "in_w": IN_W, "in_h": IN_H, "wear_aug": bool(args.wear_aug),
        "epochs": args.epochs,
        "per_band_mean": float(band.mean()), "per_band_std": float(band.std()),
        "per_coin_mean": float(coin.mean()), "per_coin_std": float(coin.std()),
        "pooled_band_acc": band_ok / len(oof_recs), "pooled_coin_acc": acc,
        "failures": [{"coin_id": c, "true": t, "pred": g, "bands": n}
                     for c, t, g, n in sorted(wrong)],
    }, indent=2))
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--freeze-epochs", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "date_reader_v1.pth")
    ap.add_argument("--eval-only", type=Path)
    ap.add_argument("--folds", type=int, default=0,
                    help="run K-fold coin-level cross-validation instead of a "
                         "single split (5 is a sensible value)")
    ap.add_argument("--verbose", action="store_true",
                    help="print every epoch during cross-validation")
    ap.add_argument("--in-size", type=str, default=None, metavar="WxH",
                    help="model input resolution, e.g. 640x240 (default "
                         "320x120). Native bands are 1346-1674px wide.")
    ap.add_argument("--wear-aug", action="store_true",
                    help="degrade pristine coins toward studio-rig conditions "
                         "(blur, low contrast, heavy colour cast)")
    ap.add_argument("--bands-root", type=Path, default=None,
                    help="a DateBands variant (manifest.json + bands/), e.g. "
                         "one built by build_raking_bands.py")
    args = ap.parse_args()

    if args.bands_root:
        global BANDS_ROOT, MANIFEST
        BANDS_ROOT = args.bands_root.resolve()
        MANIFEST = BANDS_ROOT / "manifest.json"
        print(f"bands: {BANDS_ROOT}")

    if args.in_size:
        global IN_W, IN_H
        IN_W, IN_H = (int(v) for v in args.in_size.lower().split("x"))
        print(f"input resolution: {IN_W}x{IN_H}")

    if args.wear_aug:
        global ACTIVE_TRAIN_TF
        ACTIVE_TRAIN_TF = WEAR_TF
        print("augmentation: WEAR (studio-domain simulation)")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    records = load_records()

    if args.folds:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        run_cross_validation(records, args)
        return

    train_recs, val_recs, train_coins, val_coins = split_by_coin(
        records, args.val_frac, args.seed)

    overlap = train_coins & val_coins
    assert not overlap, f"coin leaked across split: {list(overlap)[:5]}"

    print(f"bands  train={len(train_recs)}  val={len(val_recs)}")
    print(f"coins  train={len(train_coins)}  val={len(val_coins)}  (no overlap)")
    val_units = Counter(r["digits"][3] for r in val_recs)
    print("val unit-digit coverage:", dict(sorted(val_units.items())))

    train_loader = DataLoader(BandDataset(train_recs, True), args.batch_size,
                              sampler=unit_digit_sampler(train_recs),
                              num_workers=args.workers, pin_memory=PIN)
    val_loader = DataLoader(BandDataset(val_recs, False), args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=PIN)

    model = DateReader().to(DEVICE)

    if args.eval_only:
        ck_size = load_checkpoint(model, args.eval_only)
        if ck_size and tuple(ck_size) != (IN_W, IN_H):
            print(f"warning: checkpoint trained at {ck_size[0]}x{ck_size[1]}, "
                  f"evaluating at {IN_W}x{IN_H} - pass matching --in-size")
        pos, year, n, cm = evaluate(model, val_loader)
        print(f"\npos2 {pos[2]:.1%}  pos3 {pos[3]:.1%}  year {year:.1%}  (n={n})")
        print_confusion(cm)
        return

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best = -1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Warm up the heads before letting gradients into the trunk.
        frozen = epoch <= args.freeze_epochs
        for p in model.trunk.parameters():
            p.requires_grad = not frozen
        params = [p for p in model.parameters() if p.requires_grad]
        if epoch == 1 or epoch == args.freeze_epochs + 1:
            lr = args.lr * (3 if frozen else 1)
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, (args.freeze_epochs if frozen
                                   else args.epochs - args.freeze_epochs)))

        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=PIN), y.to(DEVICE, non_blocking=PIN)
            opt.zero_grad()
            out = model(x)
            loss = sum(criterion(out[:, p], y[:, p]) for p in range(N_POS))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 2.0)
            opt.step()
            running += loss.item() * x.size(0)
        sched.step()

        pos, year, _, cm = evaluate(model, val_loader)
        tag = "frozen" if frozen else "full  "
        print(f"epoch {epoch:>3}/{args.epochs} [{tag}] "
              f"loss {running/len(train_recs):.3f}  "
              f"pos0 {pos[0]:.0%} pos1 {pos[1]:.0%} "
              f"pos2 {pos[2]:.1%} pos3 {pos[3]:.1%}  year {year:.1%}")

        if year > best:
            best = year
            save_checkpoint(model, args.out)
            split_meta = {"train_coins": sorted(train_coins),
                          "val_coins": sorted(val_coins),
                          "seed": args.seed, "val_frac": args.val_frac,
                          "best_year_acc": year}
            args.out.with_suffix(".split.json").write_text(json.dumps(split_meta, indent=2))

    print(f"\nbest year accuracy {best:.1%} -> {args.out}")
    load_checkpoint(model, args.out)
    pos, year, n, cm = evaluate(model, val_loader)
    print(f"final: pos2 {pos[2]:.1%}  pos3 {pos[3]:.1%}  year {year:.1%}  (n={n})")
    print_confusion(cm)

    print("\nPhase 1 verdict:")
    if pos[3] >= 0.90:
        print("  STRONG - skip the detector, proceed to assembly (Phase 3)")
    elif pos[3] >= 0.75:
        print("  MARGINAL - try more epochs / larger input before Phase 2")
    else:
        print("  WEAK - digit localization (Phase 2) is needed")


if __name__ == "__main__":
    main()
