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

DEVICE = torch.device("cpu")        # the 3060's fan is dead - CPU only
DUAL_DATE = "1776-1976"

# Bands are ~2.88:1. Keep that aspect so digits are not squashed.
IN_W, IN_H = 320, 112
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


TRAIN_TF = transforms.Compose([
    transforms.Resize((IN_H, IN_W)),
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

EVAL_TF = transforms.Compose([
    transforms.Resize((IN_H, IN_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class BandDataset(Dataset):
    def __init__(self, records, train):
        self.records = records
        self.tf = TRAIN_TF if train else EVAL_TF

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
            pred = model(x).argmax(dim=2)
            hit = (pred == y)
            pos_correct += hit.sum(dim=0).numpy()
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
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    records = load_records()
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
                              num_workers=args.workers)
    val_loader = DataLoader(BandDataset(val_recs, False), args.batch_size,
                            shuffle=False, num_workers=args.workers)

    model = DateReader().to(DEVICE)

    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_only, map_location=DEVICE))
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
            torch.save(model.state_dict(), args.out)
            split_meta = {"train_coins": sorted(train_coins),
                          "val_coins": sorted(val_coins),
                          "seed": args.seed, "val_frac": args.val_frac,
                          "best_year_acc": year}
            args.out.with_suffix(".split.json").write_text(json.dumps(split_meta, indent=2))

    print(f"\nbest year accuracy {best:.1%} -> {args.out}")
    model.load_state_dict(torch.load(args.out, map_location=DEVICE))
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
