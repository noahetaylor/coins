#!/usr/bin/env python3
# beale_scan.py
# Ready-to-run Beale book-cipher key scanner (standard library only).

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from glob import glob
from multiprocessing import Pool, cpu_count
from typing import List, Dict

# -------------------------
# English models and helpers
# -------------------------

# Small-but-useful trigram log-probabilities (UPPERCASE, letters only).
# Values are approximate log-probs; unseen grams use TRI_FLOOR.
TRIGRAM_LOGP: Dict[str, float] = {
    "THE": -2.2, "AND": -2.7, "ING": -3.0, "HER": -3.5, "HAT": -3.7, "THA": -3.7,
    "ERE": -3.4, "ENT": -3.4, "ION": -3.5, "TIO": -3.8, "FOR": -3.7, "NTH": -3.8,
    "ATI": -3.8, "VER": -3.9, "TER": -3.7, "HIS": -3.6, "EST": -3.8, "ALL": -3.9,
    "RES": -4.0, "ONT": -4.2, "ETH": -3.9, "STA": -4.2, "ARE": -3.9, "ITH": -3.6,
    "ECT": -4.3, "ONS": -4.2, "OUR": -4.1, "MEN": -4.3, "WIT": -4.0, "NOT": -4.2,
    "PRO": -4.4, "COM": -4.4, "CON": -4.2, "TED": -4.3, "YOU": -4.4, "WAS": -4.4,
    "THI": -3.6, "OUS": -4.1, "OFT": -4.3, "RED": -4.6, "ERA": -4.4,
}
TRI_FLOOR = -6.5  # fall-back log-prob for unseen trigrams

# English letter frequency (A..Z) as percentages; used for chi-squared scoring.
ENG_FREQ: Dict[str, float] = {
    'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97, 'N': 6.75, 'S': 6.33,
    'H': 6.09, 'R': 5.99, 'D': 4.25, 'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41,
    'W': 2.36, 'F': 2.23, 'G': 2.02, 'Y': 1.97, 'P': 1.93, 'B': 1.49, 'V': 0.98,
    'K': 0.77, 'J': 0.15, 'X': 0.15, 'Q': 0.10, 'Z': 0.07
}

# Common short word list for quick substring probes (UPPERCASE).
COMMON_WORDS = set((
    "THE","OF","AND","TO","IN","A","THAT","IS","FOR","IT","AS","WITH","HIS","ON","BE","AT","BY",
    "HE","FROM","THIS","I","YOU","OR","NOT","ARE","ALL","ONE","HAVE","WAS","BUT","WE","THEY","AN",
    "WHICH","THERE","THEIR","WHAT","WHEN","WHO","HER","HIM","OUR","YOUR","MY","ME","THEM","ITS",
    "WILL","WOULD","CAN","COULD","SHALL","SHOULD","MAY","MIGHT","UPON","INTO","ABOUT","OVER",
    "UNDER","BETWEEN","WITHIN","WITHOUT","AGAINST","AMONG","TOWARD","BEFORE","AFTER","ABOVE","BELOW",
    "NORTH","SOUTH","EAST","WEST","IRON","STONE","VAULT","GOLD","SILVER","COPPER","CHEST","BOX",
    "CAVE","ROCK","RIVER","HILL","VALLEY","COUNTY","VIRGINIA","MILES","CHAIN","CHAINS","DEEP",
))

def clean_letters(s: str) -> str:
    return re.sub(r'[^A-Z]', '', s.upper())

def score_trigrams(s: str) -> float:
    s = clean_letters(s)
    if len(s) < 3:
        return -1e9
    total = 0.0
    for i in range(len(s) - 2):
        gram = s[i:i+3]
        total += TRIGRAM_LOGP.get(gram, TRI_FLOOR)
    return total / (len(s) - 2)

def score_letter_chisq(s: str) -> float:
    s = clean_letters(s)
    if not s:
        return -1e9
    counts = Counter(s)
    n = len(s)
    chisq = 0.0
    for ch, pct in ENG_FREQ.items():
        expected = pct / 100.0 * n
        observed = counts.get(ch, 0)
        if expected > 0:
            chisq += (observed - expected) ** 2 / expected
    return -chisq  # lower is better, so negate

def score_common_word_hits(s: str) -> float:
    s = clean_letters(s)
    if len(s) < 30:
        return 0.0
    hits = 0
    total = 0
    for w in COMMON_WORDS:
        if 3 <= len(w) <= 8:
            total += 1
            if w in s:
                hits += 1
    return hits / max(1, total)

def health_penalties(s: str) -> float:
    bad = s.count('?')
    non = sum(1 for c in s if not (c.isalpha() or c == '?'))
    return -5.0 * bad - 0.25 * non

def combined_score(s: str) -> float:
    return (
        1.5 * score_trigrams(s) +
        0.02 * score_letter_chisq(s) +
        1.0 * score_common_word_hits(s) +
        health_penalties(s)
    )

# -------------------------
# Normalization / tokenization
# -------------------------

import re
import sys
import os
from typing import List, Dict
from glob import glob

# Match common punctuation for stripping/splitting.
punct_chars = """.,;:!?'"(){}[]“”‘’—–-"""
PUNCT_RE = re.compile(f"[{re.escape(punct_chars)}]")

def normalize(text: str, variant: str) -> List[str]:
    if variant == "V0":
        tokens = text.split()
    elif variant == "V1":
        tokens = PUNCT_RE.sub(" ", text).split()
    elif variant == "V2":
        text2 = re.sub(r"([A-Za-z])['\"”’]+", r"\1", text)
        tokens = re.sub(f"[{re.escape(punct_chars)}]", " ", text2).split()
    elif variant == "V3":
        text2 = re.sub(r"[-—–]", " ", text)
        tokens = re.sub(f"[{re.escape(punct_chars)}]", " ", text2).split()
    elif variant == "V4":
        STOP = set("the of and to in a that is for it as with his on be at by he from this".split())
        tokens = [t for t in PUNCT_RE.sub(" ", text).split() if t.lower() not in STOP]
    else:
        tokens = text.split()
    return tokens

def normalize_corpus(corpus: Dict[str, str], variant: str) -> List[str]:
    """Flatten and normalize all corpus texts into a single token list."""
    all_tokens = []
    for text in corpus.values():
        tokens = normalize(text, variant)
        all_tokens.extend(tokens)
    return all_tokens

def first_alpha_letter(word: str) -> str:
    m = re.search(r"[A-Za-z]", word)
    return m.group(0).upper() if m else '?'

# -------------------------
# Decoding
# -------------------------

def decode_numbers(cipher_nums: List[int], tokens: List[str], offset: int = 0) -> str:
    letters = []
    n_tokens = len(tokens)
    for n in cipher_nums:
        idx = n + offset  # book ciphers are 1-based by construction
        if 1 <= idx <= n_tokens:
            letters.append(first_alpha_letter(tokens[idx - 1]))
        else:
            letters.append('?')
    return "".join(letters)

def visualize_alignment(cipher: List[int], corpus_words: List[str], offset: int = 0, max_rows: int = 50):
    print("\n=== Cipher Alignment Visualization ===")
    print(f"{'Index':>5} | {'Cipher #':>8} | {'Word':<20} | {'Letter':>6}")
    print("-" * 50)
    for i, num in enumerate(cipher[:max_rows]):
        word_index = num - 1 + offset
        if 0 <= word_index < len(corpus_words):
            word = corpus_words[word_index]
            letter = first_alpha_letter(word)
        else:
            word = "<out of bounds>"
            letter = "?"
        print(f"{i+1:>5} | {num:>8} | {word:<20} | {letter:>6}")

# -------------------------
# I/O and orchestration
# -------------------------

def read_cipher_numbers(path: str) -> List[int]:
    try:
        raw = open(path, 'r', encoding='utf-8', errors='ignore').read()
    except FileNotFoundError:
        print(f"[!] Cipher file not found: {path}", file=sys.stderr)
        return []
    nums = re.findall(r"\d+", raw)
    return [int(x) for x in nums]

def load_corpus(corpus_dir: str, min_chars: int = 1000) -> Dict[str, str]:
    files: List[str] = []
    files.extend(glob(os.path.join(corpus_dir, "**", "*.txt"), recursive=True))
    corpus: Dict[str, str] = {}
    for fp in files:
        try:
            txt = open(fp, 'r', encoding='utf-8', errors='ignore').read()
            if len(txt) >= min_chars:
                corpus[fp] = txt
        except Exception:
            continue
    return corpus

def parse_offsets(spec: str) -> List[int]:
    spec = spec.strip()
    if ":" in spec:
        a, b = spec.split(":")
        a, b = int(a), int(b)
        return list(range(a, b + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]

def evaluate_task(args):
    text_id, text, ciphers, variant, offsets, early_cut, min_score = args
    tokens = normalize(text, variant)


    best = {
        "text_id": text_id,
        "variant": variant,
        "best_offset": None,
        "score_sum": -1e99,
        "scores": [],
        "previews": [],
    }
    for off in offsets:
        cipher_scores = []
        cipher_previews = []
        score_sum = 0.0
        doomed = False
        for cipher in ciphers:
            decoded = decode_numbers(cipher, tokens, offset=off)
            sample = decoded[:early_cut] if early_cut else decoded
            s = combined_score(sample)
            if s < min_score:
                doomed = True
                break
            s_full = combined_score(decoded)
            cipher_scores.append(s_full)
            cipher_previews.append(decoded[:200])
            score_sum += s_full
        if doomed:
            continue
        if score_sum > best["score_sum"]:
            best["best_offset"] = off
            best["score_sum"] = score_sum
            best["scores"] = cipher_scores
            best["previews"] = cipher_previews
    return best

def rank_candidates(corpus: Dict[str, str], ciphers: List[List[int]], variants: List[str],
                    offsets: List[int], workers: int, top_k: int,
                    early_cut: int, min_score: float, progress_every: int = 50):
    tasks = []
    for text_id, text in corpus.items():
        for v in variants:
            tasks.append((text_id, text, ciphers, v, offsets, early_cut, min_score))
    total = len(tasks)
    results = []
    start = time.time()
    if workers == 0:
        workers = max(1, cpu_count() - 1)
    print(f"[i] Tasks: {total} (texts × variants), workers: {workers}")
    with Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap_unordered(evaluate_task, tasks, chunksize=2), 1):
            # res might have no valid offset; still collect for filtering
            results.append(res)
            if i % progress_every == 0:
                dt = time.time() - start
                print(f"[i] {i}/{total} tasks done in {dt:.1f}s")
    results = [r for r in results if r.get("best_offset") is not None]
    results.sort(key=lambda r: r["score_sum"], reverse=True)
    return results[:top_k]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_results(results, out_dir: str, cipher_files: List[str]):
    ensure_dir(out_dir)
    jl_path = os.path.join(out_dir, "top_hits.jsonl")
    csv_path = os.path.join(out_dir, "top_hits.csv")

    with open(jl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    headers = ["rank","text_id","variant","best_offset","score_sum"]
    for i in range(len(cipher_files)):
        headers.append(f"score_cipher_{i+1}")
        headers.append(f"preview_cipher_{i+1}")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for rank, r in enumerate(results, 1):
            row = [rank, r["text_id"], r["variant"], r["best_offset"], f"{r['score_sum']:.3f}"]
            for i, s in enumerate(r["scores"]):
                row.append(f"{s:.3f}")
                row.append(r["previews"][i])
            writer.writerow(row)

    print(f"[✓] Saved: {jl_path}")
    print(f"[✓] Saved: {csv_path}")

def print_top_table(results, cipher_files: List[str], max_rows: int = 20):
    n = min(max_rows, len(results))
    if n == 0:
        print("[!] No promising results met the threshold.")
        return
    print("\n=== Top candidates ===")
    for i in range(n):
        r = results[i]
        print(f"\n#{i+1}  score_sum={r['score_sum']:.3f}   offset={r['best_offset']}   text={r['text_id']}   variant={r['variant']}")
        for j, cf in enumerate(cipher_files):
            print(f"  - Cipher {j+1} ({os.path.basename(cf)}): score={r['scores'][j]:.3f}")
            print(f"    Preview: {r['previews'][j]}")

def visualize_alignment(cipher: List[int], corpus_words: List[str], offset: int = 0, max_rows: int = 50):
    print("\n=== Cipher Alignment Visualization ===")
    print(f"{'Index':>5} | {'Cipher #':>8} | {'Word':<20} | {'Letter':>6}")
    print("-" * 50)
    for i, num in enumerate(cipher[:max_rows]):
        word_index = num - 1 + offset  # book ciphers are usually 1-based
        if 0 <= word_index < len(corpus_words):
            word = corpus_words[word_index]
            letter = first_alpha_letter(word)
        else:
            word = "<out of bounds>"
            letter = "?"
        print(f"{i+1:>5} | {num:>8} | {word:<20} | {letter:>6}")


def main():
    # Hardcoded values
    cipher_files = ["/home/noah/projects/image-identifier/data/coins/beale_project/cipher2.txt"]
    corpus_dir = "image-identifier/data/coins/beale_project/corpus"
    variants = ["V0", "V1", "V2", "V3", "V4"]
    offsets = [0]  # equivalent to "0"
    top_k = 50
    workers = 0
    early_cut = 250
    min_score = -180.0
    out_dir = "out"

    # Load cipher numbers
    ciphers: List[List[int]] = []
    for p in cipher_files:
        nums = read_cipher_numbers(p)
        if not nums:
            print(f"[!] Skipping empty or unreadable cipher: {p}", file=sys.stderr)
            continue
        ciphers.append(nums)
    if not ciphers:
        print("[!] No valid ciphers loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    total_nums = sum(len(c) for c in ciphers)
    print(f"[i] Loaded {len(ciphers)} cipher(s), {total_nums} numbers total.")

    # Load corpus
    corpus = load_corpus(corpus_dir)
    print(f"[i] Loaded corpus: {len(corpus)} text files from {corpus_dir}")
    if not corpus:
        print("[!] Corpus directory has no suitable .txt files (or all are too small).", file=sys.stderr)
        sys.exit(1)

    # Use the first corpus file for normalization and visualization
    text_path = list(corpus.keys())[0]
    text = corpus[text_path]

    # Normalize using variant V2
    normalized_tokens = normalize(text, "V2")
    print(f"[i] Normalized text contains {len(normalized_tokens)} words.")
    print(f"[i] Max cipher number: {max(ciphers[0])}")

    # Visualize alignment using the same normalized tokens
    visualize_alignment(ciphers[0], normalized_tokens, offset=0)

    print("\n=== Diagnostic Alignment Check ===")
    for variant in ["V0", "V1", "V2"]:
        tokens = normalize(text, variant)
        for offset in [-1, 0, 1]:
            print(f"\n--- Variant: {variant}, Offset: {offset} ---")
            visualize_alignment(ciphers[0], tokens, offset=offset, max_rows=30)


    # Parse offsets
    parsed_offsets = parse_offsets(",".join(str(o) for o in offsets))

    # Run analysis
    start = time.time()
    results = rank_candidates(
        corpus=corpus,
        ciphers=ciphers,
        variants=variants,
        offsets=parsed_offsets,
        workers=workers,
        top_k=top_k,
        early_cut=early_cut,
        min_score=min_score
    )
    dt = time.time() - start
    print(f"\n[i] Completed in {dt:.1f}s. {len(results)} candidates retained.")

    print_top_table(results, cipher_files)
    save_results(results, out_dir, cipher_files)



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted.", file=sys.stderr)
        sys.exit(1)
