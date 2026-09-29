"""Shared v2 blocking features and challenge scoring.

Only supplied record fields are used. Keep this module identical when v2 is
eventually promoted to submission inference.
"""

from __future__ import annotations

import re
import os
import unicodedata

import numpy as np
from rapidfuzz import fuzz, process
from scipy.sparse import csr_matrix


CPU_THREADS = os.cpu_count() or 1


STOP_WORDS = {
    "inc", "corp", "corporation", "ltd", "limited", "pvt", "private",
    "llc", "llp", "co", "company", "the", "and", "of", "in", "at",
    "on", "for", "to", "a", "an", "sa", "sas", "sarl", "eurl",
    "enterprises", "solutions", "services", "group", "holdings",
    "technologies", "industries", "international",
}
ADDR_STOP_WORDS = {
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln",
    "drive", "dr", "nagar", "colony", "marg", "near", "opp",
    "opposite", "behind", "floor", "flr", "shop", "no", "plot",
    "sector", "sec", "block", "blk", "phase", "delhi", "mumbai",
    "india", "us", "usa", "france", "paris", "city", "state",
    "town", "post", "building", "bldg", "complex", "apartment",
    "apt", "west", "east", "north", "south", "new", "old",
}

BASE_FEATURES = (
    "fuzz_ratio_name", "fuzz_tsr_name", "fuzz_tset_name",
    "name_word_jaccard", "name_len_diff_ratio", "fuzz_ratio_addr",
    "fuzz_tsr_addr", "fuzz_tset_addr", "addr_word_jaccard",
    "shared_numeric_code_count", "has_shared_numeric_code",
    "missing_address_flag", "non_latin_script_flag", "domain_stem_match",
    "exact_clean_name_match",
)
EXTRA_FEATURES = (
    "name_trigram_jaccard", "addr_trigram_jaccard",
    "address_number_conflict", "addr_length_ratio", "target_is_s3",
)
FEATURES = BASE_FEATURES + EXTRA_FEATURES
ABLATION_GROUPS = {
    "name_trigram": (15,),
    "address_trigram_and_length": (16, 18),
    "number_conflict": (17,),
    "target_source": (19,),
    "legacy_flags": (12, 13, 14),
}
MAX_CANDIDATES_PER_SOURCE = 32
FEATURE_ENGINE_VERSION = "vectorized_v1"
# Bound temporary set and sparse-matrix work; only the float32 result spans the caller's batch.
SCORE_WORKSPACE_ROWS = 2048
KEY_S1_LIMITS = {"n": 512, "t": 64, "a": 64, "w": 16}
KEY_TARGET_LIMIT = 2048


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "").casefold()
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def tokens(value: str, stop_words: set[str], min_length: int) -> set[str]:
    return {word for word in normalize(value).split() if len(word) >= min_length and word not in stop_words}


def numbers(value: str) -> set[str]:
    return {
        re.sub(r"[^a-z0-9]", "", match.casefold())
        for match in re.findall(r"\b[a-zA-Z0-9\-/]{1,10}\d+[a-zA-Z0-9\-/]{0,10}\b", value or "")
        if 2 <= len(re.sub(r"[^a-z0-9]", "", match.casefold())) <= 12
    } | {x for x in re.findall(r"\b\d+\b", value or "") if 2 <= len(x) <= 10}


def block_keys(name: str, address: str) -> list[str]:
    """Same country-key construction for training and eventual v2 inference."""
    clean_name = normalize(name)
    name_words = {w for w in clean_name.split() if len(w) >= 3 and w not in STOP_WORDS}
    address_words = tokens(address, STOP_WORDS | ADDR_STOP_WORDS, 4)
    address_nums = sorted(numbers(address))[:2]
    keys = {"n:" + clean_name} if len(clean_name) >= 4 else set()
    keys.update("t:" + word for word in name_words)
    keys.update("a:" + num + ":" + word for num in address_nums for word in address_words)
    keys.update("w:" + word for word in address_words if len(word) >= 6)
    return sorted(keys)


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def trigrams(value: str) -> set[str]:
    value = normalize(value).replace(" ", "")
    return {value[i:i + 3] for i in range(len(value) - 2)}


def clean_company_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch)).lower()
    value = re.sub(r"\.(com|org|net|in|co|info|c0m|biz|io|us|fr)(\.[a-z]{2})?$", "", value)
    value = re.sub(r"[^\w\s]", " ", value)
    return "".join(word for word in value.split() if word not in STOP_WORDS)


def pair_features(name1: str, addr1: str, name2: str, addr2: str, source: int,
                  fuzzy_scores: tuple[float, ...] | None = None) -> np.ndarray:
    """Return the old 15 training features followed by five v2 features."""
    name1, name2 = name1 or "", name2 or ""
    addr1, addr2 = addr1 or "", addr2 or ""
    n1, n2 = name1.lower(), name2.lower()
    a1, a2 = addr1.lower(), addr2.lower()
    w1, w2 = tokens(name1, STOP_WORDS, 3), tokens(name2, STOP_WORDS, 3)
    aw1, aw2 = tokens(addr1, STOP_WORDS | ADDR_STOP_WORDS, 4), tokens(addr2, STOP_WORDS | ADDR_STOP_WORDS, 4)
    nums1, nums2 = numbers(addr1), numbers(addr2)
    both_addr = bool(addr1.strip() and addr2.strip())
    clean1, clean2 = clean_company_name(name1), clean_company_name(name2)
    shared = len(nums1 & nums2)
    non_latin = any(ord(ch) > 0x024F for ch in name1 + name2 if ch.isalpha())
    if fuzzy_scores is None:
        name_scores = (fuzz.ratio(n1, n2), fuzz.token_sort_ratio(n1, n2), fuzz.token_set_ratio(n1, n2))
        addr_scores = (
            fuzz.ratio(a1, a2), fuzz.token_sort_ratio(a1, a2), fuzz.token_set_ratio(a1, a2)
        ) if both_addr else (0.0, 0.0, 0.0)
    else:
        name_scores, addr_scores = fuzzy_scores[:3], fuzzy_scores[3:]
    out = [
        *name_scores,
        jaccard(w1, w2), abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1),
        *addr_scores,
        jaccard(aw1, aw2) if both_addr else 0.0,
        float(shared), float(shared > 0), float(not both_addr), float(non_latin),
        float(len(clean1) >= 4 and len(clean2) >= 4 and (clean1 in clean2 or clean2 in clean1)),
        float(bool(clean1) and clean1 == clean2),
        jaccard(trigrams(name1), trigrams(name2)),
        jaccard(trigrams(addr1), trigrams(addr2)) if both_addr else 0.0,
        float(bool(nums1 and nums2 and not shared)),
        min(len(a1), len(a2)) / max(len(a1), len(a2)) if both_addr else 0.0,
        float(source == 3),
    ]
    return np.asarray(out, dtype=np.float32)


def pair_features_batch(rows: list[tuple], workers: int = CPU_THREADS) -> np.ndarray:
    """Backward-compatible inference layout wrapper for ``score_speedup``."""
    return score_speedup(rows, layout="inference", workers=workers)


def _as_text(values: np.ndarray) -> list[str]:
    if any(value is not None and not isinstance(value, str) for value in values):
        raise TypeError("Names and addresses must be strings or None")
    return [value or "" for value in values]


def _prepare_records(values: np.ndarray, *, address: bool = False) -> dict[str, list | np.ndarray]:
    raw = _as_text(values)
    records = {"lower": [value.lower() for value in raw],
               "trigrams": [trigrams(value) for value in raw]}
    if address:
        records["address_tokens"] = [
            tokens(value, STOP_WORDS | ADDR_STOP_WORDS, 4) for value in raw
        ]
        records["numbers"] = [numbers(value) for value in raw]
    else:
        records["tokens"] = [tokens(value, STOP_WORDS, 3) for value in raw]
        records["clean"] = [clean_company_name(value) for value in raw]
        records["non_latin"] = np.fromiter(
            (any(ord(ch) > 0x024F for ch in value if ch.isalpha()) for value in raw),
            dtype=bool, count=len(raw),
        )
    return records


def _set_jaccard(left_sets: list[set[str]], right_sets: list[set[str]],
                 left_ix: np.ndarray, right_ix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return pair intersections and Jaccard scores from one batch-local CSR matrix."""
    all_sets = left_sets + right_sets
    vocabulary: dict[str, int] = {}
    indices: list[int] = []
    indptr = [0]
    sizes = np.empty(len(all_sets), dtype=np.int32)
    for row, values in enumerate(all_sets):
        sizes[row] = len(values)
        for value in sorted(values):
            indices.append(vocabulary.setdefault(value, len(vocabulary)))
        indptr.append(len(indices))
    matrix = csr_matrix(
        (np.ones(len(indices), dtype=np.int32), np.asarray(indices, dtype=np.int32),
         np.asarray(indptr, dtype=np.int32)),
        shape=(len(all_sets), len(vocabulary)), dtype=np.uint8,
    )
    left = matrix[left_ix]
    right = matrix[len(left_sets) + right_ix]
    shared = np.asarray(left.multiply(right).sum(axis=1)).reshape(-1).astype(np.int32)
    union = sizes[left_ix] + sizes[len(left_sets) + right_ix] - shared
    similarity = np.divide(shared, union, out=np.zeros(len(shared), dtype=np.float64), where=union != 0)
    return shared, similarity


def score_speedup(rows: list[tuple], *, layout: str, workers: int = CPU_THREADS) -> np.ndarray:
    """Vectorized pair features for either row layout, with bounded sparse workspaces."""
    if layout not in {"train", "inference"}:
        raise ValueError("layout must be 'train' or 'inference'")
    width = 9 if layout == "train" else 7
    if any(len(row) != width for row in rows):
        raise ValueError(f"{layout} rows must each contain {width} columns")
    if not rows:
        return np.empty((0, len(FEATURES)), dtype=np.float32)
    if len(rows) <= SCORE_WORKSPACE_ROWS:
        return _score_speedup_batch(rows, layout=layout, workers=workers)
    output = np.empty((len(rows), len(FEATURES)), dtype=np.float32)
    for start in range(0, len(rows), SCORE_WORKSPACE_ROWS):
        stop = min(len(rows), start + SCORE_WORKSPACE_ROWS)
        output[start:stop] = _score_speedup_batch(
            rows[start:stop], layout=layout, workers=workers,
        )
    return output


def _score_speedup_batch(rows: list[tuple], *, layout: str, workers: int) -> np.ndarray:

    table = np.asarray(rows, dtype=object)
    if layout == "train":
        source_col, name1_col, addr1_col, name2_col, addr2_col = 3, 5, 6, 7, 8
    else:
        source_col, name1_col, addr1_col, name2_col, addr2_col = 2, 3, 4, 5, 6
    _, left_first, left_ix = np.unique(table[:, 0], return_index=True, return_inverse=True)
    _, right_first, right_ix = np.unique(table[:, 1], return_index=True, return_inverse=True)
    left_names = _prepare_records(table[left_first, name1_col])
    right_names = _prepare_records(table[right_first, name2_col])
    left_addrs = _prepare_records(table[left_first, addr1_col], address=True)
    right_addrs = _prepare_records(table[right_first, addr2_col], address=True)
    n = len(rows)
    workers = max(1, min(int(workers), n))
    result = np.zeros((n, len(FEATURES)), dtype=np.float64)

    names1 = np.asarray(left_names["lower"], dtype=object)[left_ix].tolist()
    names2 = np.asarray(right_names["lower"], dtype=object)[right_ix].tolist()
    for column, scorer in enumerate((fuzz.ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio)):
        result[:, column] = process.cpdist(
            names1, names2, scorer=scorer, dtype=np.float64, workers=workers,
        )

    addrs1 = np.asarray(left_addrs["lower"], dtype=object)[left_ix].tolist()
    addrs2 = np.asarray(right_addrs["lower"], dtype=object)[right_ix].tolist()
    both_addr = np.fromiter(
        (bool(a.strip() and b.strip()) for a, b in zip(addrs1, addrs2)),
        dtype=bool, count=n,
    )
    valid_addr = np.flatnonzero(both_addr)
    for offset, scorer in enumerate((fuzz.ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio), 5):
        if len(valid_addr):
            result[valid_addr, offset] = process.cpdist(
                [addrs1[i] for i in valid_addr], [addrs2[i] for i in valid_addr],
                scorer=scorer, dtype=np.float64, workers=workers,
            )

    _, name_jaccard = _set_jaccard(left_names["tokens"], right_names["tokens"], left_ix, right_ix)
    result[:, 3] = name_jaccard
    len1 = np.fromiter((len(value) for value in left_names["lower"]), dtype=np.int32)[left_ix]
    len2 = np.fromiter((len(value) for value in right_names["lower"]), dtype=np.int32)[right_ix]
    result[:, 4] = np.abs(len1 - len2) / np.maximum(np.maximum(len1, len2), 1)

    _, addr_jaccard = _set_jaccard(left_addrs["address_tokens"], right_addrs["address_tokens"], left_ix, right_ix)
    result[:, 8] = np.where(both_addr, addr_jaccard, 0.0)
    shared_numbers, _ = _set_jaccard(left_addrs["numbers"], right_addrs["numbers"], left_ix, right_ix)
    result[:, 9] = shared_numbers
    result[:, 10] = shared_numbers > 0
    result[:, 11] = ~both_addr
    result[:, 12] = left_names["non_latin"][left_ix] | right_names["non_latin"][right_ix]

    clean1 = np.asarray(left_names["clean"], dtype=np.str_)[left_ix]
    clean2 = np.asarray(right_names["clean"], dtype=np.str_)[right_ix]
    domain = (np.char.str_len(clean1) >= 4) & (np.char.str_len(clean2) >= 4)
    result[:, 13] = domain & ((np.strings.find(clean2, clean1) >= 0) |
                              (np.strings.find(clean1, clean2) >= 0))
    result[:, 14] = (clean1 != "") & (clean1 == clean2)

    _, name_trigram = _set_jaccard(left_names["trigrams"], right_names["trigrams"], left_ix, right_ix)
    result[:, 15] = name_trigram
    _, addr_trigram = _set_jaccard(left_addrs["trigrams"], right_addrs["trigrams"], left_ix, right_ix)
    result[:, 16] = np.where(both_addr, addr_trigram, 0.0)
    n1_count = np.fromiter((len(values) for values in left_addrs["numbers"]), dtype=np.int32)[left_ix]
    n2_count = np.fromiter((len(values) for values in right_addrs["numbers"]), dtype=np.int32)[right_ix]
    result[:, 17] = (n1_count > 0) & (n2_count > 0) & (shared_numbers == 0)
    addr_len1 = np.fromiter((len(value) for value in left_addrs["lower"]), dtype=np.int32)[left_ix]
    addr_len2 = np.fromiter((len(value) for value in right_addrs["lower"]), dtype=np.int32)[right_ix]
    result[:, 18] = np.divide(
        np.minimum(addr_len1, addr_len2), np.maximum(addr_len1, addr_len2),
        out=np.zeros(n, dtype=np.float64), where=both_addr,
    )
    result[:, 19] = table[:, source_col] == 3
    return result.astype(np.float32)


def f05(true_count: np.ndarray, predicted_count: np.ndarray, true_positive: np.ndarray) -> np.ndarray:
    """Per-Source1 F0.5, including the challenge singleton convention."""
    out = np.zeros(len(true_count), dtype=np.float64)
    empty = (true_count == 0) & (predicted_count == 0)
    out[empty] = 1.0
    valid = true_positive > 0
    out[valid] = 1.25 * true_positive[valid] / (0.25 * true_count[valid] + predicted_count[valid])
    return out


if __name__ == "__main__":
    assert len(FEATURES) == len(pair_features("A", "1 Road", "A", "1 Road", 2)) == 20
    assert np.allclose(f05(np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([0, 0, 1])), [1, 0, 0.5])
