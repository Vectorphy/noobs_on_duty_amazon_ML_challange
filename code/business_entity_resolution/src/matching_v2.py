"""Shared v2 blocking features and challenge scoring.

Only supplied record fields are used. Keep this module identical when v2 is
eventually promoted to submission inference.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from rapidfuzz import fuzz, process


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


def pair_features_batch(rows: list[tuple], workers: int = 8) -> np.ndarray:
    """Build unchanged pair features while batching parallel RapidFuzz scorers."""
    if not rows:
        return np.empty((0, len(FEATURES)), dtype=np.float32)
    names1 = [(row[3] or "").lower() for row in rows]
    names2 = [(row[5] or "").lower() for row in rows]
    addrs1 = [(row[4] or "").lower() for row in rows]
    addrs2 = [(row[6] or "").lower() for row in rows]
    has_both_addresses = [bool(a.strip() and b.strip()) for a, b in zip(addrs1, addrs2)]
    workers = max(1, min(workers, len(rows)))

    def scores(left: list[str], right: list[str]) -> list[np.ndarray]:
        return [process.cpdist(left, right, scorer=scorer, dtype=np.float32, workers=workers)
                for scorer in (fuzz.ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio)]

    name_scores = scores(names1, names2)
    addr_scores = scores(addrs1, addrs2)
    result = np.empty((len(rows), len(FEATURES)), dtype=np.float32)
    for i, row in enumerate(rows):
        addr = tuple(float(values[i]) if has_both_addresses[i] else 0.0 for values in addr_scores)
        fuzzy = tuple(float(values[i]) for values in name_scores) + addr
        result[i] = pair_features(row[3], row[4], row[5], row[6], row[2], fuzzy)
    return result


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
