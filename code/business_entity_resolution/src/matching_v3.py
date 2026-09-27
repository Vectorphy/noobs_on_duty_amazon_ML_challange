"""Shared v3 blocking features, pair features, and challenge scoring.

Improvements over v2
--------------------
A1  Phonetic blocking  - Double Metaphone keys (p: prefix) for the first
    meaningful name token.  Pure-Python; no C extension.

A2  Postal-code compound blocking - US 5-digit ZIPs, Indian 6-digit PINs,
    French 5-digit codes.  Key: z:<postal>:<first_name_token>.

A3  Adaptive candidate depth - MAX_CANDIDATES_DEEP=48 and
    DEEP_CANDIDATE_THRESHOLD=0.72 exported for train/infer.

A4  4-char prefix key (q:) - high-recall catch-all for fuzz/typo variants.
A5  Initials key (i:) - catches "MGH" = "Maharashtra General Hospital".
A6  First-two-token bigram key (b:) - catches word-order swaps.

B1  location_mismatch feature: 1.0 when records carry differing state/postal.
B2  tfidf_name_cosine feature: TF-IDF weighted cosine on name tokens.
B3  street_number_match feature: +1/0/-1.
B4  clean_name_jw: JW similarity on legal-suffix-stripped names.
B5  sorted_token_match: 1.0 when sorted name-token fingerprints are equal.
B6  shared_rare_token: normalised max IDF weight of shared name token.
B7  city_locality_match: 1.0 when last significant address token matches.

SINGLETON_GUARD_THRESHOLD exported for inference singleton guard (C3).

No GPU engine dependency.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Optional

import numpy as np
from rapidfuzz import fuzz

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
V3_FEATURES = (
    "location_mismatch",
    "tfidf_name_cosine",
    "street_number_match",
    "clean_name_jw",
    "sorted_token_match",
    "shared_rare_token",
    "city_locality_match",
)
FEATURES = BASE_FEATURES + EXTRA_FEATURES + V3_FEATURES

ABLATION_GROUPS = {
    "name_trigram":               (15,),
    "address_trigram_and_length": (16, 18),
    "number_conflict":            (17,),
    "target_source":              (19,),
    "legacy_flags":               (12, 13, 14),
    "location_mismatch":          (20,),
    "tfidf_cosine":               (21,),
    "street_number_match":        (22,),
    "clean_name_jw":              (23,),
    "sorted_token_match":         (24,),
    "shared_rare_token":          (25,),
    "city_locality_match":        (26,),
}

MAX_CANDIDATES_PER_SOURCE  = 32
MAX_CANDIDATES_DEEP        = 48
DEEP_CANDIDATE_THRESHOLD   = 0.72
SINGLETON_GUARD_THRESHOLD  = 0.35   # C3: entities whose max score < this are predicted as singletons

KEY_S1_LIMITS   = {"n": 512, "t": 64, "a": 64, "w": 16, "p": 128, "z": 256,
                   "q": 200, "i": 64, "b": 128}
KEY_TARGET_LIMIT = 2048

# -----------------------------------------------------------------------
# Pure-Python Double Metaphone (A1)
# -----------------------------------------------------------------------

_VOWELS = set("aeiou")


def _dm_pad(s: str) -> str:
    return s + "     "


def double_metaphone(word: str) -> tuple[str, str]:
    """Return (primary, secondary) Double Metaphone codes."""
    word = word.lower()
    word = "".join(
        ch for ch in unicodedata.normalize("NFKD", word)
        if not unicodedata.combining(ch)
    )
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return "", ""

    if word[:2] in ("ae", "ai", "ao", "au", "oi", "oe", "ou"):
        p_out: list[str] = ["A"]
        s_out: list[str] = ["A"]
        start = 1
    else:
        p_out = []
        s_out = []
        start = 0
        fixups = {"gn": "n", "kn": "n", "pn": "n", "ae": "e", "wr": "r"}
        pfx = word[:2]
        if pfx in fixups:
            word = fixups[pfx] + word[2:]

    s = _dm_pad(word)
    i = start
    length = len(word)

    def add(pri: str, sec: str = "") -> None:
        p_out.append(pri)
        s_out.append(sec or pri)

    while i < length:
        c = s[i]
        if c in _VOWELS:
            if i == 0:
                add("A")
            i += 1
            continue
        if c == "b":
            add("P"); i += 2 if s[i + 1] == "b" else 1
        elif c == "c":
            if s[i:i+2] == "ch":
                add("X", "K"); i += 2
            elif s[i+1] in "iey":
                add("S"); i += 1
            else:
                add("K"); i += 1
        elif c == "d":
            if s[i:i+2] == "dg" and s[i+2] in "iey":
                add("J"); i += 3
            elif s[i:i+2] in ("dt", "dd"):
                add("T"); i += 2
            else:
                add("T"); i += 1
        elif c == "f":
            add("F"); i += 2 if s[i+1] == "f" else 1
        elif c == "g":
            if s[i:i+2] == "gh":
                if i > 0 and s[i-1] not in _VOWELS:
                    add("K")
                i += 2
            elif s[i+1] in "iey":
                add("J", "K"); i += 1
            elif s[i:i+2] == "gn":
                add("KN", "N"); i += 1
            else:
                add("K"); i += 1
        elif c == "h":
            if s[i+1] in _VOWELS:
                add("H")
            i += 1
        elif c == "j":
            add("J"); i += 1
        elif c == "k":
            add("K"); i += 2 if s[i+1] == "k" else 1
        elif c == "l":
            add("L"); i += 2 if s[i+1] == "l" else 1
        elif c == "m":
            add("M"); i += 2 if s[i+1] == "m" else 1
        elif c == "n":
            add("N"); i += 2 if s[i+1] == "n" else 1
        elif c == "p":
            if s[i+1] == "h":
                add("F"); i += 2
            else:
                add("P"); i += 2 if s[i+1] == "p" else 1
        elif c == "q":
            add("K"); i += 2 if s[i+1] == "q" else 1
        elif c == "r":
            add("R"); i += 2 if s[i+1] == "r" else 1
        elif c == "s":
            if s[i:i+2] in ("sh", "sc") or s[i:i+3] == "sch":
                add("X"); i += 2
            elif s[i+1] in "iey":
                add("S", "X"); i += 1
            else:
                add("S"); i += 2 if s[i+1] == "s" else 1
        elif c == "t":
            if s[i:i+2] == "th":
                add("T"); i += 2
            elif s[i:i+3] == "tch":
                add("X"); i += 3
            else:
                add("T"); i += 2 if s[i+1] in "td" else 1
        elif c == "v":
            add("F"); i += 2 if s[i+1] == "v" else 1
        elif c == "w":
            if s[i+1] in _VOWELS:
                add("A")
            i += 1
        elif c == "x":
            add("S"); i += 1
        elif c == "z":
            add("S"); i += 2 if s[i+1] == "z" else 1
        else:
            i += 1

    return "".join(p_out)[:6], "".join(s_out)[:6]


# -----------------------------------------------------------------------
# Postal-code extraction (A2)
# -----------------------------------------------------------------------

_RE_US_ZIP = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_RE_IN_PIN = re.compile(r"\b([1-9]\d{5})\b")
_RE_FR_ZIP = re.compile(r"\b(0[1-9]\d{3}|[1-9]\d{4})\b")


def extract_postal_codes(address: str) -> list[str]:
    """Return at most one postal code (IN > US/FR priority)."""
    if not address:
        return []
    for pattern in (_RE_IN_PIN, _RE_US_ZIP, _RE_FR_ZIP):
        m = pattern.search(address)
        if m:
            return [m.group(1)]
    return []


# -----------------------------------------------------------------------
# TF-IDF cosine (B2)
# -----------------------------------------------------------------------

_IDF_TABLE_RAW: list[tuple[str, float]] = [
    ("hospital", 0.045), ("plaza", 0.038), ("services", 0.12),
    ("solutions", 0.09), ("technologies", 0.07), ("enterprises", 0.06),
    ("trading", 0.05), ("centre", 0.04), ("center", 0.04),
    ("management", 0.035), ("systems", 0.08), ("associates", 0.05),
    ("consultants", 0.04), ("engineers", 0.03), ("construction", 0.035),
    ("foods", 0.025), ("pharma", 0.025), ("india", 0.055),
    ("global", 0.04), ("national", 0.03), ("medical", 0.03),
    ("health", 0.025), ("auto", 0.025), ("logistics", 0.02),
    ("finance", 0.02), ("capital", 0.02), ("real", 0.02),
    ("estate", 0.02), ("hotels", 0.018), ("retail", 0.015),
    ("agency", 0.015), ("digital", 0.015), ("media", 0.015),
    ("consulting", 0.015), ("research", 0.015), ("export", 0.014),
    ("import", 0.013), ("properties", 0.013), ("infrastructure", 0.012),
    ("energy", 0.012), ("power", 0.012), ("electrical", 0.01),
    ("electronics", 0.01), ("packaging", 0.01), ("textile", 0.01),
    ("textiles", 0.01), ("garments", 0.009), ("steel", 0.009),
    ("chemicals", 0.009), ("labs", 0.009), ("lab", 0.008),
    ("foundation", 0.008), ("trust", 0.008), ("welfare", 0.006),
    ("super", 0.006), ("market", 0.02), ("pharmacy", 0.01),
    ("clinic", 0.01), ("diagnostic", 0.008),
]
_IDF_N = 2_000_000
_IDF: dict[str, float] = {
    tok: math.log(_IDF_N / max(1, int(frac * _IDF_N)))
    for tok, frac in _IDF_TABLE_RAW
}
_IDF_DEFAULT = math.log(_IDF_N)


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "").casefold()
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def _tfidf_vector(name: str) -> dict[str, float]:
    words = [w for w in normalize(name).split() if len(w) >= 3 and w not in STOP_WORDS]
    if not words:
        return {}
    tf: dict[str, float] = {}
    for w in words:
        tf[w] = tf.get(w, 0.0) + 1.0
    n = len(words)
    return {w: (cnt / n) * _IDF.get(w, _IDF_DEFAULT) for w, cnt in tf.items()}


def tfidf_cosine(name1: str, name2: str) -> float:
    v1 = _tfidf_vector(name1)
    v2 = _tfidf_vector(name2)
    if not v1 or not v2:
        return 0.0
    common = set(v1) & set(v2)
    dot = sum(v1[t] * v2[t] for t in common)
    mag1 = math.sqrt(sum(x * x for x in v1.values()))
    mag2 = math.sqrt(sum(x * x for x in v2.values()))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot / (mag1 * mag2)


# -----------------------------------------------------------------------
# Street-number match (B3)
# -----------------------------------------------------------------------

_RE_LEAD_NUM = re.compile(r"^\s*(\d+[a-zA-Z]?)[\s,/\\-]")


def _lead_street_number(address: str) -> Optional[str]:
    m = _RE_LEAD_NUM.match(address or "")
    return m.group(1).lower() if m else None


def street_number_match(addr1: str, addr2: str) -> float:
    n1 = _lead_street_number(addr1)
    n2 = _lead_street_number(addr2)
    if n1 is None or n2 is None:
        return 0.0
    return 1.0 if n1 == n2 else -1.0


# -----------------------------------------------------------------------
# Location mismatch (B1)
# -----------------------------------------------------------------------

_US_STATES = {
    "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id",
    "il","in","ia","ks","ky","la","me","md","ma","mi","mn","ms",
    "mo","mt","ne","nv","nh","nj","nm","ny","nc","nd","oh","ok",
    "or","pa","ri","sc","sd","tn","tx","ut","vt","va","wa","wv",
    "wi","wy","dc",
}
_IN_STATES = {
    "ap","ar","as","br","cg","ch","dd","dl","dh","ga","gj","hr",
    "hp","jh","jk","ka","kl","la","ld","mh","ml","mn","mp","mz",
    "nl","od","pb","py","rj","sk","tn","tr","ts","up","ut","wb",
}
_RE_2L = re.compile(r"\b([a-z]{2})\b")


def _extract_location_code(address: str) -> Optional[str]:
    codes = extract_postal_codes(address)
    if codes:
        return codes[0]
    for m in _RE_2L.finditer((address or "").lower()):
        tok = m.group(1)
        if tok in _US_STATES or tok in _IN_STATES:
            return tok
    return None


def location_mismatch(addr1: str, addr2: str) -> float:
    c1 = _extract_location_code(addr1)
    c2 = _extract_location_code(addr2)
    if c1 is None or c2 is None:
        return 0.0
    return 0.0 if c1 == c2 else 1.0


# -----------------------------------------------------------------------
# City / locality tail token (B7)
# -----------------------------------------------------------------------

def _extract_city_token(address: str) -> Optional[str]:
    """Last significant non-numeric, non-stopword token in the address."""
    combined_stop = STOP_WORDS | ADDR_STOP_WORDS
    words = [
        w for w in normalize(address).split()
        if len(w) >= 4 and not w.isdigit() and w not in combined_stop
    ]
    return words[-1] if words else None


def city_locality_match(addr1: str, addr2: str) -> float:
    """1.0 when both addresses share the same trailing locality token."""
    c1 = _extract_city_token(addr1)
    c2 = _extract_city_token(addr2)
    if c1 is None or c2 is None:
        return 0.0
    return 1.0 if c1 == c2 else 0.0


# -----------------------------------------------------------------------
# Shared rare-token score (B6)
# -----------------------------------------------------------------------

def shared_rare_token_score(name1: str, name2: str) -> float:
    """Normalised max IDF weight of tokens shared between both names.

    A token unique to a specific business (high IDF / unknown to the table)
    is a very strong positive signal when shared.  Returns 0–1.
    """
    t1 = {w for w in normalize(name1).split() if len(w) >= 3 and w not in STOP_WORDS}
    t2 = {w for w in normalize(name2).split() if len(w) >= 3 and w not in STOP_WORDS}
    common = t1 & t2
    if not common:
        return 0.0
    max_idf = max(_IDF.get(w, _IDF_DEFAULT) for w in common)
    return float(min(max_idf / _IDF_DEFAULT, 1.0))


# -----------------------------------------------------------------------
# Sorted-token fingerprint match (B5)
# -----------------------------------------------------------------------

def sorted_fingerprint_match(name1: str, name2: str) -> float:
    """1.0 when alphabetically-sorted name tokens are identical.

    Catches word-order permutations that fool fuzz.ratio and JW.
    """
    f1 = " ".join(sorted(w for w in normalize(name1).split() if w not in STOP_WORDS))
    f2 = " ".join(sorted(w for w in normalize(name2).split() if w not in STOP_WORDS))
    if not f1 or not f2:
        return 0.0
    return 1.0 if f1 == f2 else 0.0



# -----------------------------------------------------------------------
# Core text utilities (unchanged from v2)
# -----------------------------------------------------------------------

def tokens(value: str, stop_words: set[str], min_length: int) -> set[str]:
    return {
        word for word in normalize(value).split()
        if len(word) >= min_length and word not in stop_words
    }


def numbers(value: str) -> set[str]:
    return {
        re.sub(r"[^a-z0-9]", "", match.casefold())
        for match in re.findall(
            r"\b[a-zA-Z0-9\-/]{1,10}\d+[a-zA-Z0-9\-/]{0,10}\b", value or ""
        )
        if 2 <= len(re.sub(r"[^a-z0-9]", "", match.casefold())) <= 12
    } | {x for x in re.findall(r"\b\d+\b", value or "") if 2 <= len(x) <= 10}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def trigrams(value: str) -> set[str]:
    value = normalize(value).replace(" ", "")
    return {value[i:i+3] for i in range(len(value) - 2)}


def clean_company_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch)).lower()
    value = re.sub(r"\.(com|org|net|in|co|info|c0m|biz|io|us|fr)(\.[a-z]{2})?$", "", value)
    value = re.sub(r"[^\w\s]", " ", value)
    return "".join(word for word in value.split() if word not in STOP_WORDS)


# -----------------------------------------------------------------------
# Blocking keys
# -----------------------------------------------------------------------

def block_keys(name: str, address: str) -> list[str]:
    """v3 blocking keys: v2 keys + p: phonetic + z: postal + q: prefix + i: initials + b: bigram."""
    clean_name = normalize(name)
    name_words = [w for w in clean_name.split() if len(w) >= 3 and w not in STOP_WORDS]
    address_words = tokens(address, STOP_WORDS | ADDR_STOP_WORDS, 4)
    address_nums = sorted(numbers(address))[:2]

    keys: set[str] = set()
    if len(clean_name) >= 4:
        keys.add("n:" + clean_name)
    keys.update("t:" + w for w in name_words)
    keys.update(
        "a:" + num + ":" + word
        for num in address_nums for word in address_words
    )
    keys.update("w:" + word for word in address_words if len(word) >= 6)

    # A1: Double Metaphone phonetic key on first name token
    if name_words:
        primary, _ = double_metaphone(name_words[0])
        if primary and len(primary) >= 2:
            keys.add("p:" + primary)

    # A2: Postal-code compound key
    postal = extract_postal_codes(address)
    if postal and name_words:
        keys.add("z:" + postal[0] + ":" + name_words[0])

    # A4: 4-char prefix key — high-recall catch-all for fuzz/typo variants
    if len(clean_name) >= 4:
        keys.add("q:" + clean_name[:4])

    # A5: Initials key — catches abbreviations like "MGH" for "Maharashtra General Hospital"
    if len(name_words) >= 2:
        initials = "".join(w[0] for w in name_words if w)
        if len(initials) >= 2:
            keys.add("i:" + initials)

    # A6: First-two-token bigram — catches word-order swaps between sources
    if len(name_words) >= 2:
        keys.add("b:" + name_words[0] + "_" + name_words[1])

    return sorted(keys)


# -----------------------------------------------------------------------
# Pair features (20 v2 + 7 v3 = 27)
# -----------------------------------------------------------------------

def pair_features(
    name1: str, addr1: str,
    name2: str, addr2: str,
    source: int,
) -> np.ndarray:
    name1, name2 = name1 or "", name2 or ""
    addr1, addr2 = addr1 or "", addr2 or ""
    n1, n2 = name1.lower(), name2.lower()
    a1, a2 = addr1.lower(), addr2.lower()
    w1  = tokens(name1, STOP_WORDS, 3)
    w2  = tokens(name2, STOP_WORDS, 3)
    aw1 = tokens(addr1, STOP_WORDS | ADDR_STOP_WORDS, 4)
    aw2 = tokens(addr2, STOP_WORDS | ADDR_STOP_WORDS, 4)
    nums1, nums2 = numbers(addr1), numbers(addr2)
    both_addr = bool(addr1.strip() and addr2.strip())
    clean1, clean2 = clean_company_name(name1), clean_company_name(name2)
    shared = len(nums1 & nums2)
    non_latin = any(ord(ch) > 0x024F for ch in name1 + name2 if ch.isalpha())
    out = [
        fuzz.ratio(n1, n2), fuzz.token_sort_ratio(n1, n2), fuzz.token_set_ratio(n1, n2),
        jaccard(w1, w2), abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1),
        fuzz.ratio(a1, a2) if both_addr else 0.0,
        fuzz.token_sort_ratio(a1, a2) if both_addr else 0.0,
        fuzz.token_set_ratio(a1, a2) if both_addr else 0.0,
        jaccard(aw1, aw2) if both_addr else 0.0,
        float(shared), float(shared > 0), float(not both_addr), float(non_latin),
        float(len(clean1) >= 4 and len(clean2) >= 4 and (clean1 in clean2 or clean2 in clean1)),
        float(bool(clean1) and clean1 == clean2),
        jaccard(trigrams(name1), trigrams(name2)),
        jaccard(trigrams(addr1), trigrams(addr2)) if both_addr else 0.0,
        float(bool(nums1 and nums2 and not shared)),
        min(len(a1), len(a2)) / max(len(a1), len(a2)) if both_addr else 0.0,
        float(source == 3),
        location_mismatch(addr1, addr2),
        tfidf_cosine(name1, name2),
        street_number_match(addr1, addr2),
        # B4: Jaro-Winkler on legal-suffix-stripped names
        fuzz.ratio(clean1, clean2) / 100.0,
        # B5: Sorted-token fingerprint match
        sorted_fingerprint_match(name1, name2),
        # B6: Shared rare-token score
        shared_rare_token_score(name1, name2),
        # B7: City / locality tail-token match
        city_locality_match(addr1, addr2),
    ]
    return np.asarray(out, dtype=np.float32)


# -----------------------------------------------------------------------
# Challenge scoring (unchanged from v2)
# -----------------------------------------------------------------------

def f05(
    true_count: np.ndarray,
    predicted_count: np.ndarray,
    true_positive: np.ndarray,
) -> np.ndarray:
    out = np.zeros(len(true_count), dtype=np.float64)
    empty = (true_count == 0) & (predicted_count == 0)
    out[empty] = 1.0
    valid = true_positive > 0
    out[valid] = (
        1.25 * true_positive[valid]
        / (0.25 * true_count[valid] + predicted_count[valid])
    )
    return out


if __name__ == "__main__":
    n = len(pair_features("A", "1 Road", "A", "1 Road", 2))
    assert n == 27 == len(FEATURES), f"Expected 27, got {n}"
    assert np.allclose(f05(np.array([0,1,2]),np.array([0,1,2]),np.array([0,0,1])),[1,0,.5])
    ks = block_keys("Kalyani Medicals", "Pin 700130 Kolkata")
    assert any(k.startswith("p:") for k in ks), f"Missing p: in {ks}"
    assert any(k.startswith("z:") for k in ks), f"Missing z: in {ks}"
    assert any(k.startswith("q:") for k in ks), f"Missing q: in {ks}"
    assert any(k.startswith("i:") for k in ks), f"Missing i: in {ks}"
    assert location_mismatch("105 Main St IL 60601","105 Main St CA 90001") == 1.0
    assert street_number_match("105 Ribbon Ln","11 Ribbon Ln") == -1.0
    assert sorted_fingerprint_match("Widget Alpha", "Alpha Widget") == 1.0
    assert shared_rare_token_score("Zydeco Pharma", "Zydeco Pharma") > 0.0
    assert city_locality_match("10 MG Road Bangalore", "22 MG Road Bangalore") == 1.0
    print("All v3 self-tests passed (27 features).")
