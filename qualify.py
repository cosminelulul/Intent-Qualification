"""
Company Qualification System
Usage: python qualification_system.p
       Then enter a query at the prompt.
"""

# --------------------- Libraries --------------------- #

import ast
import json
import re
import string
import sys

import anthropic
import pandas as pd
from rank_bm25 import BM25Okapi
from openai import OpenAI
from pathlib import Path


# --------------------- Config --------------------- #

DECOMPOSE_MODEL = "anthropic/claude-haiku-4-5"
RERANK_MODEL    = "anthropic/claude-haiku-4-5"
DIRECT_LLM_THRESHOLD = 30   # survivors <= this: skip BM25, go straight to LLM
RETRIEVE_TOP_K       = 50   # BM25 narrows to this many before LLM rerank
BATCH_SIZE           = 10   # companies per LLM rerank prompt
TOP_N_RESULTS        = 10   # results printed

client = OpenAI(
    api_key="",  # <-- add your key here or set OPENAI_API_KEY env var
    base_url="https://openrouter.ai/api/v1",
)


# --------------------- Data loading --------------------- #

def _parse_dict_field(val) -> dict:
    """Safely parse a field that may be a dict or a stringified Python dict."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            pass
        # Regex fallback for negative longitudes that break ast
        result = {}
        for key in ("country_code", "region_name", "town", "code", "label"):
            m = re.search(rf"'{key}'\s*:\s*'([^']*)'", val)
            if m:
                result[key] = m.group(1)
        return result
    return {}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)

    addr  = df["address"].apply(_parse_dict_field)
    naics = df["primary_naics"].apply(_parse_dict_field)

    df["country_code"] = addr.apply(lambda d: d.get("country_code", "").lower())
    df["naics_label"]  = naics.apply(lambda d: d.get("label", ""))
    df["naics_code"]   = naics.apply(lambda d: d.get("code", ""))

    for col in ("business_model", "core_offerings", "target_markets"):
        df[col] = df[col].apply(lambda v: v if isinstance(v, list) else [])

    def embed_text(row) -> str:
        def s(v):
            return str(v) if (v and not (isinstance(v, float) and pd.isna(v))) else ""
        parts = [
            s(row.get("operational_name")),
            s(row.get("naics_label")),
            s(row.get("description") or "")[:400],
            "Offerings: " + ", ".join(row.get("core_offerings", [])[:6]),
            "Markets: "   + ", ".join(row.get("target_markets", [])[:5]),
            "Model: "     + ", ".join(row.get("business_model", [])[:3]),
            f"Country: {s(row.get('country_code')).upper()}",
        ]
        return " | ".join(p for p in parts if p.strip())

    df["embed_text"] = df.apply(embed_text, axis=1)
    return df


# --------------------- BM25 index --------------------- #

def _tokenize(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if len(t) > 1]


def build_bm25(df: pd.DataFrame) -> list[list[str]]:
    return [_tokenize(t) for t in df["embed_text"]]


def bm25_search(
    reformulated_query: str,
    candidate_idx: list[int],
    tokenized_corpus: list[list[str]],
    top_k: int,
) -> list[int]:
    """
    Score only the candidate rows with BM25.
    Builds a sub-corpus from survivors so this stays O(survivors), not O(total).
    Scales to 100k: if hard filters cut 100k to 500, BM25 runs on 500.
    """
    if not candidate_idx:
        return []
    q_tokens = _tokenize(reformulated_query)
    if not q_tokens:
        return candidate_idx[:top_k]

    sub_corpus = [tokenized_corpus[i] for i in candidate_idx]
    bm25       = BM25Okapi(sub_corpus)
    scores     = bm25.get_scores(q_tokens)

    ranked = sorted(range(len(candidate_idx)), key=lambda j: scores[j], reverse=True)
    return [candidate_idx[j] for j in ranked[:top_k]]


# ---------------------  Stage 1 - Query decomposition --------------------- #

DECOMPOSE_PROMPT = """\
You are a query decomposition engine for a company search system.

Given a user query, return a JSON object with these keys:

hard_filters — object, fields with direct database equivalents only.
  Allowed keys:
    country_code      string  ISO-2 lowercase (single country)
    country_codes     list    ISO-2 strings (for regions like Scandinavia or Europe)
    min_employees     number
    max_employees     number
    min_revenue       number  (USD)
    max_revenue       number  (USD)
    min_year_founded  number
    max_year_founded  number
    is_public         boolean
    naics_keywords    list    strings to softly match in NAICS label or description
  Omit keys you are not confident about.

soft_filters — list of strings.
  Semantic criteria about what the company must DO or BE.
  Be role-specific: supplier vs buyer, manufacturer vs distributor.
  Example: "provides HR software as a B2B SaaS product"

inferred_constraints — list of strings.
  Things implied by the query but not stated.
  Example: "startup" implies small, recent, probably private.
  Example: "competing with traditional banks" implies NOT a traditional bank.

complexity — one of: "structured", "mixed", "judgment"
  structured  = mostly hard filters, minimal semantic reasoning needed
  mixed       = hard filters + meaningful soft criteria
  judgment    = primarily semantic, role-inference, or supply-chain reasoning

reformulated_query — string.
  Role-aware rewrite for text retrieval. Describe what the TARGET COMPANY does.
  Wrong: "cosmetics packaging"
  Right: "company that manufactures or supplies packaging materials to consumer goods brands"

  Clarification : complexity should only be "structured" if ALL criteria map to hard database fields.
If any criterion requires understanding what industry or sector a company operates in
(e.g. "software company", "fintech", "logistics"), classify as "mixed" at minimum.


Output ONLY valid JSON. No markdown, no explanation.

User query: {query}"""


def decompose(query: str) -> dict:
    resp = client.chat.completions.create(
        model=DECOMPOSE_MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(query=query)}],
    )
    text = resp.choices[0].message.content.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)




# --------------------- Stage 2 - Hard filter pass --------------------- #


# Regional groupings for query expansion (e.g. "Europe" → list of country codes).

## -------------- Europe
SCANDINAVIA = {"se", "no", "dk", "fi", "is"}
WESTERN_EUROPE = {"gb","fr","de","nl","be","lu","ie","at","ch"}
EASTERN_EUROPE = {"pl","cz","sk","hu","ro","bg","hr","si","ee","lv","lt","ua","by","md","rs","al","mk","ba","me"}
SOUTHERN_EUROPE = {"es","pt","it","gr","mt","cy","hr","si","al","mk","ba","me","rs"}

EUROPE = SCANDINAVIA | WESTERN_EUROPE | EASTERN_EUROPE | SOUTHERN_EUROPE | {"fi","is"}

## -------------- Americas
NORTH_AMERICA = {"us", "ca", "mx"}
LATIN_AMERICA = {
    "mx","gt","bz","hn","sv","ni","cr","pa","cu","do","ht","jm","tt",
    "co","ve","gy","sr","br","ec","pe","bo","py","cl","ar","uy",
}

AMERICAS = NORTH_AMERICA | LATIN_AMERICA

## ------------- Middle East & North Africa (MENA)
MIDDLE_EAST = {
    "ae","sa","qa","kw","bh","om","ye","il","jo","lb","sy","iq","ir","tr",
}
AFRICA = {
    "za","ng","ke","et","gh","tz","ug","rw","sn","ci","cm","ao","eg","ma",
    "tn","dz","ly","sd","zw","zm","mz","mg","mu",
}

## ------------- Asia & Oceania

EAST_ASIA = {"cn","jp","kr","tw","hk","mo","mn"}
SOUTHEAST_ASIA = {"vn","th","id","ph","my","sg","mm","kh","la","bn","tl"}
SOUTH_ASIA = {"in","pk","bd","lk","np","bt","mv"}

ASIA = EAST_ASIA | SOUTHEAST_ASIA | SOUTH_ASIA | {"kz","uz","tm","kg","tj"}
ASIA_PACIFIC = ASIA | {"au","nz","pg","fj"}

OCEANIA = {"au","nz","pg","fj","ws","to","vu","sb","ki","fm","pw","nr","tv"}

REGION_MAP = {
    # European sub-regions
    "scandinavia":      list(SCANDINAVIA),
    "nordics":          list(SCANDINAVIA),
    "europe":           list(EUROPE),
    "western europe":   list(WESTERN_EUROPE),
    "eastern europe":   list(EASTERN_EUROPE),
    "southern europe":  list(SOUTHERN_EUROPE),

    # Americas
    "north america":    list(NORTH_AMERICA),
    "latin america":    list(LATIN_AMERICA),
    "south america":    ["br","co","ve","ar","cl","pe","ec","bo","py","uy","gy","sr"],
    "central america":  ["mx","gt","bz","hn","sv","ni","cr","pa"],
    "caribbean":        ["cu","do","ht","jm","tt","bb","lc","vc","gd","ag","dm","kn"],
    "americas":         list(AMERICAS),

    # Asia
    "asia":             list(ASIA),
    "east asia":        list(EAST_ASIA),
    "southeast asia":   list(SOUTHEAST_ASIA),
    "south asia":       list(SOUTH_ASIA),
    "asia pacific":     list(ASIA_PACIFIC),
    "apac":             list(ASIA_PACIFIC),

    # Middle East
    "middle east":      list(MIDDLE_EAST),
    "mena":             list(MIDDLE_EAST | AFRICA),  # common business region grouping
    "gulf":             ["ae","sa","qa","kw","bh","om"],
    "gcc":              ["ae","sa","qa","kw","bh","om"],

    # Africa
    "africa":           list(AFRICA),
    "sub-saharan africa":["za","ng","ke","et","gh","tz","ug","rw","sn","ci","cm","ao","zw","zm","mz","mg","mu"],
    "north africa":     ["eg","ma","tn","dz","ly","sd"],

    # Oceania
    "oceania":          list(OCEANIA),
    "australia":        ["au"],  # often used as a region label, not just a country
}

def apply_hard_filters(df: pd.DataFrame, hf: dict) -> pd.Series:
    """
    Returns a boolean mask. Missing numeric data is kept (penalised later),
    not dropped. This is the scaling-safe approach at 100k.
    """
    mask = pd.Series(True, index=df.index)

    if "country_code" in hf:
        mask &= df["country_code"] == hf["country_code"].lower()

    if "country_codes" in hf:
        codes = [c.lower() for c in hf["country_codes"]]
        mask &= df["country_code"].isin(codes)

    if "is_public" in hf:
        mask &= df["is_public"] == bool(hf["is_public"])

    def numeric_filter(col, lo_key, hi_key):
        nonlocal mask
        lo, hi = hf.get(lo_key), hf.get(hi_key)
        if lo is not None:
            mask &= df[col].isna() | (df[col] >= lo)
        if hi is not None:
            mask &= df[col].isna() | (df[col] <= hi)

    numeric_filter("employee_count", "min_employees",    "max_employees")
    numeric_filter("revenue",        "min_revenue",      "max_revenue")
    numeric_filter("year_founded",   "min_year_founded", "max_year_founded")

    return mask


def data_penalty(row: pd.Series, hf: dict) -> float:
    """Penalty for missing data on constrained numeric fields (0.0–0.4)."""
    penalty = 0.0
    checks = [
        ("employee_count", "min_employees",    "max_employees"),
        ("revenue",        "min_revenue",      "max_revenue"),
        ("year_founded",   "min_year_founded", "max_year_founded"),
    ]
    for col, lo_key, hi_key in checks:
        if (hf.get(lo_key) or hf.get(hi_key)) and pd.isna(row[col]):
            penalty += 0.12
    return min(penalty, 0.4)


# --------------------- Stage 4 - LLM batch reranking --------------------- #

RERANK_PROMPT = """\
You are evaluating whether companies match a search query.

Query: {query}

What the matching company must do or be:
{soft_filters}

What the query implies but does not state:
{inferred_constraints}

For each company, return a JSON array. Each element:
  index  — the integer index shown
  score  — float 0.0 to 1.0 (1.0 = perfect match, 0.0 = clearly wrong)
  reason — one sentence

Be strict about role. A cosmetics company is NOT a match for "packaging supplier for cosmetics".
Score 0.5 only for genuine borderline cases.

Companies:
{companies}

Output ONLY the JSON array."""


def llm_rerank(
    candidates: list[tuple[int, pd.Series]],
    decomp: dict,
    query: str,
) -> dict[int, float]:
    scores = {}
    for i in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[i : i + BATCH_SIZE]

        companies_text = "\n\n".join(
            f"[{idx}] {row['operational_name']}\n"
            f"  NAICS: {row['naics_label']}\n"
            f"  Description: {(row['description'] or '')[:250]}\n"
            f"  Offerings: {', '.join(row['core_offerings'][:5])}\n"
            f"  Markets: {', '.join(row['target_markets'][:4])}\n"
            f"  Model: {', '.join(row['business_model'][:3])}"
            for idx, row in batch
        )
        soft     = "\n".join(f"- {s}" for s in decomp.get("soft_filters", [])) or "  (none)"
        inferred = "\n".join(f"- {s}" for s in decomp.get("inferred_constraints", [])) or "  (none)"

        resp = client.chat.completions.create(
            model=RERANK_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": RERANK_PROMPT.format(
                query                = query,
                soft_filters         = soft,
                inferred_constraints = inferred,
                companies            = companies_text,
            )}],
        )
        text = resp.choices[0].message.content.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        try:
            for r in json.loads(text):
                scores[r["index"]] = float(r["score"])
        except Exception as e:
            print(f"  [warn] rerank parse error: {e}")

    return scores



# --------------------- Stage 5 - Score fusion --------------------- #


def fuse(
    hard_penalty: float,
    bm25_rank: int | None,
    bm25_pool: int,
    llm_score: float | None,
) -> tuple[float, str]:
    base = 1.0 - hard_penalty

    if llm_score is not None and bm25_rank is not None:
        rank_score = 1.0 - (bm25_rank / max(bm25_pool, 1))
        score = llm_score * 0.60 + rank_score * 0.25 + base * 0.15
    elif llm_score is not None:
        score = llm_score * 0.75 + base * 0.25
    elif bm25_rank is not None:
        rank_score = 1.0 - (bm25_rank / max(bm25_pool, 1))
        score = rank_score * 0.65 + base * 0.35
    else:
        score = base

    score = max(0.0, min(1.0, score))
    confidence = "high" if score >= 0.72 else "probable" if score >= 0.45 else "borderline"
    return score, confidence


# --------------------- Main pipeline --------------------- #


def qualify(query: str, df: pd.DataFrame, tokenized: list[list[str]]) -> list[dict]:
    # Stage 1 — decompose
    print(f"\n[1/4] Decomposing query...")
    decomp = decompose(query)

    complexity   = decomp.get("complexity", "mixed")
    hf           = decomp.get("hard_filters", {})
    reformulated = decomp.get("reformulated_query", query)

    # Expand region aliases in country_codes
    for key in ("country_code", "country_codes"):
        if key not in hf:
            continue
        raw = hf[key] if isinstance(hf[key], list) else [hf[key]]
        expanded = []
        for code in raw:
            expanded.extend(REGION_MAP.get(code.lower(), [code.lower()]))
        if expanded:
            hf.pop("country_code", None)
            hf["country_codes"] = list(set(expanded))

    print(f"    complexity   : {complexity}")
    print(f"    hard_filters : {hf}")
    print(f"    soft_filters : {decomp.get('soft_filters', [])}")
    print(f"    inferred     : {decomp.get('inferred_constraints', [])}")
    print(f"    reformulated : {reformulated}")

    # Stage 2 ->  hard filters
    print(f"\n[2/4] Applying hard filters...")
    mask      = apply_hard_filters(df, hf)
    survivors = df[mask].copy()
    survivors["_penalty"] = survivors.apply(lambda row: data_penalty(row, hf), axis=1)
    print(f"    {len(survivors)} / {len(df)} candidates survive")

    # Zero-survivor fallback
    if len(survivors) == 0:
        print("    No survivors — falling back to full corpus for semantic pass.")
        survivors = df.copy()
        survivors["_penalty"] = 0.3

    candidate_idx = survivors.index.tolist()
    n = len(candidate_idx)
    results = []

    # Routing + Stages 3 & 4
    if n <= 5:
        print(f"\n[3/4] Skipped — too few candidates")
        print(f"[4/4] Skipped")
        for idx in candidate_idx:
            row = df.loc[idx]
            score, conf = fuse(survivors.loc[idx, "_penalty"], None, 0, None)
            results.append({"row": row, "score": score, "confidence": conf,
                             "llm_score": None, "bm25_rank": None})

    elif n <= DIRECT_LLM_THRESHOLD:
        print(f"\n[3/4] Skipped BM25 ({n} candidates ≤ threshold, going direct)")
        print(f"[4/4] LLM reranking {n} candidates...")
        batch      = [(idx, df.loc[idx]) for idx in candidate_idx]
        llm_scores = llm_rerank(batch, decomp, query)
        for idx in candidate_idx:
            ls = llm_scores.get(idx)
            score, conf = fuse(survivors.loc[idx, "_penalty"], None, 0, ls)
            results.append({"row": df.loc[idx], "score": score, "confidence": conf,
                             "llm_score": ls, "bm25_rank": None})

    else:
        print(f"\n[3/4] BM25 retrieval over {n} candidates...")
        top_idx = bm25_search(reformulated, candidate_idx, tokenized, RETRIEVE_TOP_K)
        print(f"    Narrowed to {len(top_idx)} candidates")

        if complexity == "structured":
            print(f"[4/4] Skipped LLM rerank (structured query)")
            for rank, idx in enumerate(top_idx):
                score, conf = fuse(survivors.loc[idx, "_penalty"], rank, len(top_idx), None)
                results.append({"row": df.loc[idx], "score": score, "confidence": conf,
                                 "llm_score": None, "bm25_rank": rank})
        else:
            print(f"[4/4] LLM reranking {len(top_idx)} candidates...")
            batch      = [(idx, df.loc[idx]) for idx in top_idx]
            llm_scores = llm_rerank(batch, decomp, query)
            for rank, idx in enumerate(top_idx):
                ls = llm_scores.get(idx)
                score, conf = fuse(survivors.loc[idx, "_penalty"], rank, len(top_idx), ls)
                results.append({"row": df.loc[idx], "score": score, "confidence": conf,
                                 "llm_score": ls, "bm25_rank": rank})
                
    seen = set()
    unique = []
    for r in results:
        key = r["row"].get("operational_name") or r["row"].get("website")
        if key not in seen:
            seen.add(key)
            unique.append(r)
    results = unique

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:TOP_N_RESULTS]



# --------------------- Output --------------------- #

def save_results(results: list[dict], query: str):
    # Build filename from query
    slug = re.sub(r'[^\w\s-]', '', query.lower())
    slug = re.sub(r'\s+', '_', slug.strip())[:60]
    filename = f"results_{slug}.txt"

    lines = []
    lines.append(f"Query: {query}\n")

    if not results:
        lines.append("No results found.\n")
    else:
        for i, r in enumerate(results, 1):
            row   = r["row"]
            name = row.get("operational_name")
            name = "Unknown" if not name or str(name) == "nan" else name
            cc    = (row.get("country_code") or "??").upper()
            naics = row.get("naics_label") or ""
            desc  = (row.get("description") or "")[:120]

            llm_str  = f"  llm={r['llm_score']:.2f}" if r["llm_score"]  is not None else ""
            rank_str = f"  bm25=#{r['bm25_rank']}"   if r["bm25_rank"] is not None else ""

            lines.append(f"# ------------------#--------------------- #")
            lines.append(f"Company {i}: {name}")
            lines.append(f"# ------------------#--------------------- # ")
            lines.append(f"[{r['confidence'].upper()}]  score={r['score']:.3f}{llm_str}{rank_str}")
            lines.append(f"Country: {cc}  |  NAICS: {naics}")

            emp = row.get("employee_count")
            rev = row.get("revenue")
            if pd.notna(emp) or pd.notna(rev):
                emp_str = f"Employees: {int(emp):,}" if pd.notna(emp) else ""
                rev_str = f"Revenue: ${rev:,.0f}"    if pd.notna(rev) else ""
                lines.append(f"{emp_str}  {rev_str}".strip())

            lines.append(f"Description: {desc}...")
            lines.append("")

    Path(filename).write_text("\n".join(lines), encoding="utf-8")
    print(f"Results saved to {filename}")




# ---------------------  Entry point --------------------- #


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "companies.jsonl"


    print(f"Loading {data_path}...")
    df = load_data(data_path)
    print(f"Loaded {len(df)} companies. Building BM25 index...")
    tokenized = build_bm25(df)
    print("Ready.\n")

    while True:
        try:
            query = input("Query (or Ctrl+C to exit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not query:
            continue

        results = qualify(query, df, tokenized)
        save_results(results, query)
        print()
