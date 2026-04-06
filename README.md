# Intent-Qualification


# 1. Approach

## The core problem

A search index finds candidates based on their proximity.  We need something else: a system that checks to see if each candidate really meets the user's needs. 
This is fundamentally an **intent-to-entity** matching problem under real-world limitations. The **user** asks for something in natural language, a **database** holds structured **company-profiles**, and the system must determine whether each company actually meets the **criteria**, and isn't just topic related.

The system is designed around this distinction.

---

## Let's talk architecture:

The pipeline has **five** stages, but with an important observation: not every query will be sent through all 5 stages. The architecture only sends each query through the stages it really needs, based on a complexity classification made at the time of decomposition.

This is a visuals schema of the architecture:
[Schema](Schema.png)

The design was chosen after better analysing the naive approach of sing LLMs. While they offer great accuracy and strong semantics, Large Language Models are not magic, even at a small sample size(of 477) companies, there's a lot of potential to optimize the use of LLMs, and this approach reduces LLM calls by a wide margin. Complexity of the query directly affects the cost/time of the ranking. 

---

### Stage 1 - Query decomposition

One LLM call parses the raw query into a structured object:

- **hard_filters**: fields with direct database equivalents (`country_code`, `is_public`, `min_employees`, `min_revenue`, `min_year_founded`, `naics_keywords`). These are applied as pandas boolean masks with zero API cost.
- **soft_filters**: semantic criteria about what the company must do or be ("provides HR software as a B2B SaaS product"). These cannot be checked by field lookup.
- **inferred_constraints**: things the query implies but does not state ("startup implies small, recent, probably private", "competing with banks implies NOT a bank").
- **complexity**: one of `structured` / `mixed` / `judgment` — drives all routing decisions downstream.
- **reformulated_query**: a role-aware rewrite optimised for text retrieval. The key design decision here is that the reformulation describes what the _target company does_, not the user's domain. As an **example**: "Cosmetics packaging" becomes "a company that manufactures or supplies packaging materials to consumer goods brands." This prevents the embedding/retrieval stage from matching cosmetics brands instead of their suppliers.


Advantage compared to **BASELINE A** -> Instead of sending each company individually to the LLM, we're only calling the LLM for the query itself. 

### Stage 2 - Hard filter pass:

Applied as vectorised pandas operations over the entire dataset. The most important design choice here is how to handle missing data. For example, a company with a null `employee_count` is not thrown out, but instead applied a confidence penalty.  If you drop nulls, you will silently remove valid candidates. Real-world datasets are incomplete, and not having data doesn't mean that there isn't a match.

The penalty is additive (0.12 per constrained-but-null field, capped at 0.40) and flows into the final score fusion rather than being used as a hard cutoff.

Geographic filtering uses an extensive region map. When the LLM returns `"scandinavia"` or `"europe"` as a country code, the system expands it to the full list of `SO 3166-1 alpha-2` codes for that region before applying the filter. The region map covers all major global groupings: Scandinavia, Western/Eastern/Southern Europe, MENA, APAC, Americas, sub-regions.

### Stage 3 - BM25 retrieval

BM25 ([Okapi BM25](https://en.wikipedia.org/w/index.php?title=Okapi_BM25&oldid=1330270429)) is used as the retrieval stage for large candidate pools. The implementation builds a **sub-corpus from survivors only** on each query call, not the full dataset. This is the most important scaling choice: if hard filters cut 100,000 companies down to 800, BM25 runs on 800 documents. Instead of being kept globally, the index is rebuilt for each query from the survivor subset. This keeps memory usage low and makes sure that retrieval is always over the most relevant candidate pool.

The reformulated query from Stage 1 is used as the BM25 query, not the raw user query. This is what prevents the **role-confusion failure mode**.

### Stage 4 - LLM batch reranking

Applied only when the complexity label is `mixed` or `judgment`, and only over the top-K candidates from BM25 (default: top 50). Companies are sent in batches of 10 per API call, each batch producing a 0–1 score and a one-sentence reason per company.

The rerank prompt explicitly provides the soft filters and inferred constraints as the evaluation rubric. The LLM is not asked "does this match the query?",  it is asked "does this company satisfy these specific criteria?" This produces more consistent, more auditable scores.

### Stage 5 - Score fusion

Three signals are combined with weights that shift based on which signals are available:

| Signals available          | Weights                        |
| -------------------------- | ------------------------------ |
| LLM score + BM25 rank      | LLM 0.60, BM25 0.25, base 0.15 |
| LLM score only             | LLM 0.75, base 0.25            |
| BM25 rank only             | BM25 0.65, base 0.35           |
| Neither (hard filter only) | base 1.0                       |
Where:
- `base` is `1.0 - hard_penalty`, where `hard_penalty` accumulates from missing constrained fields.
- Confidence tiers: **high** ($\geq$ 0.72), **probable** ($\geq$ 0.45), **borderline** ($\geq$ 0.45).
- **Deduplication** runs before the final sort, keyed on `operational_name` (falling back to `website`). This removes duplicate records in the source dataset from polluting the top-10 output.

---
# 2. Tradeoffs:

## What was optimised for:

- **Cost over raw accuracy.** The LLM is the only genuinely expensive component. The architecture is designed so that the LLM only ever sees candidates that have already been pre-filtered by cheap stages.
	- Example on the given dataset: At 477 companies and 10 queries, the total cost was $0.134. The LLM rerank stager an on at most 50 companies per query, batched into ~5 API calls, rather than 477 sequential calls.
-  **Scalability of the retrieval stage**.  **BM25** was chosen over dense embeddings specifically because it requires no network access, no GPU, and builds a sub-corpus per query rather than maintaining a global vector index. At 100k companies this matters more.
- **Consistency above flexibility**. By decomposing the query into a fixed schema first, the system applies the same rubric at every downstream stage. The LLM reranker evaluates against explicit criteria, not a vague "does this match?" prompt. This makes results more reproducible across runs.


### Intentional tradeoffs:
While great, there are a few distinct sacrifces:

- **BM25 has weaker semantic understanding than dense embeddings.** A dense embedding model (e.g. `all-MiniLM-L6-v2`) would handle paraphrase matching and cross-lingual company names better. BM25 is lexical => if the query tokens don't appear in the company's text, it scores zero regardless of semantic similarity. This is partially compensated by the reformulated query and the richness of the `embed_text` field, but it is a real limitation.
- **The complexity router can misclassify queries**. A query that looks structured but requires industry understanding *(Example: "public software companies with more than 1,000 employees". That's great, but  what counts as a software company?)*  will be routed incorrectly if the LLM calls it `structured`. The decomposition prompt now includes an explicit clarification that any industry or sector criterion forces `mixed` at minimum, but this is still LLM-dependent.
-  **LLM-dependency**. Specifically, the performance of the LLM model we're using.  For example, with Anthropic's `claude-haiku-4-5`, the results were overall solid, but less can be said about using a SML such as `qwen2.5:7b-instruct-q4_K_M`, which **hiccupped** when dealing with the `JSON Lines` format. 
- **No persistent index.** The BM25 index is rebuilt from tokenized text on every startup. At 100k companies this adds startup latency. A persistent serialized index (pickle of the tokenized corpus) would remove this.

---

# 3. Error Analysis

## Where the system struggles:

### **Queries that require industry understanding routed as structured**:

The query "public software companies with more than 1,000 employees" initially routed as `structured` and skipped LLM reranking. The result included Alcon (pharma), PulteGroup (homebuilding), and Vestas (wind turbines) -> all large, public, with >1,000 employees, but none of them software companies. BM25 cannot distinguish "large public company" from "large public software company" without the LLM pass. After adding the complexity clarification to the decomposition prompt, the system correctly routed this as `mixed` and produced clean results (EPAM, Globant, Leidos, EPAM, Fujitsu, CGI, TCS).

### **Trade associations and networks surfacing as manufacturers**:

"Food and beverage manufacturers in France" returned "Produit en Bretagne" -> a regional trade association that represents food companies, not a manufacturer itself. It passed the country filter and scored well on BM25 because its description extensively mentions food manufacturing. This ran as a `structured` query (France + food/beverage), so the LLM reranker that would have caught this was skipped. The fix would be to treat any query involving an industry category as `mixed` rather than `structured`.

### **Borderline role candidates at the tail**:

For "logistics companies in Romania," the system correctly returned Brasov Industrial Portfolio (#1, warehousing/logistics facilities), CFR (#2, national rail freight), and Portul Constanta (#3, Black Sea port). But the tail included OSCAR (petroleum distribution), Poșta Română (postal service), and Bunge Romania (agri-commodity trader). These were correctly pushed to `probable` and `borderline` confidence tiers by the LLM, so the scoring is working -> but a user who only looks at company names and ignores confidence labels would see noise in positions 7–10.

### **Fintech query missing some valid candidates**:

The "fast-growing fintech competing with traditional banks in Europe" query returned 10 results but the dataset likely contains additional qualifying companies that scored below threshold. The inferred constraint ("NOT a traditional bank") correctly excluded banking institutions, but the BM25 retrieval may have missed smaller fintech companies whose descriptions used different terminology than the reformulated query.

---

# 4. Scaling

The **architecture** was designed with 100,000 companies in mind, which means there are not as many drastic changes to make. Regardless, given the limited data-size, there's important things to note down that could(and likely will) affect the architecture's performance.

### What stays the same:
The pipeline structure. Stage 1 is O(1) with respect to the data size, only one API call per query regardless. Stage 5(score fusion, deduplication and sorting) is O(K) where K is the number of LLM reranked candidates, which caps at 50.

### What to expect:

* **Hard filter pass will remain O(n) as pandas handles it well**. Vectorised boolean operations over a 100k-row DataFrame are fast, well under a second for the filter types used here
*  **BM25 sub-corpus construction scales correctly**. The key design is that BM25 always runs on survivors, not the full corpus. If hard filters reduce 100k to 3,000 candidates, BM25 runs on 3,000. The per-query index build is the bottleneck: at 3,000 documents this is ~100ms. If survivor pools regularly exceed 10,000, pre-building the full BM25 index at startup and filtering post-scores would be faster.
* **The full corpus BM25 index at startup is the main change needed.** Currently `build_bm25()` tokenizes all records at startup (very fast at 477, ~5 seconds at 100k). The tokenized corpus should be serialized to disk and loaded, not recomputed on every run. Additionally, for very large survivor pools, building a sub-index per query becomes expensive. At this scale, it would be ideal to switch to a pre-built [FAISS index](https://faiss.ai/index.html) with a filtering step.
* **LLM reranking cost is already bounded.** The system only ever sends top-50 candidates to the LLM regardless of dataset size. 100k companies does not change API cost unless the hard filter pass is too weak to reduce the pool sufficiently. If hard filters consistently leave 50k+ survivors, the BM25 top-K retrieval handles the reduction before the LLM ever sees them.
### Recommended changes:

1)  Serialize the tokenized BM25 corpus to disk on first run, load on subsequent runs
2)  Pre-build a FAISS for the retrieval stage, replacing BM25 for large survivor pools (>5000)
3) Add caching for decomposition results on identical or near-identical queries
4) Run LLM rerank calls in parallel (asyncio) rather than sequentially per batch

--- 
# 5. Failure Modes

## When the system produces confident but incorrect results

### **Structurally identical but semantically wrong companies**

The most dangerous failure mode is a company that passes every hard filter and gets a high score on BM25 because it uses the same words as the target role, but actually plays the opposite role in a supply chain. The packaging query was the best example from testing: systems that use embedding always return cosmetics brands instead of cosmetics packaging suppliers because the vocabulary is very similar. 

### **Data quality issues propagating silently**

If a company record has an incorrect NAICS code (e.g. a solar panel installer coded as "Semiconductor Manufacturing"), it will pass NAICS-based soft boosts it should not, and its `embed_text` will contain misleading industry signals. The system has no way to detect or correct source data quality issues.

### **LLM overconfidence on plausible-but-wrong candidates**

The LLM reranker assigns scores based on the company's description and the rubric provided. If a company's description is well-written and plausibly describes the target role -> even if the company does not actually operate in that role -> the LLM will score it highly. This is particularly likely for companies with generic, broad descriptions ("provides innovative solutions to businesses across multiple sectors").

### **Region expansion false positives**

When a query specifies "Europe," the region map expands to about 40 country codes. A company based in a European country but working only in Southeast Asian markets will pass the geographic filter. The system does not have a signal for the market of operation that is separate from the headquarters country.

#### What to monitor in production:
- **Score distribution per query type**: if mean LLM scores for a query class drift above 0.90, the LLM may be over-scoring; if they drift below 0.40, the reformulated query or batch prompt may be misaligned
-  **Complexity label distribution**: track what fraction of queries are classified `structured` vs `mixed` vs `judgment`; a sudden shift suggests the decomposition prompt is behaving differently
- **Zero-survivor fallback rate**: if this triggers frequently, hard filters are too aggressive or the decomposition is hallucinating unsupported filter fields
- **Deduplication rate**: a high deduplication rate signals source data quality issues (many duplicate records) that should be addressed upstream
- **Rerank parse error rate**: logged as `[warn] rerank parse error` -> if this exceeds ~2% of batches, the rerank prompt needs adjustment for the model being used

---

Extra: 13 .txt files avaible as query results
