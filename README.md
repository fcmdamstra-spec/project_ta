# project_ta

A rule-based pipeline that classifies Reddit posts as **story** or **non-story** using syntactic, semantic, and pragmatic linguistic analysis.


## Scripts

### `main.py` — Full pipeline
Runs all three analysis modules on the dev or test dataset, combines their predictions via a weighted voting system, prints accuracy per rule to the console, and writes detailed results to `analysis_log.csv`.


### `syntaxis.py` — Syntactic analysis module
Analyses POS-tag frequencies (pronouns, proper nouns) and dependency patterns (adverbial modifiers) to classify texts. Run standalone to inspect feature distributions across the dataset and calibrate the classification rules; writes a pattern report to `syntax_patterns.txt`.


### `semantics.py` — Semantic analysis module
Uses coreference resolution (coreferee), named-entity recognition (spaCy), WordNet noun counts, and word-sense ambiguity to classify texts. Run standalone to evaluate each rule individually and calibrate thresholds; writes a pattern report to `semantic_patterns.txt`.


### `pragmatics.py` — Pragmatic analysis module
Uses sentence-level VADER sentiment (via asent) to measure emotional range, sentiment shifts, and sentence count. Run standalone to evaluate each rule and see observed averages per class; writes a pattern report to `pragmatic_patterns.txt`.


All four scripts prompt you to choose the **dev** (`d`) or **test** (`t`) dataset at runtime.