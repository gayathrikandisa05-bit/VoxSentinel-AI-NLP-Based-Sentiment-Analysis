# Phase 2 dataset sources and use notes

Phase 2 processes only datasets already present under `data/raw/`. Dataset-specific labels and tasks stay separate. This inventory is a provenance check, not legal advice or permission to redistribute source text.

| Dataset | Local files | Task and label format | Source / use notes |
|---|---|---|---|
| IMDb | Present; 25,000 train + 25,000 test + 50,000 unlabeled rows per local manifest | Binary sentiment; `0=negative`, `1=positive`; unsupervised records remain unlabeled | [Hugging Face dataset card](https://huggingface.co/datasets/stanfordnlp/imdb) identifies license as `other`; review the original dataset terms before redistribution. |
| TweetEval sentiment | Present; train/validation/test per local manifest | Three-class sentiment; `0=negative`, `1=neutral`, `2=positive` | [TweetEval dataset card](https://huggingface.co/datasets/cardiffnlp/tweet_eval) describes different subset licenses and Twitter Terms of Service. The current card metadata says `unknown`, while its licensing section lists CC-BY 3.0 for sentiment; treat terms as requiring review and follow the original platform terms. |
| GoEmotions simplified | Present; train/validation/test per local manifest | Multi-label emotion classification; integer IDs map to 28 named labels | [Hugging Face dataset card](https://huggingface.co/datasets/google-research-datasets/go_emotions) lists Apache-2.0. |
| SemEval ABSA | Directory exists but contains no files | Aspect term/category polarity labels not loaded | [Official task page](https://alt.qcri.org/semeval2014/task4/) describes the dataset. No local data was fabricated or downloaded. Confirm dataset terms for the chosen data release before use. |
| SARC | No local files | Sarcasm classification not loaded | [Princeton SARC page](https://nlp.cs.princeton.edu/SARC/) is the retrieval reference; the page could not be fetched during this phase. No download was attempted; check current access and use terms before retrieval. |
| Amazon Reviews 2023 | No local files | Review rating/sentiment task not loaded | [Dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) and its [license discussion](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/1) state the authors cannot assign a dataset license. No data was downloaded. |

## Preparation decisions

- Source splits for TweetEval and GoEmotions are preserved. Exact normalized-text duplicates across splits are removed with test rows taking precedence, then validation, then train, to avoid train/holdout leakage.
- IMDb has no validation split in the local manifest. Its official test split is preserved; 10% of labeled train data is assigned to validation using a fixed seed and per-class shuffling.
- IMDb's unsupervised split is retained separately and excluded from labeled sentiment class statistics.
- GoEmotions is kept as multi-label emotion classification. Its label indices use the dataset card order and are written as names; sentiment labels are not inferred.
- Cleaning applies Unicode normalization, HTML entity/tag handling, URL replacement with `<URL>`, and whitespace normalization. It retains negation, mentions, hashtags, contractions, punctuation, emoji, and original case.
- Cleaned text and original text are both retained in processed records. Duplicate comparison uses case-folded cleaned text; this intentionally collapses case-only variants.
