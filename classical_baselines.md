# Classical sentiment baselines

## Method

Phase 3 trains separate TF-IDF + Logistic Regression and TF-IDF + Linear SVM pipelines for IMDb and TweetEval sentiment. The prepared train split is used for fitting. Validation macro-F1 selects the preferred classifier per dataset. Test metrics are reported separately and are not used for selection.

Each vectorizer uses the configured 1–2 gram range and 50,000 feature cap, sublinear term frequency, and float32 sparse values. Logistic Regression and Linear SVM use the configured `C=1.0`; the Logistic Regression iteration cap is 1,000. The seed is read from `configs/model_config.yaml` (42 by default).

## Results from this run

| Dataset | Model | Validation accuracy | Validation macro-F1 | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---|---:|---:|---:|---:|---:|
| IMDb | Logistic Regression | 0.8931 | 0.8931 | 0.9017 | 0.9017 | 0.9017 |
| IMDb | Linear SVM | 0.9012 | 0.9012 | 0.9007 | 0.9007 | 0.9007 |
| TweetEval sentiment | Logistic Regression | 0.6910 | 0.6450 | 0.5934 | 0.5569 | 0.5728 |
| TweetEval sentiment | Linear SVM | 0.6830 | 0.6527 | 0.5843 | 0.5698 | 0.5808 |

Validation macro-F1 selects Linear SVM for both datasets. Test behavior shows why the selection split and the final report are distinct: on IMDb, Logistic Regression has slightly higher test macro-F1; on TweetEval, Linear SVM has higher test macro-F1 while Logistic Regression has higher test accuracy. The benchmark reports both instead of selecting on the test set.

Full per-class metrics, confusion matrices, fit and inference timings, artifact sizes, parameters, and scikit-learn version are recorded in the JSON and Markdown reports under `artifacts/reports/`.

## Limitations

- These are lexical baselines; they do not understand compositional context beyond unigram and bigram features.
- Lowercasing means case is not a feature. Punctuation and emoji are available to the tokenizer, but this sparse representation does not model emoji sequences or sentiment context robustly.
- Linear SVM produces decision scores rather than calibrated class probabilities. Logistic Regression probabilities are also not calibrated by this phase.
- IMDb and TweetEval differ in domain and label taxonomy; their metrics should not be read as a direct model comparison.
- The TweetEval validation-to-test macro-F1 gap indicates weaker generalization on its test split. Per-class results in the benchmark should guide later error analysis.
- No model is promoted as production-ready based on these results.
