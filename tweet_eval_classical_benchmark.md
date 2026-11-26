# tweet_eval Classical Sentiment Baselines

- Train / validation / test: 45,585 / 2,000 / 12,280
- Classes: negative, neutral, positive
- Model selection: validation macro F1; selected `linear_svm`.

| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 | Test weighted F1 | Fit seconds | Test records/sec | Artifact MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.6910 | 0.6450 | 0.5934 | 0.5569 | 0.5728 | 3.04 | 108468.6 | 1.05 |
| linear_svm | 0.6830 | 0.6527 | 0.5843 | 0.5698 | 0.5808 | 1.59 | 107046.2 | 1.62 |

## Test confusion matrices

### logistic_regression

Label order: `['negative', 'neutral', 'positive']`

```text
[[1212, 2319, 440], [457, 4576, 901], [70, 806, 1499]]
```

### linear_svm

Label order: `['negative', 'neutral', 'positive']`

```text
[[1737, 1786, 448], [896, 3888, 1150], [120, 705, 1550]]
```

## Limitations

- TF-IDF and linear classifiers are lexical baselines; they do not model word order beyond unigrams and bigrams.
- The vectorizer lowercases text; it does not preserve casing as a feature.
- Reported confidence is not calibrated; LinearSVC does not provide class probabilities.
- IMDb and TweetEval use different source domains and label taxonomies; scores are not directly comparable across datasets.
