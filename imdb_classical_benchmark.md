# imdb Classical Sentiment Baselines

- Train / validation / test: 22,290 / 2,489 / 24,799
- Classes: negative, positive
- Model selection: validation macro F1; selected `linear_svm`.

| Model | Validation accuracy | Validation macro F1 | Test accuracy | Test macro F1 | Test weighted F1 | Fit seconds | Test records/sec | Artifact MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.8931 | 0.8931 | 0.9017 | 0.9017 | 0.9017 | 5.19 | 8058.8 | 0.73 |
| linear_svm | 0.9012 | 0.9012 | 0.9007 | 0.9007 | 0.9007 | 5.47 | 8203.1 | 0.92 |

## Test confusion matrices

### logistic_regression

Label order: `['negative', 'positive']`

```text
[[11054, 1306], [1131, 11308]]
```

### linear_svm

Label order: `['negative', 'positive']`

```text
[[11140, 1220], [1243, 11196]]
```

## Limitations

- TF-IDF and linear classifiers are lexical baselines; they do not model word order beyond unigrams and bigrams.
- The vectorizer lowercases text; it does not preserve casing as a feature.
- Reported confidence is not calibrated; LinearSVC does not provide class probabilities.
- IMDb and TweetEval use different source domains and label taxonomies; scores are not directly comparable across datasets.
