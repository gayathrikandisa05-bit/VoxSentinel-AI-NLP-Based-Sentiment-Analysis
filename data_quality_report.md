# Data Quality Report

Random seed: `42`

## Availability

| Dataset | Status |
|---|---|
| imdb | prepared |
| tweet_eval | prepared |
| go_emotions | prepared |
| semeval_absa | unavailable: no local files |
| sarc | unavailable: no local files |
| amazon_reviews | unavailable: no local files |

## imdb

- Task: `binary_sentiment_classification`
- Raw rows: 100,000
- Valid rows: 98,463
- Missing values: `{"train": {}, "test": {}, "unsupervised": {}}`
- Duplicate text rows before deduplication: 1,537
- Duplicate rows removed: 1,537
- Duplicate-text label conflicts: 620
- Invalid records: `{"missing_or_blank_text": 0, "non_string_text": 0, "invalid_or_missing_label": 0, "duplicate_rows_removed": 1537, "duplicate_text_label_conflicts": 620}`
- Split sizes: `{"train": 22290, "validation": 2489, "test": 24799, "unlabeled": 48885}`
- Class distribution by split: `{"train": {"negative": 11095, "positive": 11195}, "validation": {"negative": 1241, "positive": 1248}, "test": {"negative": 12360, "positive": 12439}, "unlabeled": {"unlabeled": 48885}}`
- Text length (characters): `{"scope": "all retained records, including unlabeled rows", "count": 98463, "min": 32, "mean": 1298.66, "median": 962.0, "p95": 3378.0, "max": 14106}`

## tweet_eval

- Task: `three_class_sentiment_classification`
- Raw rows: 59,899
- Valid rows: 59,865
- Missing values: `{"train": {}, "validation": {}, "test": {}}`
- Duplicate text rows before deduplication: 34
- Duplicate rows removed: 34
- Duplicate-text label conflicts: 6
- Invalid records: `{"missing_or_blank_text": 0, "non_string_text": 0, "invalid_or_missing_label": 0, "duplicate_rows_removed": 34, "duplicate_text_label_conflicts": 6}`
- Split sizes: `{"train": 45585, "validation": 2000, "test": 12280, "unlabeled": 0}`
- Class distribution by split: `{"train": {"negative": 7091, "neutral": 20655, "positive": 17839}, "validation": {"negative": 312, "neutral": 869, "positive": 819}, "test": {"negative": 3971, "neutral": 5934, "positive": 2375}}`
- Text length (characters): `{"scope": "all retained records, including unlabeled rows", "count": 59865, "min": 6, "mean": 103.38, "median": 109.0, "p95": 140.0, "max": 200}`

## go_emotions

- Task: `multi_label_emotion_classification`
- Raw rows: 54,263
- Valid rows: 53,943
- Missing values: `{"train": {}, "validation": {}, "test": {}}`
- Duplicate text rows before deduplication: 320
- Duplicate rows removed: 320
- Duplicate-text label conflicts: 81
- Invalid records: `{"missing_or_blank_text": 0, "non_string_text": 0, "invalid_or_missing_label": 0, "duplicate_rows_removed": 320, "duplicate_text_label_conflicts": 81}`
- Split sizes: `{"train": 43112, "validation": 5410, "test": 5421, "unlabeled": 0}`
- Class distribution by split: `{"train": {"admiration": 4107, "amusement": 2322, "anger": 1562, "annoyance": 2462, "approval": 2930, "caring": 1079, "confusion": 1361, "curiosity": 2183, "desire": 641, "disappointment": 1264, "disapproval": 2018, "disgust": 790, "embarrassment": 302, "excitement": 837, "fear": 592, "gratitude": 2590, "grief": 77, "joy": 1432, "love": 2049, "nervousness": 164, "neutral": 14153, "optimism": 1573, "pride": 110, "realization": 1107, "relief": 153, "remorse": 543, "sadness": 1318, "surprise": 1055}, "validation": {"admiration": 487, "amusement": 303, "anger": 195, "annoyance": 303, "approval": 396, "caring": 152, "confusion": 151, "curiosity": 247, "desire": 77, "disappointment": 163, "disapproval": 292, "disgust": 97, "embarrassment": 35, "excitement": 95, "fear": 90, "gratitude": 351, "grief": 13, "joy": 172, "love": 250, "nervousness": 21, "neutral": 1765, "optimism": 209, "pride": 15, "realization": 127, "relief": 18, "remorse": 68, "sadness": 143, "surprise": 129}, "test": {"admiration": 504, "amusement": 264, "anger": 198, "annoyance": 320, "approval": 351, "caring": 134, "confusion": 153, "curiosity": 284, "desire": 83, "disappointment": 151, "disapproval": 267, "disgust": 123, "embarrassment": 37, "excitement": 102, "fear": 78, "gratitude": 351, "grief": 6, "joy": 160, "love": 237, "nervousness": 23, "neutral": 1786, "optimism": 186, "pride": 16, "realization": 145, "relief": 11, "remorse": 56, "sadness": 156, "surprise": 141}}`
- Text length (characters): `{"scope": "all retained records, including unlabeled rows", "count": 53943, "min": 1, "mean": 68.58, "median": 65.0, "p95": 131.0, "max": 703}`
