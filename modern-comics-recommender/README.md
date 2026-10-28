# Modern DC/Marvel comics recommender

An independent local recommender covering 550 acclaimed modern comics and runs: 275
DC and 275 Marvel. It does not replace or modify the Geoff Johns project.

## Setup and use

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m modern_comics_recommender quiz
python -m modern_comics_recommender recommend
```

The quiz accepts `1` (horrible) through `5` (amazing), `u` for unread, `s` to skip,
and `q` to save and quit. Answers save immediately in `data/ratings.json`. Filter a
session with `quiz --publisher dc` or `quiz --publisher marvel`, and revisit answers
with `quiz --all`.

Other commands:

```bash
python -m modern_comics_recommender validate
python -m modern_comics_recommender train
python -m modern_comics_recommender recommend --limit 25
```

## What the model learns

The catalog stores actual publisher, line, writer names, artist names, central
characters, years, and issue counts. It intentionally has no subjective creator
notability fields. Publisher, writer, and artist identities receive the strongest
feature scaling, followed by line and central characters. For example, a strong
rating for a Grant Morrison/Frank Quitely DC run can transfer through three distinct
signals: publisher, writer, and artist.

Before ten sufficiently varied numeric ratings exist, recommendations use a
deterministic identity-aware similarity ranking. At the threshold, a regularized
PyTorch 1D CNN trains cumulative ordinal targets for the ordered 1-5 scale. With at
least fifteen ratings, 20% are held out and the app reports mean absolute error in
rating points before fitting the final model on all ratings. Unread and skipped rows
are never labels.

The four network outputs estimate `P(rating > 1)` through `P(rating > 4)` and combine
into a predicted 1-5 rating. Python and PyTorch use fixed seed `2026`; missing issue
counts are median-imputed with an explicit missing-value feature.

The CNN first projects six semantic groups—numeric values, publisher, writers,
artists, line, and central characters—into 16-channel tokens. Two `Conv1d` layers
(16→24 and 24→32 channels, kernel size 3) learn interactions between neighboring
groups. The pooled convolution output is concatenated with a direct summary of the
projected group tokens, which prevents the small CNN from collapsing to a constant
prediction. GELU activations, adaptive pooling, 25% dropout, and the four ordinal
outputs follow. It trains for 1,200 epochs, increased from the original 600.

See `DATA_NOTES.md` for selection scope, sources, credit limitations, categorical
weights, and the complete feature policy.

## Tests

```bash
python -m unittest discover -s tests -v
```
