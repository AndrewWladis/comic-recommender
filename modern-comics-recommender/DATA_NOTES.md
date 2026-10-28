# Catalog and feature notes

## Scope and selection

The catalog contains 550 curated comics and runs: 275 DC and 275 Marvel. “Modern” means the
creative era from the mid-1980s through the current publishing lines, with a few
foundational runs beginning just before 1984 when they are inseparable from modern
comics discussion. “Best” is deliberately a consensus-oriented editorial selection,
not an objective ranking. The catalog is alphabetically neutral and has no hidden
quality score—the user's 1-5 answers supply all preference labels.

Selections emphasize acclaimed sustained runs and self-contained limited series,
balanced across major heroes, teams, imprints, and creator-owned lines published by
DC or Marvel. Official publisher pages were used to confirm the current Absolute and
Ultimate line terminology. Broad critical lists and recurring reader consensus were
used as discovery inputs, then credits were curated into a consistent schema.

Useful reference points:

- DC All In and Absolute Universe: https://www.dc.com/allin
- Marvel Ultimate Spider-Man series page: https://www.marvel.com/comics/series/38809/ultimate_spider-man_%282024_-_present%29
- Marvel Ultimate Universe announcement: https://www.marvel.com/articles/comics/ultimate-invasion-new-ultimate-universe-jonathan-hickman-bryan-hitch
- Marvel series index: https://www.marvel.com/comics/series
- Recent critical discovery list: https://www.mycomicscollection.com/en/blog/the-best-modern-comic-book-runs-2010-2025-the-tier-list/

Credits list representative principal writers and artists, not every fill-in creator.
Issue counts describe the named run or core series, not every tie-in. Blank issue
counts are preserved for ongoing or structurally complex runs; approximate counts are
explicitly marked in `issue_count_basis`.

## Model features and emphasis

There are no writer- or artist-notability scores. Names are modeled directly as
multi-hot categorical identities. Each run contributes:

- normalized start year and issue count, plus missing-value indicators;
- publisher identity;
- writer identities;
- artist identities;
- publishing line identity;
- central-character/team identities.

The saved preprocessor records the exact vocabularies. To express the requested
priority, categorical inputs use these fixed scales:

| Feature group | Scale |
|---|---:|
| Publisher | 2.25 |
| Writer | 2.00 |
| Artist | 2.00 |
| Line | 1.25 |
| Central characters | 1.00 |

Scaling affects both the regularized neural model and the deterministic similarity
fallback. It makes publisher, writer, and artist matches easier for the model to use,
but it does not hard-code a preference for DC, Marvel, or any creator. The user's
ratings determine direction and magnitude.

The neural architecture is a semantic-group 1D CNN rather than a convolution over an
arbitrary alphabetized feature vector. Separate learned projections turn numeric,
publisher, writer, artist, line, and character groups into six 16-channel tokens. Two
kernel-3 convolutional layers operate across that six-token sequence before adaptive
pooling. A projection-summary skip path is concatenated before ordinal output to
preserve per-run variation when categorical inputs are sparse. Training uses 1,200
full-batch Adam epochs with dropout and weight decay; more epochs increase fitting time
but do not remove the need for holdout evaluation.

Identity values absent from the labeled training set are ignored by the trained model
rather than assigned random weights. Retraining after broader ratings expands the
vocabularies. The fallback can compare every identity in the full catalog even before
the neural threshold is reached.
