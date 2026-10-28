# Geoff Johns reading recommender

A local, terminal-based checklist and recommender built from the supplied
`works.txt`. It keeps the source unchanged, stores 1-5 reading ratings in readable
JSON, works before any ratings exist, and uses a small ordinal PyTorch neural network
once the user has supplied enough varied examples.

## Setup

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch 2.x is the only runtime dependency. The project also works with an existing
Python environment that already has PyTorch installed.

## Checklist workflow

Start or resume the checklist:

```bash
python3 -m geoff_recommender quiz
```

For each readable work, enter a rating from 1 through 5:

- `1` — horrible
- `2` — poor
- `3` — okay
- `4` — great
- `5` — amazing

Enter `u` for unread, `s` to skip without saving an answer, or `q` to save and quit.
Answers save immediately to `data/ratings.json`. Run `quiz --all` to revisit existing
answers. The JSON may also be edited directly; values must be integer `1`-`5` or the
string `"unread"`. A skipped work is simply absent from the ratings object.

## Migration from the former binary format

Any CLI command detects a schema-version-1 ratings file before loading it. The app
first copies the original to `ratings.json.v1.backup` (or a numbered backup if that
name already exists), then writes schema version 2 using the explicit conversion
`liked` → `5`, `disliked` → `1`, and `unread` → `unread`. It prints the backup path and
records the mapping in the converted JSON. Unknown legacy values stop migration with
an error; the original is never silently reinterpreted or overwritten without a
backup. Old model artifacts are rejected and must be retrained on the converted data.

## Recommendations and training

```bash
python3 -m geoff_recommender recommend
python3 -m geoff_recommender recommend --limit 20
python3 -m geoff_recommender train
python3 -m geoff_recommender validate
```

Unread and skipped rows are never training labels. Before there are at least six
numeric ratings using two or more values that span at least two scale points, the app
clearly reports that training is premature. It ranks unread works with a deterministic
fallback combining central-character similarity, character prominence, artist
notability, recency, reading-length accessibility, and similarity to every available
personal rating. Both
fallback and model results are displayed as intuitive estimated ratings from 1 to 5.

At the threshold, the app median-imputes missing numeric features, adds missing-value
indicators, standardizes the numeric features, and trains a small regularized
PyTorch network with fixed seed `2026`:

1. release start year;
2. main character/team prominence (1-10, with 10 most prominent);
3. representative artist notability (1-5, with 5 most notable);
4. issue/volume count.
5. multi-hot canonical central-character/team tags derived from the catalog's
   `character_or_team` metadata.

Related identities share a franchise tag—for example, Wally West and Barry Allen both
activate `flash`, while retaining their specific identity tags. The same approach
connects Hal Jordan/Green Lantern, Justice Society variants, Shazam/Billy Batson, and
other repeated characters or teams. Character tags never present in the training
ratings are safely ignored by that trained model rather than assigned random meaning.
The cold-start fallback can still compare all catalog character tags.

The model uses cumulative ordinal targets for the four ordered thresholds (`>1`,
`>2`, `>3`, and `>4`) with binary cross-entropy loss. Its output is converted to an
expected 1-5 rating while enforcing monotonic cumulative probabilities. This retains
the ordering information that ordinary five-class classification would ignore.

With ten or more numeric ratings, 20% (at least two) are held out for a disclosed
mean-absolute-error sanity check measured in rating points. A final model is then fit
on every rating for ranking. With less data, the app states that no holdout metric was
computed; it never presents training-set performance as evaluation. Artifacts are
written under `artifacts/` only after the training threshold is met.

`recommend` retrains by default so edited answers are reflected. Use `--no-train` only
to reuse compatible existing artifacts. The result is deterministic on the same
supported PyTorch/CPU stack.

## Data schema and limitations

The derived catalog is `data/works.csv`; `works.txt` remains the untouched source.
Important columns are `id`, `title`, `release_start_year`, `release_end_year`,
`character_or_team`, `character_prominence`, `artist_names`, `artist_notability`,
`issue_count`, `item_type`, `recommendable`, `issue_count_basis`, and
`metadata_confidence`. See `DATA_NOTES.md` for the catalog scoring rubric, missing-data
policy, source limitations, overlapping entries, and the excluded non-reading row.

This is a preference aid, not a canonical bibliography. Artist credits are
representative rather than exhaustive, several contribution/issue counts are unknown
or approximate, and the neural model remains statistically fragile with few ratings.

## Tests

```bash
python3 -m unittest discover -s tests -v
```
