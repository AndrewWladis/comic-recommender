from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .data import PROJECT_ROOT, RatingValue, Work


ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.pt"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.json"
SEED = 2026
MIN_LABELS = 6
MIN_RATING_SPAN = 2
FEATURE_NAMES = [
    "release_start_year",
    "character_prominence",
    "artist_notability",
    "issue_count",
]
FEATURE_SCHEMA = "numeric_missing_plus_character_multihot_v1"


CHARACTER_ALIASES = {
    "the avengers": "avengers",
    "the flash": "flash",
    "barry allen": "flash",
    "wally west": "flash",
    "hal jordan": "green lantern",
    "green lantern corps": "green lantern",
    "the green lantern corps": "green lantern",
    "justice society": "justice society of america",
    "jsa": "justice society of america",
    "justice league of america": "justice league",
    "dc ensemble": "dc universe",
    "dc universe ensemble": "dc universe",
    "the joker": "joker",
    "shazam family": "shazam",
    "billy batson": "shazam",
    "tariq geiger": "geiger",
    "simon pure": "redcoat",
}


def character_tags(work: Work) -> tuple[str, ...]:
    """Return stable, shared character/team tags from catalog metadata."""
    normalized = work.character_or_team.lower().replace("/", ";").replace(" and ", ";")
    tags: set[str] = set()
    for raw_tag in normalized.split(";"):
        tag = raw_tag.strip()
        if not tag or tag == "unspecified":
            continue
        canonical = CHARACTER_ALIASES.get(tag, tag)
        tags.add(canonical)
        # Preserve specific identities when they add information beyond a franchise tag.
        if tag in {"barry allen", "wally west", "hal jordan", "billy batson"}:
            tags.add(tag)
    return tuple(sorted(tags))


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class PreferenceNet(nn.Module):
    def __init__(self, input_size: int = 8) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(8, 4),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


@dataclass
class Preprocessor:
    medians: list[float]
    means: list[float]
    scales: list[float]
    feature_names: list[str]
    character_vocabulary: list[str]

    @classmethod
    def fit(cls, rows: torch.Tensor, works: list[Work]) -> "Preprocessor":
        medians: list[float] = []
        filled = rows.clone()
        for column in range(rows.shape[1]):
            present = rows[:, column][~torch.isnan(rows[:, column])]
            median = float(torch.median(present)) if len(present) else 0.0
            medians.append(median)
            filled[:, column] = torch.nan_to_num(rows[:, column], nan=median)
        means_tensor = filled.mean(dim=0)
        scales_tensor = filled.std(dim=0, unbiased=False)
        scales_tensor = torch.where(scales_tensor < 1e-8, torch.ones_like(scales_tensor), scales_tensor)
        return cls(
            medians=medians,
            means=means_tensor.tolist(),
            scales=scales_tensor.tolist(),
            feature_names=FEATURE_NAMES.copy(),
            character_vocabulary=sorted(
                {tag for work in works for tag in character_tags(work)}
            ),
        )

    def transform(self, rows: torch.Tensor, works: list[Work]) -> torch.Tensor:
        if len(rows) != len(works):
            raise ValueError("Feature rows and works must have the same length.")
        missing = torch.isnan(rows).to(torch.float32)
        filled = rows.clone()
        for column, median in enumerate(self.medians):
            filled[:, column] = torch.nan_to_num(filled[:, column], nan=median)
        means = torch.tensor(self.means, dtype=torch.float32)
        scales = torch.tensor(self.scales, dtype=torch.float32)
        numeric = torch.cat(((filled - means) / scales, missing), dim=1)
        vocabulary_index = {
            tag: index for index, tag in enumerate(self.character_vocabulary)
        }
        characters = torch.zeros(
            (len(works), len(self.character_vocabulary)), dtype=torch.float32
        )
        for row_index, work in enumerate(works):
            for tag in character_tags(work):
                column = vocabulary_index.get(tag)
                if column is not None:
                    characters[row_index, column] = 1.0
        return torch.cat((numeric, characters), dim=1)


@dataclass
class TrainingReport:
    trained: bool
    message: str
    labeled_count: int
    distinct_ratings: int
    minimum_rating: int | None
    maximum_rating: int | None
    validation_mae: float | None = None
    validation_count: int = 0
    epochs: int = 0


def raw_features(works: list[Work]) -> torch.Tensor:
    return torch.tensor(
        [
            [
                float(work.release_start_year),
                float("nan") if work.character_prominence is None else work.character_prominence,
                float("nan") if work.artist_notability is None else work.artist_notability,
                float("nan") if work.issue_count is None else work.issue_count,
            ]
            for work in works
        ],
        dtype=torch.float32,
    )


def _labeled(
    works: list[Work], ratings: dict[str, RatingValue]
) -> tuple[list[Work], torch.Tensor]:
    labeled_works: list[Work] = []
    labels: list[float] = []
    for work in works:
        rating = ratings.get(work.id, "unread")
        if work.recommendable and isinstance(rating, int) and not isinstance(rating, bool):
            labeled_works.append(work)
            labels.append(float(rating))
    return labeled_works, torch.tensor(labels, dtype=torch.float32)


def readiness(works: list[Work], ratings: dict[str, RatingValue]) -> TrainingReport:
    _, labels = _labeled(works, ratings)
    distinct = len(torch.unique(labels)) if len(labels) else 0
    minimum = int(labels.min().item()) if len(labels) else None
    maximum = int(labels.max().item()) if len(labels) else None
    span = (maximum - minimum) if minimum is not None and maximum is not None else 0
    ready = len(labels) >= MIN_LABELS and distinct >= 2 and span >= MIN_RATING_SPAN
    if ready:
        message = "Enough varied 1-5 ratings to train the neural recommender."
    else:
        message = (
            f"Training is premature: rate at least {MIN_LABELS} works and use at least two "
            f"values spanning {MIN_RATING_SPAN} points on the 1-5 scale. "
            "Using deterministic fallback."
        )
    return TrainingReport(ready, message, len(labels), distinct, minimum, maximum)


def _holdout_split(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    if len(labels) < 10:
        return None
    indices = list(range(len(labels)))
    random.Random(SEED).shuffle(indices)
    validation_count = max(2, round(len(indices) * 0.2))
    validation_indices = indices[:validation_count]
    training_indices = [index for index in range(len(labels)) if index not in validation_indices]
    return torch.tensor(training_indices), torch.tensor(validation_indices)


def _fit_network(features: torch.Tensor, labels: torch.Tensor, epochs: int = 500) -> PreferenceNet:
    set_seeds()
    model = PreferenceNet(features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=0.02)
    loss_function = nn.BCEWithLogitsLoss()
    thresholds = torch.arange(1, 5, dtype=torch.float32)
    ordinal_targets = (labels.unsqueeze(1) > thresholds).to(torch.float32)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_function(logits, ordinal_targets)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _expected_ratings(logits: torch.Tensor) -> torch.Tensor:
    cumulative = torch.sigmoid(logits)
    monotonic = torch.cummin(cumulative, dim=1).values
    return 1.0 + monotonic.sum(dim=1)


def train_and_save(works: list[Work], ratings: dict[str, RatingValue]) -> TrainingReport:
    report = readiness(works, ratings)
    if not report.trained:
        return report

    labeled_works, labels = _labeled(works, ratings)
    raw = raw_features(labeled_works)
    split = _holdout_split(labels)
    if split is not None:
        train_indices, validation_indices = split
        training_works = [labeled_works[index] for index in train_indices.tolist()]
        validation_works = [labeled_works[index] for index in validation_indices.tolist()]
        evaluation_preprocessor = Preprocessor.fit(raw[train_indices], training_works)
        evaluation_model = _fit_network(
            evaluation_preprocessor.transform(raw[train_indices], training_works),
            labels[train_indices],
        )
        with torch.no_grad():
            predictions = _expected_ratings(
                evaluation_model(
                    evaluation_preprocessor.transform(raw[validation_indices], validation_works)
                )
            )
        report.validation_mae = float(
            torch.mean(torch.abs(predictions - labels[validation_indices]))
        )
        report.validation_count = len(validation_indices)
        evaluation_note = (
            f" Holdout mean absolute error: {report.validation_mae:.2f} rating points "
            f"on {report.validation_count} ratings."
        )
    else:
        evaluation_note = " No holdout metric: the labeled set is too small for a meaningful split."

    preprocessor = Preprocessor.fit(raw, labeled_works)
    transformed = preprocessor.transform(raw, labeled_works)
    model = _fit_network(transformed, labels)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": transformed.shape[1],
            "output_type": "cumulative_ordinal_1_to_5",
            "feature_schema": FEATURE_SCHEMA,
            "seed": SEED,
        },
        MODEL_PATH,
    )
    PREPROCESSOR_PATH.write_text(
        json.dumps(asdict(preprocessor), indent=2) + "\n", encoding="utf-8"
    )
    report.epochs = 500
    training_note = (
        "Trained a final model on all numeric ratings after the disclosed holdout check."
        if split is not None
        else "Trained a final model on all numeric ratings."
    )
    report.message = training_note + evaluation_note
    return report


def model_scores(works: list[Work]) -> list[float]:
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError("Model artifacts are missing; run the train command first.")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("output_type") != "cumulative_ordinal_1_to_5"
        or checkpoint.get("feature_schema") != FEATURE_SCHEMA
    ):
        raise ValueError(
            "Model artifact uses an older feature or rating format; "
            "rerun recommend without --no-train."
        )
    payload = json.loads(PREPROCESSOR_PATH.read_text(encoding="utf-8"))
    preprocessor = Preprocessor(**payload)
    model = PreferenceNet(int(checkpoint["input_size"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        return _expected_ratings(
            model(preprocessor.transform(raw_features(works), works))
        ).tolist()


def fallback_scores(works: list[Work], ratings: dict[str, RatingValue]) -> list[float]:
    """Deterministic baseline, personalized by nearby rated works when available."""
    features = raw_features(works)
    global_preprocessor = Preprocessor.fit(features, works)
    all_features = global_preprocessor.transform(features, works)
    numeric_features = all_features[:, :4]
    character_features = all_features[:, 8:]
    transformed = torch.cat((numeric_features, character_features * 1.25), dim=1)
    years = features[:, 0]
    year_span = max(float(years.max() - years.min()), 1.0)
    scores: list[float] = []
    rated = [
        (index, value)
        for index, work in enumerate(works)
        if isinstance((value := ratings.get(work.id)), int) and not isinstance(value, bool)
    ]
    for index, work in enumerate(works):
        prominence = (work.character_prominence or 5.0) / 10.0
        artist = (work.artist_notability or 3.0) / 5.0
        recency = (work.release_start_year - float(years.min())) / year_span
        issues = work.issue_count
        accessibility = 0.5 if issues is None else 1.0 / (1.0 + math.log1p(issues) / 4.0)
        editorial = 0.40 * prominence + 0.30 * artist + 0.15 * recency + 0.15 * accessibility
        if rated:
            rated_indices = [rated_index for rated_index, _ in rated]
            distances = torch.cdist(
                transformed[index].unsqueeze(0), transformed[rated_indices]
            ).squeeze(0)
            weights = torch.exp(-distances / 2.0)
            preferences = torch.tensor(
                [(float(rating) - 1.0) / 4.0 for _, rating in rated], dtype=torch.float32
            )
            affinity = float((weights * preferences).sum() / weights.sum().clamp_min(1e-8))
        else:
            affinity = editorial
        normalized_score = 0.65 * editorial + 0.35 * affinity
        scores.append(1.0 + 4.0 * normalized_score)
    return scores
