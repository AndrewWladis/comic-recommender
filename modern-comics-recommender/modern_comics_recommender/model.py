from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .data import ComicRun, PROJECT_ROOT, RatingValue


ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.pt"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.json"
SEED = 2026
MIN_LABELS = 10
MIN_RATING_SPAN = 2
EPOCHS = 1200
FEATURE_SCHEMA = "weighted_identity_semantic_group_cnn_v3"

# Scaling makes matching identities cheaper for the regularized network to use and
# more influential in cold-start distance. These are modeling priorities, not ratings.
GROUP_WEIGHTS = {
    "publisher": 2.25,
    "writers": 2.0,
    "artists": 2.0,
    "line": 1.25,
    "characters": 1.0,
}


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _canonical(value: str) -> str:
    return " ".join(value.lower().replace(".", "").split())


class PreferenceNet(nn.Module):
    """A 1D CNN over six semantic feature-group tokens."""

    def __init__(self, group_sizes: list[int]) -> None:
        super().__init__()
        if len(group_sizes) != 6 or any(size < 1 for size in group_sizes):
            raise ValueError("CNN requires six non-empty feature groups.")
        self.group_sizes = group_sizes
        self.projections = nn.ModuleList(
            nn.Linear(size, 16) for size in group_sizes
        )
        self.convolutions = nn.Sequential(
            nn.Conv1d(16, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(24, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(0.25)
        self.output = nn.Linear(48, 4)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        offset = 0
        tokens: list[torch.Tensor] = []
        for size, projection in zip(self.group_sizes, self.projections):
            group = values[:, offset : offset + size]
            tokens.append(torch.nn.functional.gelu(projection(group)))
            offset += size
        if offset != values.shape[1]:
            raise ValueError("CNN group sizes do not match the input width.")
        sequence = torch.stack(tokens, dim=2)  # batch x 16 channels x 6 groups
        convolved = self.convolutions(sequence).squeeze(-1)
        projected_summary = sequence.mean(dim=2)
        combined = torch.cat((convolved, projected_summary), dim=1)
        return self.output(self.dropout(combined))


@dataclass
class Preprocessor:
    medians: list[float]
    means: list[float]
    scales: list[float]
    vocabularies: dict[str, list[str]]
    group_weights: dict[str, float]

    def feature_group_sizes(self) -> list[int]:
        return [
            4,  # two normalized numeric values plus two missing-value indicators
            len(self.vocabularies["publisher"]),
            len(self.vocabularies["writers"]),
            len(self.vocabularies["artists"]),
            len(self.vocabularies["line"]),
            len(self.vocabularies["characters"]),
        ]

    @classmethod
    def fit(cls, numeric: torch.Tensor, runs: list[ComicRun]) -> "Preprocessor":
        filled = numeric.clone()
        medians: list[float] = []
        for column in range(numeric.shape[1]):
            present = numeric[:, column][~torch.isnan(numeric[:, column])]
            median = float(torch.median(present)) if len(present) else 0.0
            medians.append(median)
            filled[:, column] = torch.nan_to_num(numeric[:, column], nan=median)
        means = filled.mean(dim=0)
        scales = filled.std(dim=0, unbiased=False)
        scales = torch.where(scales < 1e-8, torch.ones_like(scales), scales)
        vocabularies = {
            "publisher": sorted({_canonical(run.publisher) for run in runs}),
            "line": sorted({_canonical(run.line) for run in runs}),
            "writers": sorted({_canonical(name) for run in runs for name in run.writers}),
            "artists": sorted({_canonical(name) for run in runs for name in run.artists}),
            "characters": sorted(
                {_canonical(name) for run in runs for name in run.central_characters}
            ),
        }
        return cls(medians, means.tolist(), scales.tolist(), vocabularies, GROUP_WEIGHTS.copy())

    def transform(self, numeric: torch.Tensor, runs: list[ComicRun]) -> torch.Tensor:
        if len(numeric) != len(runs):
            raise ValueError("Numeric rows and catalog runs must have equal lengths.")
        missing = torch.isnan(numeric).to(torch.float32)
        filled = numeric.clone()
        for column, median in enumerate(self.medians):
            filled[:, column] = torch.nan_to_num(filled[:, column], nan=median)
        standardized = (filled - torch.tensor(self.means)) / torch.tensor(self.scales)
        groups = [standardized, missing]
        values_by_group = {
            "publisher": lambda run: (run.publisher,),
            "line": lambda run: (run.line,),
            "writers": lambda run: run.writers,
            "artists": lambda run: run.artists,
            "characters": lambda run: run.central_characters,
        }
        for group_name in ("publisher", "writers", "artists", "line", "characters"):
            vocabulary = self.vocabularies[group_name]
            positions = {value: index for index, value in enumerate(vocabulary)}
            encoded = torch.zeros((len(runs), len(vocabulary)), dtype=torch.float32)
            for row_index, run in enumerate(runs):
                raw_values = tuple(values_by_group[group_name](run))
                value_weight = self.group_weights[group_name] / math.sqrt(len(raw_values))
                for raw_value in raw_values:
                    column = positions.get(_canonical(raw_value))
                    if column is not None:
                        encoded[row_index, column] = value_weight
            groups.append(encoded)
        return torch.cat(groups, dim=1)


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


def numeric_features(runs: list[ComicRun]) -> torch.Tensor:
    return torch.tensor(
        [
            [
                float(run.start_year),
                float("nan") if run.issue_count is None else run.issue_count,
            ]
            for run in runs
        ],
        dtype=torch.float32,
    )


def _labeled(
    runs: list[ComicRun], ratings: dict[str, RatingValue]
) -> tuple[list[ComicRun], torch.Tensor]:
    labeled_runs: list[ComicRun] = []
    labels: list[float] = []
    for run in runs:
        value = ratings.get(run.id, "unread")
        if isinstance(value, int) and not isinstance(value, bool):
            labeled_runs.append(run)
            labels.append(float(value))
    return labeled_runs, torch.tensor(labels, dtype=torch.float32)


def readiness(runs: list[ComicRun], ratings: dict[str, RatingValue]) -> TrainingReport:
    _, labels = _labeled(runs, ratings)
    distinct = len(torch.unique(labels)) if len(labels) else 0
    minimum = int(labels.min()) if len(labels) else None
    maximum = int(labels.max()) if len(labels) else None
    span = maximum - minimum if minimum is not None and maximum is not None else 0
    ready = len(labels) >= MIN_LABELS and distinct >= 2 and span >= MIN_RATING_SPAN
    message = (
        "Enough varied ratings to train the identity-aware model."
        if ready
        else f"Training is premature: add at least {MIN_LABELS} ratings spanning "
        f"{MIN_RATING_SPAN} points. Using the deterministic identity-aware fallback."
    )
    return TrainingReport(ready, message, len(labels), distinct, minimum, maximum)


def _holdout_split(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    if len(labels) < 15:
        return None
    indices = list(range(len(labels)))
    random.Random(SEED).shuffle(indices)
    validation_size = max(3, round(len(indices) * 0.2))
    validation = indices[:validation_size]
    training = [index for index in indices if index not in validation]
    return torch.tensor(training), torch.tensor(validation)


def _ordinal_targets(labels: torch.Tensor) -> torch.Tensor:
    return (labels.unsqueeze(1) > torch.arange(1, 5, dtype=torch.float32)).float()


def _fit_network(
    features: torch.Tensor, labels: torch.Tensor, group_sizes: list[int]
) -> PreferenceNet:
    set_seeds()
    model = PreferenceNet(group_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=0.005)
    loss_function = nn.BCEWithLogitsLoss()
    targets = _ordinal_targets(labels)
    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(features), targets)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _expected_rating(logits: torch.Tensor) -> torch.Tensor:
    cumulative = torch.cummin(torch.sigmoid(logits), dim=1).values
    return 1.0 + cumulative.sum(dim=1)


def train_and_save(
    runs: list[ComicRun], ratings: dict[str, RatingValue]
) -> TrainingReport:
    report = readiness(runs, ratings)
    if not report.trained:
        return report
    labeled_runs, labels = _labeled(runs, ratings)
    numeric = numeric_features(labeled_runs)
    split = _holdout_split(labels)
    if split is not None:
        train_indices, validation_indices = split
        training_runs = [labeled_runs[index] for index in train_indices.tolist()]
        validation_runs = [labeled_runs[index] for index in validation_indices.tolist()]
        evaluation_preprocessor = Preprocessor.fit(numeric[train_indices], training_runs)
        evaluation_model = _fit_network(
            evaluation_preprocessor.transform(numeric[train_indices], training_runs),
            labels[train_indices],
            evaluation_preprocessor.feature_group_sizes(),
        )
        with torch.no_grad():
            predictions = _expected_rating(
                evaluation_model(
                    evaluation_preprocessor.transform(
                        numeric[validation_indices], validation_runs
                    )
                )
            )
        report.validation_mae = float(
            torch.mean(torch.abs(predictions - labels[validation_indices]))
        )
        report.validation_count = len(validation_indices)
        evaluation_note = (
            f" Holdout MAE: {report.validation_mae:.2f} rating points on "
            f"{report.validation_count} runs."
        )
    else:
        evaluation_note = " No holdout metric: fewer than 15 numeric ratings."
    preprocessor = Preprocessor.fit(numeric, labeled_runs)
    transformed = preprocessor.transform(numeric, labeled_runs)
    group_sizes = preprocessor.feature_group_sizes()
    model = _fit_network(transformed, labels, group_sizes)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "group_sizes": group_sizes,
            "architecture": "semantic_group_conv1d_with_projection_skip",
            "epochs": EPOCHS,
            "feature_schema": FEATURE_SCHEMA,
            "seed": SEED,
        },
        MODEL_PATH,
    )
    PREPROCESSOR_PATH.write_text(
        json.dumps(asdict(preprocessor), indent=2) + "\n", encoding="utf-8"
    )
    report.message = "Trained the final ordinal model on all numeric ratings." + evaluation_note
    return report


def model_scores(runs: list[ComicRun]) -> list[float]:
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError("Model artifacts are missing; run train first.")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if checkpoint.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("Model artifacts use an incompatible feature schema; retrain them.")
    preprocessor = Preprocessor(
        **json.loads(PREPROCESSOR_PATH.read_text(encoding="utf-8"))
    )
    if checkpoint.get("architecture") != "semantic_group_conv1d_with_projection_skip":
        raise ValueError("Model artifact is not the expected CNN architecture; retrain it.")
    model = PreferenceNet([int(size) for size in checkpoint["group_sizes"]])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        return _expected_rating(
            model(preprocessor.transform(numeric_features(runs), runs))
        ).tolist()


def fallback_scores(
    runs: list[ComicRun], ratings: dict[str, RatingValue]
) -> list[float]:
    numeric = numeric_features(runs)
    preprocessor = Preprocessor.fit(numeric, runs)
    features = preprocessor.transform(numeric, runs)
    rated = [
        (index, value)
        for index, run in enumerate(runs)
        if isinstance((value := ratings.get(run.id)), int) and not isinstance(value, bool)
    ]
    publisher_prior = {"dc": 0.67, "marvel": 0.67}
    scores: list[float] = []
    for index, run in enumerate(runs):
        issue_accessibility = (
            0.55
            if run.issue_count is None
            else 1.0 / (1.0 + math.log1p(run.issue_count) / 4.0)
        )
        recency = min(max((run.start_year - 1979) / 47.0, 0.0), 1.0)
        baseline = 0.65 * publisher_prior[run.publisher.lower()] + 0.20 * issue_accessibility + 0.15 * recency
        if rated:
            rated_indices = [rated_index for rated_index, _ in rated]
            distances = torch.cdist(
                features[index].unsqueeze(0), features[rated_indices]
            ).squeeze(0)
            weights = torch.exp(-distances / 3.0)
            preferences = torch.tensor(
                [(float(value) - 1.0) / 4.0 for _, value in rated]
            )
            affinity = float((weights * preferences).sum() / weights.sum().clamp_min(1e-8))
            # Distant ratings should not exert the same force as close publisher/
            # creator matches, especially when only one answer exists.
            confidence = float(weights.max().clamp(0.0, 1.0))
            personalized_weight = 0.65 * confidence
            normalized = (1.0 - personalized_weight) * baseline + personalized_weight * affinity
        else:
            normalized = baseline
        scores.append(1.0 + 4.0 * normalized)
    return scores
