from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, TypeAlias


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data" / "works.csv"
DEFAULT_RATINGS_PATH = PROJECT_ROOT / "data" / "ratings.json"
RatingValue: TypeAlias = int | Literal["unread"]
LEGACY_STATUSES = {"unread", "liked", "disliked"}
RATING_MIN = 1
RATING_MAX = 5


@dataclass(frozen=True)
class Work:
    id: str
    title: str
    release_start_year: int
    release_end_year: int | None
    character_or_team: str
    character_prominence: float | None
    artist_names: str
    artist_notability: float | None
    issue_count: float | None
    item_type: str
    recommendable: bool
    issue_count_basis: str
    metadata_confidence: str
    significance: str


def _optional_number(raw: str, cast: type[int] | type[float]) -> int | float | None:
    value = raw.strip()
    return cast(value) if value else None


def load_catalog(path: Path = DEFAULT_CATALOG_PATH) -> list[Work]:
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")
    works: list[Work] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "id", "title", "release_start_year", "character_or_team",
            "character_prominence", "artist_names", "artist_notability",
            "issue_count", "item_type", "recommendable",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Catalog is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            work_id = row["id"].strip()
            if not work_id or work_id in seen:
                raise ValueError(f"Missing or duplicate work id at CSV line {line_number}: {work_id!r}")
            seen.add(work_id)
            try:
                works.append(
                    Work(
                        id=work_id,
                        title=row["title"].strip(),
                        release_start_year=int(row["release_start_year"]),
                        release_end_year=_optional_number(row.get("release_end_year", ""), int),
                        character_or_team=row["character_or_team"].strip(),
                        character_prominence=_optional_number(row["character_prominence"], float),
                        artist_names=row["artist_names"].strip(),
                        artist_notability=_optional_number(row["artist_notability"], float),
                        issue_count=_optional_number(row["issue_count"], float),
                        item_type=row["item_type"].strip(),
                        recommendable=row["recommendable"].strip().lower() == "true",
                        issue_count_basis=row.get("issue_count_basis", "").strip(),
                        metadata_confidence=row.get("metadata_confidence", "").strip(),
                        significance=row.get("significance", "").strip(),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid catalog value at CSV line {line_number}: {exc}") from exc
    if not works:
        raise ValueError("Catalog is empty.")
    return works


def _valid_rating(value: object) -> bool:
    return value == "unread" or (
        isinstance(value, int)
        and not isinstance(value, bool)
        and RATING_MIN <= value <= RATING_MAX
    )


def _load_ratings_payload(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ratings file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Ratings JSON must contain a top-level object.")
    return payload


def migrate_legacy_ratings(path: Path = DEFAULT_RATINGS_PATH) -> Path | None:
    """Back up and convert a schema-v1 binary file; return the backup path."""
    if not path.exists():
        return None
    payload = _load_ratings_payload(path)
    version = payload.get("schema_version")
    if version != 1:
        return None
    ratings = payload.get("ratings", {})
    if not isinstance(ratings, dict):
        raise ValueError("Legacy ratings JSON must contain an object named 'ratings'.")
    invalid = {
        str(work_id): status
        for work_id, status in ratings.items()
        if status not in LEGACY_STATUSES
    }
    if invalid:
        raise ValueError(
            "Cannot migrate schema-v1 ratings with unknown status values: "
            + ", ".join(f"{key}={value!r}" for key, value in sorted(invalid.items()))
        )

    backup = Path(str(path) + ".v1.backup")
    suffix = 1
    while backup.exists():
        backup = Path(str(path) + f".v1.backup.{suffix}")
        suffix += 1
    shutil.copy2(path, backup)
    mapping: dict[str, RatingValue] = {"liked": 5, "disliked": 1, "unread": "unread"}
    converted = {str(work_id): mapping[str(status)] for work_id, status in ratings.items()}
    save_ratings(converted, path, migrated_from=str(backup))
    return backup


def load_ratings(path: Path = DEFAULT_RATINGS_PATH) -> dict[str, RatingValue]:
    if not path.exists():
        return {}
    payload = _load_ratings_payload(path)
    version = payload.get("schema_version")
    if version == 1:
        raise ValueError(
            "Legacy schema-v1 ratings require migration before loading; run any CLI command "
            "to create a backup and convert them."
        )
    if version != 2:
        raise ValueError(f"Unsupported ratings schema_version {version!r}; expected 2.")
    ratings = payload.get("ratings", {})
    if not isinstance(ratings, dict):
        raise ValueError("Ratings JSON must contain an object named 'ratings'.")
    clean: dict[str, RatingValue] = {}
    for work_id, value in ratings.items():
        if not _valid_rating(value):
            raise ValueError(
                f"Invalid rating {value!r} for {work_id!r}; use an integer 1-5 or 'unread'."
            )
        clean[str(work_id)] = value  # type: ignore[assignment]
    return clean


def save_ratings(
    ratings: dict[str, RatingValue],
    path: Path = DEFAULT_RATINGS_PATH,
    *,
    migrated_from: str | None = None,
) -> None:
    invalid = {repr(value) for value in ratings.values() if not _valid_rating(value)}
    if invalid:
        raise ValueError(f"Cannot save invalid ratings: {sorted(invalid)}; use 1-5 or 'unread'.")
    payload = {
        "schema_version": 2,
        "ratings": dict(sorted(ratings.items())),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if migrated_from is not None:
        payload["migration"] = {
            "from_schema_version": 1,
            "mapping": {"liked": 5, "disliked": 1, "unread": "unread"},
            "backup": migrated_from,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_catalog(works: Iterable[Work]) -> list[str]:
    errors: list[str] = []
    for work in works:
        if not work.title:
            errors.append(f"{work.id}: title is empty")
        if not 1900 <= work.release_start_year <= 2100:
            errors.append(f"{work.id}: implausible release year {work.release_start_year}")
        if work.release_end_year is not None and work.release_end_year < work.release_start_year:
            errors.append(f"{work.id}: end year precedes start year")
        if work.character_prominence is not None and not 1 <= work.character_prominence <= 10:
            errors.append(f"{work.id}: prominence must be 1-10")
        if work.artist_notability is not None and not 1 <= work.artist_notability <= 5:
            errors.append(f"{work.id}: artist notability must be 1-5")
        if work.issue_count is not None and work.issue_count <= 0:
            errors.append(f"{work.id}: issue count must be positive")
        if work.recommendable and work.character_prominence is None:
            errors.append(f"{work.id}: recommendable work needs a prominence score")
    return errors
