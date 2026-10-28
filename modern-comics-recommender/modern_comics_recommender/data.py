from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeAlias


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data" / "runs.csv"
DEFAULT_RATINGS_PATH = PROJECT_ROOT / "data" / "ratings.json"
RatingValue: TypeAlias = int | Literal["unread"]


@dataclass(frozen=True)
class ComicRun:
    id: str
    title: str
    publisher: str
    line: str
    start_year: int
    end_year: int | None
    writers: tuple[str, ...]
    artists: tuple[str, ...]
    central_characters: tuple[str, ...]
    issue_count: float | None
    issue_count_basis: str


def _split_names(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(";") if part.strip())


def _optional_number(value: str, cast: type[int] | type[float]) -> int | float | None:
    return cast(value.strip()) if value.strip() else None


def load_catalog(path: Path = DEFAULT_CATALOG_PATH) -> list[ComicRun]:
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")
    runs: list[ComicRun] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "id", "title", "publisher", "line", "start_year", "end_year",
            "writers", "artists", "central_characters", "issue_count",
            "issue_count_basis",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Catalog is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            run_id = row["id"].strip()
            if not run_id or run_id in seen:
                raise ValueError(f"Missing or duplicate id at CSV line {line_number}: {run_id!r}")
            seen.add(run_id)
            try:
                runs.append(
                    ComicRun(
                        id=run_id,
                        title=row["title"].strip(),
                        publisher=row["publisher"].strip(),
                        line=row["line"].strip(),
                        start_year=int(row["start_year"]),
                        end_year=_optional_number(row["end_year"], int),
                        writers=_split_names(row["writers"]),
                        artists=_split_names(row["artists"]),
                        central_characters=_split_names(row["central_characters"]),
                        issue_count=_optional_number(row["issue_count"], float),
                        issue_count_basis=row["issue_count_basis"].strip(),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid catalog value at CSV line {line_number}: {exc}") from exc
    return runs


def _valid_rating(value: object) -> bool:
    return value == "unread" or (
        isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 5
    )


def load_ratings(path: Path = DEFAULT_RATINGS_PATH) -> dict[str, RatingValue]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Ratings file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Ratings JSON must be an object with schema_version 1.")
    values = payload.get("ratings", {})
    if not isinstance(values, dict):
        raise ValueError("Ratings JSON must contain an object named 'ratings'.")
    ratings: dict[str, RatingValue] = {}
    for run_id, value in values.items():
        if not _valid_rating(value):
            raise ValueError(f"Invalid rating {value!r} for {run_id!r}; use 1-5 or 'unread'.")
        ratings[str(run_id)] = value  # type: ignore[assignment]
    return ratings


def save_ratings(
    ratings: dict[str, RatingValue], path: Path = DEFAULT_RATINGS_PATH
) -> None:
    invalid = [value for value in ratings.values() if not _valid_rating(value)]
    if invalid:
        raise ValueError(f"Cannot save invalid ratings: {invalid!r}")
    payload = {
        "schema_version": 1,
        "ratings": dict(sorted(ratings.items())),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_catalog(runs: list[ComicRun]) -> list[str]:
    errors: list[str] = []
    if len(runs) != 550:
        errors.append(f"expected 550 catalog rows, found {len(runs)}")
    for run in runs:
        if run.publisher not in {"DC", "Marvel"}:
            errors.append(f"{run.id}: publisher must be DC or Marvel")
        if not run.title or not run.line:
            errors.append(f"{run.id}: title and line are required")
        if not run.writers or not run.artists or not run.central_characters:
            errors.append(f"{run.id}: writer, artist, and central-character metadata are required")
        if not 1970 <= run.start_year <= 2100:
            errors.append(f"{run.id}: implausible start year {run.start_year}")
        if run.end_year is not None and run.end_year < run.start_year:
            errors.append(f"{run.id}: end year precedes start year")
        if run.issue_count is not None and run.issue_count <= 0:
            errors.append(f"{run.id}: issue count must be positive")
    return errors
