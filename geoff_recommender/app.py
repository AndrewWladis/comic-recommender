from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_RATINGS_PATH,
    RatingValue,
    Work,
    load_catalog,
    load_ratings,
    migrate_legacy_ratings,
    save_ratings,
    validate_catalog,
)
from .model import fallback_scores, model_scores, readiness, train_and_save


CHOICES: dict[str, RatingValue] = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "u": "unread",
    "unread": "unread",
}


def year_label(work: Work) -> str:
    if work.release_end_year is None:
        return f"{work.release_start_year}-present/announced"
    if work.release_end_year == work.release_start_year:
        return str(work.release_start_year)
    return f"{work.release_start_year}-{work.release_end_year}"


def quiz(
    works: list[Work],
    ratings: dict[str, RatingValue],
    include_rated: bool,
    ratings_path: Path,
) -> None:
    eligible = [work for work in works if work.recommendable]
    queue = eligible if include_rated else [work for work in eligible if work.id not in ratings]
    if not queue:
        print("Nothing to review. Use --all to revisit saved answers.")
        return
    print("\nGeoff Johns reading checklist")
    print("Rate each read work from 1-5: 1=horrible, 2=poor, 3=okay, 4=great, 5=amazing.")
    print("Enter [u]nread, [s]kip, or [q]uit when appropriate. Answers save immediately.\n")
    for position, work in enumerate(queue, start=1):
        current = ratings.get(work.id, "unread")
        print(f"[{position}/{len(queue)}] {work.title} ({year_label(work)})")
        print(f"  {work.character_or_team} | artist(s): {work.artist_names or 'unknown'}")
        while True:
            try:
                answer = input(f"  rating [{current}] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nStopped; previous answers are saved.")
                return
            if answer in {"q", "quit"}:
                print("Stopped; previous answers are saved.")
                return
            if answer in {"", "s", "skip"}:
                break
            rating = CHOICES.get(answer)
            if rating is None:
                print("  Please enter 1, 2, 3, 4, 5, u, s, or q.")
                continue
            ratings[work.id] = rating
            save_ratings(ratings, ratings_path)
            break
    print(f"Checklist complete. Saved to {ratings_path}")


def recommend(
    works: list[Work], ratings: dict[str, RatingValue], limit: int, train: bool
) -> None:
    unread = [
        work for work in works
        if work.recommendable and ratings.get(work.id, "unread") == "unread"
    ]
    if not unread:
        print("No unread works remain. Mark a title unread in the quiz or ratings JSON.")
        return
    report = readiness(works, ratings)
    if train and report.trained:
        report = train_and_save(works, ratings)
        print(report.message)
    elif train:
        print(report.message)

    if report.trained:
        scores = model_scores(unread)
        method = "PyTorch predicted rating"
    else:
        fallback_works = [work for work in works if work.recommendable]
        scores = fallback_scores(fallback_works, ratings)
        score_by_id = dict(zip((work.id for work in fallback_works), scores))
        scores = [score_by_id[work.id] for work in unread]
        method = "deterministic estimated rating"

    ranked = sorted(zip(unread, scores), key=lambda item: (-item[1], item[0].title))
    print(f"\nTop unread recommendations — {method}")
    for rank, (work, score) in enumerate(ranked[:limit], start=1):
        issues = "unknown issues" if work.issue_count is None else f"{int(work.issue_count)} issue(s)"
        print(
            f"{rank:>2}. {work.title:<38} {score:>4.2f}/5  "
            f"{year_label(work)}, {issues}, {work.artist_names or 'artist unknown'}"
        )


def validate(works: list[Work], ratings: dict[str, RatingValue]) -> int:
    errors = validate_catalog(works)
    known_ids = {work.id for work in works}
    unknown_ratings = sorted(set(ratings) - known_ids)
    errors.extend(f"ratings: unknown work id {work_id!r}" for work_id in unknown_ratings)
    if errors:
        print("Validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    recommendable = sum(work.recommendable for work in works)
    missing_issues = sum(work.recommendable and work.issue_count is None for work in works)
    numeric_count = sum(isinstance(value, int) and not isinstance(value, bool) for value in ratings.values())
    unread_count = sum(value == "unread" for value in ratings.values())
    print(
        f"Catalog valid: {len(works)} source entries, {recommendable} readable works, "
        f"{missing_issues} safely missing issue counts, {numeric_count} numeric ratings, "
        f"{unread_count} explicit unread answers."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m geoff_recommender",
        description="Local Geoff Johns reading checklist and PyTorch recommender.",
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    quiz_parser = subparsers.add_parser("quiz", help="Rate works from 1 (horrible) to 5 (amazing)")
    quiz_parser.add_argument("--all", action="store_true", help="Include already rated works")
    recommendation_parser = subparsers.add_parser("recommend", help="Rank unread works")
    recommendation_parser.add_argument("--limit", type=int, default=10)
    recommendation_parser.add_argument(
        "--no-train", action="store_true", help="Reuse artifacts instead of retraining"
    )
    subparsers.add_parser("train", help="Train and save the model when enough labels exist")
    subparsers.add_parser("validate", help="Validate catalog and ratings")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        works = load_catalog(args.catalog)
        backup = migrate_legacy_ratings(args.ratings)
        if backup is not None:
            print(
                "Migrated legacy ratings to the 1-5 scale "
                f"(liked -> 5, disliked -> 1); original backup: {backup}"
            )
        ratings = load_ratings(args.ratings)
        unknown_ids = sorted(set(ratings) - {work.id for work in works})
        if unknown_ids:
            raise ValueError(
                "Ratings contain unknown catalog id(s): " + ", ".join(unknown_ids)
            )
        if args.command == "quiz":
            quiz(works, ratings, args.all, args.ratings)
            return 0
        if args.command == "recommend":
            if args.limit < 1:
                raise ValueError("--limit must be at least 1")
            recommend(works, ratings, args.limit, not args.no_train)
            return 0
        if args.command == "train":
            report = train_and_save(works, ratings)
            print(report.message)
            rating_range = (
                "none"
                if report.minimum_rating is None
                else f"{report.minimum_rating}-{report.maximum_rating}"
            )
            print(
                f"Labels: {report.labeled_count} total, {report.distinct_ratings} distinct, "
                f"range {rating_range}."
            )
            return 0
        if args.command == "validate":
            return validate(works, ratings)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0
