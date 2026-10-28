from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .data import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_RATINGS_PATH,
    ComicRun,
    RatingValue,
    load_catalog,
    load_ratings,
    save_ratings,
    validate_catalog,
)
from .model import fallback_scores, model_scores, readiness, train_and_save


def _year_label(run: ComicRun) -> str:
    if run.end_year is None:
        return f"{run.start_year}-present"
    return str(run.start_year) if run.start_year == run.end_year else f"{run.start_year}-{run.end_year}"


def quiz(
    runs: list[ComicRun],
    ratings: dict[str, RatingValue],
    ratings_path: Path,
    include_answered: bool,
    publisher: str | None,
) -> None:
    queue = [run for run in runs if publisher is None or run.publisher.lower() == publisher]
    if not include_answered:
        queue = [run for run in queue if run.id not in ratings]
    if not queue:
        print("Nothing to review. Use --all or change --publisher.")
        return
    print("\nModern DC/Marvel reading checklist")
    print("1=horrible, 2=poor, 3=okay, 4=great, 5=amazing; u=unread, s=skip, q=quit.\n")
    for position, run in enumerate(queue, start=1):
        current = ratings.get(run.id, "unread")
        print(f"[{position}/{len(queue)}] {run.title} ({_year_label(run)})")
        print(f"  {run.publisher} / {run.line}")
        print(f"  Writer(s): {', '.join(run.writers)}")
        print(f"  Artist(s): {', '.join(run.artists)}")
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
            value: RatingValue
            if answer in {"1", "2", "3", "4", "5"}:
                value = int(answer)
            elif answer in {"u", "unread"}:
                value = "unread"
            else:
                print("  Enter 1, 2, 3, 4, 5, u, s, or q.")
                continue
            ratings[run.id] = value
            save_ratings(ratings, ratings_path)
            break
    print(f"Checklist complete. Saved to {ratings_path}")


def recommend(
    runs: list[ComicRun], ratings: dict[str, RatingValue], limit: int, train: bool
) -> None:
    unread = [run for run in runs if ratings.get(run.id, "unread") == "unread"]
    if not unread:
        print("No unread runs remain.")
        return
    report = readiness(runs, ratings)
    if train and report.trained:
        report = train_and_save(runs, ratings)
        print(report.message)
    elif train:
        print(report.message)
    if report.trained:
        scores = model_scores(unread)
        method = "weighted PyTorch predicted rating"
    else:
        all_scores = fallback_scores(runs, ratings)
        by_id = dict(zip((run.id for run in runs), all_scores))
        scores = [by_id[run.id] for run in unread]
        method = "deterministic identity-aware estimate"
    ranked = sorted(zip(unread, scores), key=lambda item: (-item[1], item[0].title))
    print(f"\nTop unread recommendations — {method}")
    for rank, (run, score) in enumerate(ranked[:limit], start=1):
        print(
            f"{rank:>2}. {run.title:<52} {score:>4.2f}/5  "
            f"{run.publisher} / {run.line} | {', '.join(run.writers)} | {', '.join(run.artists)}"
        )


def validate(runs: list[ComicRun], ratings: dict[str, RatingValue]) -> int:
    errors = validate_catalog(runs)
    unknown = sorted(set(ratings) - {run.id for run in runs})
    errors.extend(f"ratings: unknown id {run_id!r}" for run_id in unknown)
    if errors:
        print("Validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    publishers = {publisher: sum(run.publisher == publisher for run in runs) for publisher in ("DC", "Marvel")}
    numeric = sum(isinstance(value, int) for value in ratings.values())
    print(
        f"Catalog valid: {len(runs)} runs ({publishers['DC']} DC, {publishers['Marvel']} Marvel), "
        f"{numeric} numeric ratings, {sum(value == 'unread' for value in ratings.values())} unread."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m modern_comics_recommender",
        description="Rate and rank 550 acclaimed modern DC and Marvel comics and runs.",
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    quiz_parser = commands.add_parser("quiz", help="Run the 1-5 reading checklist")
    quiz_parser.add_argument("--all", action="store_true", help="Include answered runs")
    quiz_parser.add_argument("--publisher", choices=("dc", "marvel"))
    recommend_parser = commands.add_parser("recommend", help="Rank unread runs")
    recommend_parser.add_argument("--limit", type=int, default=15)
    recommend_parser.add_argument("--no-train", action="store_true")
    commands.add_parser("train", help="Train when enough varied ratings exist")
    commands.add_parser("validate", help="Validate catalog and ratings")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        runs = load_catalog(args.catalog)
        ratings = load_ratings(args.ratings)
        unknown = sorted(set(ratings) - {run.id for run in runs})
        if unknown:
            raise ValueError("Ratings contain unknown catalog id(s): " + ", ".join(unknown))
        if args.command == "quiz":
            quiz(runs, ratings, args.ratings, args.all, args.publisher)
            return 0
        if args.command == "recommend":
            if args.limit < 1:
                raise ValueError("--limit must be at least 1")
            recommend(runs, ratings, args.limit, not args.no_train)
            return 0
        if args.command == "train":
            report = train_and_save(runs, ratings)
            print(report.message)
            print(f"Labels: {report.labeled_count}; distinct ratings: {report.distinct_ratings}.")
            return 0
        if args.command == "validate":
            return validate(runs, ratings)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0
