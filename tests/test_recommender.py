from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from geoff_recommender.data import (
    load_catalog,
    load_ratings,
    migrate_legacy_ratings,
    save_ratings,
    validate_catalog,
)
from geoff_recommender.model import (
    character_tags,
    fallback_scores,
    model_scores,
    readiness,
    train_and_save,
)


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.works = load_catalog()

    def test_all_source_rows_are_represented(self) -> None:
        source_rows = [
            line for line in Path("works.txt").read_text(encoding="utf-8").splitlines()
            if line[:1].isdigit() and line.count("\t") >= 2
        ]
        self.assertEqual(len(source_rows), 48)
        self.assertEqual(len(self.works), 48)
        self.assertEqual(sum(work.recommendable for work in self.works), 47)
        self.assertFalse(next(work for work in self.works if work.id == "ghost-machine-company").recommendable)

    def test_catalog_valid_and_missing_values_preserved(self) -> None:
        self.assertEqual(validate_catalog(self.works), [])
        self.assertTrue(any(work.issue_count is None for work in self.works))

    def test_related_central_characters_share_canonical_model_tags(self) -> None:
        by_id = {work.id: work for work in self.works}
        self.assertIn("flash", character_tags(by_id["flash-wally"]))
        self.assertIn("flash", character_tags(by_id["flash-barry-2010"]))
        self.assertIn("green lantern", character_tags(by_id["green-lantern-rebirth"]))
        self.assertIn("green lantern", character_tags(by_id["green-lantern"]))

    def test_ratings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.json"
            expected = {"flashpoint": 5, "three-jokers": 1, "geiger-2021": "unread"}
            save_ratings(expected, path)
            self.assertEqual(load_ratings(path), expected)

    def test_invalid_numeric_ratings_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.json"
            for invalid in (0, 6, 3.5, True, "5"):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    save_ratings({"flashpoint": invalid}, path)  # type: ignore[dict-item]

    def test_legacy_migration_backs_up_and_converts_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.json"
            original = {
                "schema_version": 1,
                "ratings": {
                    "flashpoint": "liked",
                    "three-jokers": "disliked",
                    "geiger-2021": "unread",
                },
                "updated_at": None,
            }
            path.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")
            backup = migrate_legacy_ratings(path)
            self.assertIsNotNone(backup)
            self.assertEqual(json.loads(backup.read_text(encoding="utf-8")), original)  # type: ignore[union-attr]
            self.assertEqual(
                load_ratings(path),
                {"flashpoint": 5, "three-jokers": 1, "geiger-2021": "unread"},
            )
            migrated_payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(migrated_payload["schema_version"], 2)
            self.assertEqual(
                migrated_payload["migration"]["mapping"],
                {"liked": 5, "disliked": 1, "unread": "unread"},
            )
            self.assertIsNone(migrate_legacy_ratings(path))

    def test_fallback_is_deterministic_and_bounded(self) -> None:
        ratings = {"flashpoint": 5, "three-jokers": 1}
        first = fallback_scores(self.works, ratings)
        second = fallback_scores(self.works, ratings)
        self.assertEqual(first, second)
        self.assertTrue(all(1.0 <= score <= 5.0 for score in first))

    def test_cold_start_and_trainable_path(self) -> None:
        self.assertFalse(readiness(self.works, {}).trained)
        eligible = [work for work in self.works if work.recommendable][:6]
        ratings = {
            **{work.id: 5 for work in eligible[:3]},
            **{work.id: 1 for work in eligible[3:]},
        }
        self.assertTrue(readiness(self.works, ratings).trained)
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            model_path = artifact_dir / "model.pt"
            preprocessor_path = artifact_dir / "preprocessor.json"
            with patch("geoff_recommender.model.ARTIFACT_DIR", artifact_dir), patch(
                "geoff_recommender.model.MODEL_PATH", model_path
            ), patch("geoff_recommender.model.PREPROCESSOR_PATH", preprocessor_path):
                report = train_and_save(self.works, ratings)
                self.assertTrue(report.trained)
                self.assertIsNone(report.validation_mae)
                self.assertTrue(model_path.exists())
                self.assertTrue(preprocessor_path.exists())
                preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
                self.assertIn("character_vocabulary", preprocessor)
                self.assertGreater(len(preprocessor["character_vocabulary"]), 0)
                scores = model_scores(self.works[:2])
                self.assertEqual(len(scores), 2)
                self.assertTrue(all(1.0 <= score <= 5.0 for score in scores))

    def test_holdout_metric_uses_separate_examples_when_data_allows(self) -> None:
        eligible = [work for work in self.works if work.recommendable][:10]
        values = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        ratings = {work.id: value for work, value in zip(eligible, values)}
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            with patch("geoff_recommender.model.ARTIFACT_DIR", artifact_dir), patch(
                "geoff_recommender.model.MODEL_PATH", artifact_dir / "model.pt"
            ), patch(
                "geoff_recommender.model.PREPROCESSOR_PATH",
                artifact_dir / "preprocessor.json",
            ):
                report = train_and_save(self.works, ratings)
        self.assertEqual(report.validation_count, 2)
        self.assertIsNotNone(report.validation_mae)
        self.assertGreaterEqual(report.validation_mae or 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
