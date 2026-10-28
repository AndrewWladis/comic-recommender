from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from torch import nn

from modern_comics_recommender.data import (
    load_catalog,
    load_ratings,
    save_ratings,
    validate_catalog,
)
from modern_comics_recommender.model import (
    EPOCHS,
    GROUP_WEIGHTS,
    Preprocessor,
    PreferenceNet,
    fallback_scores,
    model_scores,
    numeric_features,
    readiness,
    train_and_save,
)


class ModernComicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runs = load_catalog()

    def test_catalog_has_balanced_550_runs(self) -> None:
        self.assertEqual(len(self.runs), 550)
        self.assertEqual(Counter(run.publisher for run in self.runs), {"DC": 275, "Marvel": 275})
        self.assertEqual(validate_catalog(self.runs), [])

    def test_requested_lines_are_present(self) -> None:
        lines = {run.line for run in self.runs}
        self.assertTrue({"Mainline", "Black Label", "Absolute Universe"}.issubset(lines))
        self.assertIn("Ultimate Universe (2023)", lines)
        self.assertIn("Ultimate Universe (2000)", lines)

    def test_every_run_has_identity_metadata(self) -> None:
        for run in self.runs:
            self.assertTrue(run.writers, run.id)
            self.assertTrue(run.artists, run.id)
            self.assertTrue(run.central_characters, run.id)

    def test_publisher_writer_artist_are_emphasized(self) -> None:
        self.assertGreater(GROUP_WEIGHTS["publisher"], GROUP_WEIGHTS["line"])
        self.assertGreater(GROUP_WEIGHTS["writers"], GROUP_WEIGHTS["characters"])
        self.assertGreater(GROUP_WEIGHTS["artists"], GROUP_WEIGHTS["characters"])
        preprocessor = Preprocessor.fit(numeric_features(self.runs), self.runs)
        self.assertIn("grant morrison", preprocessor.vocabularies["writers"])
        self.assertIn("marco checchetto", preprocessor.vocabularies["artists"])
        self.assertEqual(set(preprocessor.vocabularies["publisher"]), {"dc", "marvel"})

    def test_model_is_a_two_convolution_cnn_with_more_epochs(self) -> None:
        preprocessor = Preprocessor.fit(numeric_features(self.runs), self.runs)
        model = PreferenceNet(preprocessor.feature_group_sizes())
        convolutions = [module for module in model.modules() if isinstance(module, nn.Conv1d)]
        self.assertEqual(len(convolutions), 2)
        self.assertGreater(EPOCHS, 600)

    def test_ratings_round_trip_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.json"
            expected = {self.runs[0].id: 5, self.runs[75].id: 1, self.runs[1].id: "unread"}
            save_ratings(expected, path)
            self.assertEqual(load_ratings(path), expected)
            for invalid in (0, 6, 3.5, True, "5"):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    save_ratings({self.runs[0].id: invalid}, path)  # type: ignore[dict-item]

    def test_cold_start_fallback_is_deterministic_and_bounded(self) -> None:
        ratings = {self.runs[0].id: 5, self.runs[75].id: 1}
        first = fallback_scores(self.runs, ratings)
        second = fallback_scores(self.runs, ratings)
        self.assertEqual(first, second)
        self.assertTrue(all(1 <= value <= 5 for value in first))

    def test_creator_and_publisher_matches_affect_fallback_similarity(self) -> None:
        by_id = {run.id: index for index, run in enumerate(self.runs)}
        baseline = fallback_scores(self.runs, {})
        scores = fallback_scores(self.runs, {"dc-watchmen": 5})
        same_writer_index = by_id["dc-promethea"]
        unrelated_index = by_id["marvel-silver-surfer-black"]
        same_writer_uplift = scores[same_writer_index] - baseline[same_writer_index]
        unrelated_uplift = scores[unrelated_index] - baseline[unrelated_index]
        self.assertGreater(same_writer_uplift, unrelated_uplift)

    def test_training_and_saved_identity_vocabularies(self) -> None:
        sample = self.runs[:5] + self.runs[75:80]
        ratings = {run.id: (1 if index < 5 else 5) for index, run in enumerate(sample)}
        self.assertTrue(readiness(self.runs, ratings).trained)
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            model_path = artifact_dir / "model.pt"
            preprocessor_path = artifact_dir / "preprocessor.json"
            with patch("modern_comics_recommender.model.EPOCHS", 40), patch(
                "modern_comics_recommender.model.ARTIFACT_DIR", artifact_dir
            ), patch(
                "modern_comics_recommender.model.MODEL_PATH", model_path
            ), patch(
                "modern_comics_recommender.model.PREPROCESSOR_PATH", preprocessor_path
            ):
                report = train_and_save(self.runs, ratings)
                self.assertTrue(report.trained)
                payload = json.loads(preprocessor_path.read_text(encoding="utf-8"))
                self.assertIn("writers", payload["vocabularies"])
                self.assertIn("artists", payload["vocabularies"])
                self.assertIn("publisher", payload["vocabularies"])
                scores = model_scores(self.runs[20:23])
                self.assertEqual(len(scores), 3)
                self.assertTrue(all(1 <= value <= 5 for value in scores))
                self.assertGreater(max(scores) - min(scores), 1e-5)

    def test_holdout_uses_unseen_examples(self) -> None:
        sample = self.runs[:20]
        ratings = {run.id: 1 + index % 5 for index, run in enumerate(sample)}
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            with patch("modern_comics_recommender.model.EPOCHS", 40), patch(
                "modern_comics_recommender.model.ARTIFACT_DIR", artifact_dir
            ), patch(
                "modern_comics_recommender.model.MODEL_PATH", artifact_dir / "model.pt"
            ), patch(
                "modern_comics_recommender.model.PREPROCESSOR_PATH",
                artifact_dir / "preprocessor.json",
            ):
                report = train_and_save(self.runs, ratings)
        self.assertEqual(report.validation_count, 4)
        self.assertIsNotNone(report.validation_mae)


if __name__ == "__main__":
    unittest.main()
