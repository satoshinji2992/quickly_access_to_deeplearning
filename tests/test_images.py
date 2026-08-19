"""Read-only integrity checks for tutorial image assets.

These tests deliberately avoid judging visual semantics.  They catch the two
mechanical failures that previously made figure fixes easy to lose: a corrupt
PNG and a task-local copy drifting away from the canonical asset.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "assets" / "images"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ImageAssetTests(unittest.TestCase):
    def test_all_committed_pngs_decode_and_have_positive_dimensions(self):
        failures = []
        for path in sorted(ROOT.rglob("*.png")):
            if ".git" in path.parts:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                if width <= 0 or height <= 0:
                    failures.append(f"{path.relative_to(ROOT)}: {width}x{height}")
            except Exception as error:  # pragma: no cover - diagnostic detail
                failures.append(f"{path.relative_to(ROOT)}: {error}")
        self.assertEqual([], failures, "invalid PNG assets:\n" + "\n".join(failures))

    def test_task_local_assets_match_their_canonical_copy(self):
        mismatches = []
        for local in sorted((ROOT / "exercises").glob("**/assets/*.png")):
            canonical = CANONICAL / local.name
            if not canonical.exists():
                mismatches.append(
                    f"{local.relative_to(ROOT)}: no assets/images/{local.name}"
                )
            elif digest(local) != digest(canonical):
                mismatches.append(
                    f"{local.relative_to(ROOT)} != {canonical.relative_to(ROOT)}"
                )
        self.assertEqual(
            [], mismatches, "task-local image copies drifted:\n" + "\n".join(mismatches)
        )

    def test_misclassification_grid_records_real_inference_provenance(self):
        path = CANONICAL / "misclassified_examples.png"
        with Image.open(path) as image:
            description = json.loads(image.info["Description"])
        self.assertEqual(description["dataset"], "official CIFAR-100 test split")
        self.assertEqual(description["dataset_size"], 10_000)
        self.assertFalse(description["synthetic_samples"])
        self.assertEqual(len(description["test_indices"]), 8)
        self.assertTrue(
            all(
                truth != prediction
                for truth, prediction in zip(
                    description["true_labels"], description["predicted_labels"]
                )
            )
        )
        self.assertEqual(len(description["checkpoint_sha256"]), 64)

    def test_figure_assets_have_public_integrity_checks(self):
        required = (
            ROOT / "scripts" / "validate_figure_content.py",
            ROOT / "scripts" / "render_cifar100_errors.py",
        )
        self.assertEqual([], [str(path.relative_to(ROOT)) for path in required if not path.is_file()])


if __name__ == "__main__":
    unittest.main()
