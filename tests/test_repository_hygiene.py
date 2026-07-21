from __future__ import annotations

import subprocess
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRECTORIES = {"runs", "weights", "checkpoints"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx"}


class RepositoryHygieneTest(unittest.TestCase):
    def test_git_tracks_no_outputs_checkpoints_or_weights(self) -> None:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        tracked = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        violations = []
        for path in tracked:
            parts = set(path.parts)
            if parts & FORBIDDEN_DIRECTORIES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
                violations.append(path.as_posix())
            if path.parts and path.parts[0] == "outputs" and path.name != ".gitkeep":
                violations.append(path.as_posix())
        self.assertEqual(violations, [])
