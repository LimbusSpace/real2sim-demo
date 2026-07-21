from __future__ import annotations

import hashlib
import json
from pathlib import Path

from real2sim_demo.snapshot import verify_snapshot


def test_verify_snapshot_accepts_matching_file(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    source = root / "trainer.py"
    source.write_text("print('ok')\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "trainer.py",
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert verify_snapshot(manifest, root) == []


def test_verify_snapshot_reports_changed_file(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "trainer.py").write_text("changed\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"files": [{"path": "trainer.py", "sha256": "0" * 64}]}),
        encoding="utf-8",
    )

    failures = verify_snapshot(manifest, root)

    assert len(failures) == 1
    assert failures[0].startswith("hash mismatch: trainer.py")
