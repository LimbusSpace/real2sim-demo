from __future__ import annotations

from pathlib import Path

from PIL import Image

from real2sim_demo.gaussian import parse_hyworld_evaluation, prepare_scaled_images


def test_prepare_scaled_images_creates_factor_directory(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (16, 12), (10, 20, 30)).save(images_dir / "frame.png")

    scaled_dir = prepare_scaled_images(tmp_path, factor=4)

    assert scaled_dir == tmp_path / "images_4"
    with Image.open(scaled_dir / "frame.png") as image:
        assert image.size == (4, 3)


def test_parse_hyworld_evaluation_uses_final_ansi_colored_result(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "PSNR: 12.0, SSIM: 0.5, LPIPS: 0.4 Time: 0.1s/image Number of GS: 100\n"
        "\x1b[32mPSNR: 15.208, SSIM: 0.6822, LPIPS: 0.421 "
        "Time: 0.047s/image Number of GS: 7498\x1b[0m\n",
        encoding="utf-8",
    )

    assert parse_hyworld_evaluation(log_path) == {
        "psnr": 15.208,
        "ssim": 0.6822,
        "lpips": 0.421,
        "gaussian_count": 7498,
    }


def test_parse_hyworld_evaluation_returns_empty_for_missing_metrics(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text("training started\n", encoding="utf-8")

    assert parse_hyworld_evaluation(log_path) == {}
