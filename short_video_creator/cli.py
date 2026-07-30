import argparse
import concurrent.futures
import logging
import multiprocessing
import os
from pathlib import Path
import time

from .errors import PreflightError
from .pipeline import create_short, preflight
from .probe import list_video_files
from .settings import Settings

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.WARNING),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _worker(input_path: Path, output_path: Path, backgrounds_dir: Path) -> Path:
    configure_logging()
    start_time = time.time()
    logger.info("Processing: %s", input_path.name)
    try:
        return create_short(input_path, output_path, backgrounds_dir)
    finally:
        logger.info("Runtime: %s - %s", round(time.time() - start_time, 2), input_path.name)


def build_parser() -> argparse.ArgumentParser:
    cwd = Path.cwd()
    parser = argparse.ArgumentParser(description="Create captioned short videos in a batch.")
    parser.add_argument("--input-dir", type=Path, default=cwd / "INPUT_VIDEOS")
    parser.add_argument("--output-dir", type=Path, default=cwd / "OUTPUT_VIDEOS")
    parser.add_argument("--backgrounds-dir", type=Path, default=cwd / "BACKGROUND_VIDEOS")
    parser.add_argument("--processes", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    if args.processes < 1:
        logger.error("--processes must be at least 1")
        return 1
    args.input_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        preflight(args.backgrounds_dir, Settings())
    except PreflightError as error:
        logger.error("%s", error)
        return 1
    pending = list_video_files(args.input_dir)
    logger.info("STARTED")
    failures = []
    if pending:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.processes, mp_context=context
        ) as executor:
            futures = {
                executor.submit(
                    _worker,
                    args.input_dir / file_name,
                    args.output_dir / file_name,
                    args.backgrounds_dir,
                ): file_name
                for file_name in pending
            }
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception:
                    failures.append(name)
                    logger.exception("Worker failed: %s", name)
    if failures:
        logger.error(
            "%d of %d videos failed: %s",
            len(failures),
            len(pending),
            ", ".join(sorted(failures)),
        )
        return 1
    logger.info("MAIN PROCESS COMPLETE")
    return 0
