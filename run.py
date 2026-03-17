import json
import logging
import os
import sys
import time
import argparse
from pathlib import Path
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run")

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
REF_DIR = DATA_DIR / "ref_audio"
REQUEST_FILE = DATA_DIR / "requests.json"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_requests() -> list[dict]:
    if not REQUEST_FILE.exists():
        logger.error(f"requests.json not found at {REQUEST_FILE}")
        sys.exit(1)
    with open(REQUEST_FILE, encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} request(s) from {REQUEST_FILE}")
    return data


def build_engine(mode: str = "standard"):
    from engine.viet_engine import VietEngine

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = os.getenv("HF_TOKEN")
    logger.info(f"Device : {device}")
    logger.info(f"Mode : {mode}")
    if hf_token:
        logger.info("HF_TOKEN : found")
    else:
        logger.info("HF_TOKEN : not set (public model, should be fine)")

    return VietEngine(device=device, hf_token=hf_token, mode=mode)


def process(engine, requests: list[dict], mode: str = "standard") -> None:
    total = len(requests)
    success = 0
    failed = 0

    for i, req in enumerate(requests, 1):
        req_id = req.get("id", f"{i:03d}")
        ref_audio = req.get("ref_audio", "")
        ref_text = req.get("ref_text", "")
        target = req.get("target", "")

        if not ref_audio or not ref_text or not target:
            logger.warning(f"[{i}/{total}] id={req_id}  SKIP — missing field")
            failed += 1
            continue

        ref_path = REF_DIR / ref_audio
        if not ref_path.exists():
            logger.warning(f"[{i}/{total}] id={req_id}  SKIP — ref audio not found: {ref_path}")
            failed += 1
            continue

        out_name = f"{req_id}_{ref_path.stem}.wav"
        out_path = OUTPUT_DIR / out_name

        logger.info(f"[{i}/{total}] id={req_id}  ref={ref_audio}")
        logger.info(f" ref_text : {ref_text[:60]}")
        logger.info(f" target   : {target[:60]}")

        t0 = time.perf_counter()
        try:
            wav = engine.clone(
                ref_audio_path = ref_path,
                ref_text = ref_text,
                target_text = target,
            )
            engine.save(wav, out_path)
            elapsed = time.perf_counter() - t0
            logger.info(f"SAVED → {out_path.name}  ({elapsed:.1f}s)")
            success += 1

        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error(f"FAILED ({elapsed:.1f}s): {e}")
            failed += 1

    logger.info(f"\n{'─'*50}")
    logger.info(f"Done  success={success}  failed={failed}  total={total}")
    logger.info(f"Output folder: {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Vietnamese Zero-Shot TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "fast"],
        default="standard",
        help="Generation mode: 'standard' for quality, 'fast' for speed (default: standard)",
    )
    args = parser.parse_args()

    requests = load_requests()
    engine = build_engine(mode=args.mode)
    process(engine, requests, mode=args.mode)


if __name__ == "__main__":
    main()
