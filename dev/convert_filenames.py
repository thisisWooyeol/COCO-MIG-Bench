#!/usr/bin/env python3
"""
Script to convert image filenames from the current format to the expected format for MIG Bench evaluation.

Current format: "<prompt>_<number1>_<number2>.png"
Expected format: "<prompt_idx>_<itr>_<level>_<prompt>.[png|jpg]"
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict


def load_dataset(jsonl_path: str) -> Dict[str, dict]:
    """Load the JSONL dataset and create a mapping from prompt to sample data."""
    prompt_to_sample = {}

    with open(jsonl_path, "r") as f:
        for idx, line in enumerate(f):
            sample = json.loads(line.strip())
            sample["idx"] = idx  # Add the index
            prompt_to_sample[sample["prompt"]] = sample

    return prompt_to_sample


def extract_info_from_filename(filename: str) -> tuple:
    """
    Extract information from current filename format.

    Current format: "<prompt>_<number1>_<number2>.png"
    Returns: (prompt, number1, number2, extension)
    """
    path = Path(filename)
    stem = path.stem
    extension = path.suffix

    # Find the last two numbers separated by underscores
    pattern = r"^(.+)_(\d+)_(\d+)$"
    match = re.match(pattern, stem)

    if not match:
        raise ValueError(f"Filename {filename} doesn't match expected pattern")

    prompt = match.group(1)
    number1 = int(match.group(2))
    number2 = int(match.group(3))

    return prompt, number1, number2, extension


def create_new_filename(prompt_idx: int, itr: int, level: int, prompt: str, extension: str) -> str:
    """
    Create new filename in expected format.

    Format: "<prompt_idx>_<itr>_<level>_<prompt>.[png|jpg]"
    """
    # Clean the prompt to make it filesystem-safe
    safe_prompt = re.sub(r'[<>:"/\\|?*]', "_", prompt)
    return f"{prompt_idx}_{itr}_{level}_{safe_prompt}{extension}"


def convert_filenames(image_dir: str, jsonl_path: str, dry_run: bool = True, default_itr: int = 0):
    """
    Convert all image filenames in the directory to the expected format.

    Args:
        image_dir: Directory containing the images
        jsonl_path: Path to the JSONL dataset file
        dry_run: If True, only print what would be renamed without actually renaming
        default_itr: Default iteration number to use (since current format doesn't specify iteration)
    """
    # Load dataset
    print(f"Loading dataset from {jsonl_path}...")
    prompt_to_sample = load_dataset(jsonl_path)
    print(f"Loaded {len(prompt_to_sample)} samples.")

    # Get all image files
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        raise FileNotFoundError(f"Image directory {image_dir} does not exist")

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    print(f"Found {len(image_files)} image files.")

    successful_renames = 0
    failed_renames = []

    for filename in image_files:
        try:
            # Extract info from current filename
            prompt, number1, number2, extension = extract_info_from_filename(filename)

            # Find matching sample in dataset
            if prompt not in prompt_to_sample:
                print(f"Warning: Prompt '{prompt}' not found in dataset. Skipping {filename}")
                failed_renames.append((filename, "Prompt not found in dataset"))
                continue

            sample = prompt_to_sample[prompt]
            prompt_idx = sample["idx"]
            level = sample["level"]

            # Create new filename
            new_filename = create_new_filename(prompt_idx, default_itr, level, prompt, extension)

            old_path = image_dir_path / filename
            new_path = image_dir_path / new_filename

            if dry_run:
                print(f"Would rename: {filename} -> {new_filename}")
            else:
                if new_path.exists():
                    print(f"Warning: Target file {new_filename} already exists. Skipping {filename}")
                    failed_renames.append((filename, "Target file already exists"))
                    continue

                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")

            successful_renames += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            failed_renames.append((filename, str(e)))

    # Print summary
    print("\nSummary:")
    print(f"Successfully processed: {successful_renames}")
    print(f"Failed: {len(failed_renames)}")

    if failed_renames:
        print("\nFailed files:")
        for filename, reason in failed_renames:
            print(f"  {filename}: {reason}")

    if dry_run:
        print("\nThis was a dry run. Use --no-dry-run to actually rename files.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert image filenames to MIG Bench expected format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview changes)
  python dev/convert_filenames.py --image-dir ./example/infer_coco_mig_check --dataset ./data/mig_bench.jsonl

  # Actually rename files
  python dev/convert_filenames.py --image-dir ./example/infer_coco_mig_check --dataset ./data/mig_bench.jsonl --no-dry-run

  # Use custom iteration number
  python dev/convert_filenames.py --image-dir ./example/infer_coco_mig_check --dataset ./data/mig_bench.jsonl --iteration 5 --no-dry-run
        """,
    )

    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing the images to rename")

    parser.add_argument("--dataset", type=str, required=True, help="Path to the JSONL dataset file")

    parser.add_argument(
        "--iteration", type=int, default=0, help="Iteration number to use in the new filenames (default: 0)"
    )

    parser.add_argument("--no-dry-run", action="store_true", help="Actually rename files (default is dry run mode)")

    args = parser.parse_args()

    convert_filenames(
        image_dir=args.image_dir, jsonl_path=args.dataset, dry_run=not args.no_dry_run, default_itr=args.iteration
    )


if __name__ == "__main__":
    main()
