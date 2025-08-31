#!/usr/bin/env python3
"""
Convert MIG bench data from original format to JSONL format.

This script converts the original MIG benchmark data (mig_bench.json)
to a JSONL format suitable for AIBL's specific use case, parsing the
existing captions, segments, bboxes, and labels.
"""

import argparse
import json
import re
from typing import Tuple


def parse_colored_label(label: str) -> Tuple[str, str]:
    """
    Parse a label like "a blue fork" to extract color and object.

    Args:
        label: Label string like "a blue fork"

    Returns:
        Tuple of (color, object)
    """
    # Remove "a " or "an " from the beginning
    cleaned_label = re.sub(r"^(a|an)\s+", "", label.strip())

    # Common colors to look for
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray", "grey"]

    # Find color in the label
    found_color = ""
    remaining_text = cleaned_label

    for color in colors:
        if cleaned_label.startswith(color + " "):
            found_color = color
            remaining_text = cleaned_label[len(color) :].strip()
            break

    return found_color, remaining_text


def convert_mig_bench_to_jsonl(mig_bench_path: str, output_path: str) -> None:
    """
    Convert MIG bench JSON to JSONL format by parsing existing data.

    Args:
        mig_bench_path: Path to input mig_bench.json file
        output_path: Path to output JSONL file
    """
    # Load the MIG bench data
    with open(mig_bench_path, "r") as f:
        mig_data = json.load(f)

    jsonl_entries = []

    for idx, entry in mig_data.items():
        # Extract basic information
        caption = entry["caption"]
        segments = entry["segment"]
        image_id = entry.get("image_id", "")

        # Extract phrases and bounding boxes from segments
        phrases = []
        bounding_boxes = []
        colors = []
        objects = []

        for segment in segments:
            label = segment["label"]
            bbox = segment["bbox"]

            # Parse color and object from label
            color, obj = parse_colored_label(label)

            phrases.append(label)
            bounding_boxes.append(bbox)
            colors.append(color)
            objects.append(obj)

        num_objects = len(phrases)

        # Create the prompt from caption (clean it up)
        prompt = caption.strip()
        # # Remove extra spaces and clean up the caption
        # prompt = re.sub(r"\s+", " ", prompt)
        # if not prompt.endswith("."):
        #     prompt += "."

        # Create JSONL entry
        entry_data = {
            "prompt": prompt,
            "phrases": phrases,
            "bounding_boxes": bounding_boxes,
            "num_objects": num_objects,
            "num_bboxes": num_objects,
            "image_id": image_id,
        }

        # Add expected objects and colors (up to 6)
        for i in range(num_objects):
            obj_key = f"expected_obj{i + 1}"
            color_key = f"color{i + 1}"

            entry_data[obj_key] = objects[i]
            entry_data[color_key] = colors[i]

        # Level: 0 - 5 (computer index friendly)
        entry_data["level"] = num_objects - 2
        jsonl_entries.append(entry_data)

    # Sort entries by level (ascending order)
    jsonl_entries.sort(key=lambda x: x["level"])

    # Write to JSONL file
    with open(output_path, "w") as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Converted {len(jsonl_entries)} entries from {mig_bench_path} to {output_path} (sorted by level)")


def main():
    parser = argparse.ArgumentParser(description="Convert MIG bench data to JSONL format")
    parser.add_argument("--input", "-i", default="dev/mig_bench.json", help="Input MIG bench JSON file path")
    parser.add_argument("--output", "-o", default="data/mig_bench.jsonl", help="Output JSONL file path")

    args = parser.parse_args()

    # Convert the data
    convert_mig_bench_to_jsonl(mig_bench_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
