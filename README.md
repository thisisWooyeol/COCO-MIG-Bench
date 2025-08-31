# MIG_Bench

The MIG benchmark of CVPR2024 MIGC: Multi-Instance Generation Controller for Text-to-Image Synthesis

**_NOTE_** This is the copy of the original repository, with slight config changes. See the original [repository](https://github.com/LeyRio/MIG_Bench) for more details.

### [[Paper]](https://arxiv.org/pdf/2402.05408.pdf) [[Project Page]](https://migcproject.github.io/) [[Code]](https://github.com/limuloo/MIGC)

**MIGC: Multi-Instance Generation Controller for Text-to-Image Synthesis**

## Installation

### Environment setup with uv

```bash
uv sync
source .venv/bin/activate
```

### Checkpoints

To run the evaluation process, you need to download GroundingDINO's checkpoint:

```bash
mkdir -p pretrained

# Download the GroundingDINO checkpoint:
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O pretrained/groundingdino_swint_ogc.pth
```

## Evaluation Pipeline

### Step 1 Generation

The MIG bench data specified in `dev/mig_bench.json` was downloaded from Google Drive [URL](https://drive.google.com/drive/folders/1mXxO7miVqgTq3N6q2QS7gFp_ML-qpsw2?usp=sharing), which is the original dataset.

Yet, for AIBL's specific use case, we modify the prompts and layouts in `jsonl` format to better suit our requirements. Example line:

```json
{
  "prompt": "a orange car, a red cat, a yellow banana and a green dog.",
  "phrases": ["orange car", "red cat", "yellow banana", "green dog"],
  "bounding_boxes": [
    [0.09765625, 0.1953125, 0.390625, 0.48828125],
    [0.46875, 0.234375, 0.6640625, 0.41015625],
    [0.72265625, 0.1171875, 0.87890625, 0.25390625],
    [0.2734375, 0.5859375, 0.556640625, 0.80078125]
  ],
  "num_objects": 4,
  "num_bboxes": 4,
  "expected_obj1": "car",
  "expected_obj2": "cat",
  "expected_obj3": "banana",
  "expected_obj4": "dog",
  "color1": "orange",
  "color2": "red",
  "color3": "yellow",
  "color4": "green",
  "level": 4
}
```

Level should be equal to the number of phrases (i.e., the number of distinct objects). We used the automatic conversion script `dev/convert_mig_bench_to_jsonl.py` to facilitate this process.

```bash
python dev/convert_mig_bench_to_jsonl.py
```

This outputs `data/mig_bench.jsonl`, which we already created.
We expect to generate images with `data/mig_bench.jsonl` dataset. The generated images should be contained in a single folder and should follow the naming convention: `<prompt_idx>_<itr>_<level>_<prompt>.[png|jpg]`.

### Step 2 Evaluation

Finally, you can start evaluating your model now.

```bash
python run_migbench.py \
    --image_dir /path/of/image/ \
    --metric_name 'eval' \
    --need_miou_score \
    --need_instance_success_ratio \
    --num_iters 8  # we expect 8 images for each prompt are sampled with different initial seeds
```

**_Bypassing image generation_**

If you want to bypass the image generation step and use existing images, you can use the pre-generated images from MIGC authors in Google Drive [URL](https://drive.google.com/drive/folders/1UyhNpZ099OTPy5ILho2cmWkiOH2j-FrB). Then, convert the filenames to the expected format using the provided script:

```bash
unzip migc_output.zip
mv infer_coco_mig_check/ example/
python dev/convert_filenames.py --image-dir ./example/infer_coco_mig_check --dataset ./data/mig_bench.jsonl
```

Now you can test evaluation as follows:

```bash
python run_migbench.py \
    --image_dir ./example/infer_coco_mig_check/ \
    --metric_name 'MIGC' \
    --need_miou_score \
    --need_instance_success_ratio  \
    --num_iters 1
```

You will get the following results in `./output/metric_MIGC.json`:

```json
{
  "metric_name": "MIGC",
  "image_path": "./example/infer_coco_mig_check/",
  "inst_success_ratio": 0.664375,
  "inst_level_success_ratio": [
    0.740625, 0.6729166666666667, 0.6703125, 0.6325, 0.6572916666666667
  ],
  "miou": 0.5696216019807909,
  "miou_level": [
    0.638372193888771, 0.576004312635142, 0.5694519321768042,
    0.5401029994419481, 0.5682253313359809
  ]
}
```

## Evaluation Results

Accumulate your evaluation results in [LEADERBOARD](LEADERBOARD.md).
