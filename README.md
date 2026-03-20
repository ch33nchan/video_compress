# comma2k25 compression challenge 🤏

<p align="center">
<img height="300" alt="image" src="https://github.com/user-attachments/assets/f72fae51-96bd-47c6-b4ff-6e17e79220cb" />
</p>

 `./test_videos/b0c9d2329ad1606b|2018-07-27--06-03-57/10/video.hevc` is a 1 minute long driving video of size 37.5 MB. Make it as small as possible while preserving semantic content (evaluated by a segmentation model) and temporal dynamics (evaluated by an egomotion relative pose model).

- distortion:
  - SegNet distortion: average class disagreements between the predictions of a SegNet evaluated on original vs. reconstructed frames
  - PoseNet distortion: MSE of the outputs of a PoseNet (x,y,z velocities and roll,pitch,yaw rates) evaluated on original vs. reconstructed seq_len frames
- rate
  - the size of the compressed archive devided by the size of the original archive
- score: a weighted average of the different components of the distortion and the rate

```
score = 100 * segnet_distortion + sqrt(10 * posenet_distortion) + 25 * rate
```

<p align="center">
<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/e4421c23-8fbe-4293-b8de-9a77e6d568ab"/>
</p>

## Quickstart
```
# clone the repo
git clone https://github.com/commaai/comma2k25_compression_challenge.git
cd comma2k25_compression_challenge

# git lfs
git lfs install
git lfs pull

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# pick one: cu126/ cu128 / cu130 / cpu
uv sync --group cu128

# activate
source .venv/bin/activate

# test dataloaders
python frame_utils.py

# test models
python modules.py

# create a submission dir and copy the fast baseline scripts
mkdir -p submissions/my_submission
cp submissions/baseline_fast/compress.sh submissions/my_submission/
cp submissions/baseline_fast/inflate.sh submissions/my_submission/

# naively recompress (creates submissions/my_submission/archive.zip)
bash submissions/my_submission/compress.sh

# evaluate the submission
bash evaluate.sh --submission-dir ./submissions/my_submission --device cuda
```

If everything worked as expected, this should producce a `report.txt` file with this content:

```
=== Evaluation config ===
  batch_size: 16
  device: cuda
  num_threads: 2
  prefetch_queue_depth: 4
  report: submissions/my_submission/report.txt
  seed: 1234
  submission_dir: submissions/my_submission
  uncompressed_dir: /home/batman/comma2k25_compression_challenge/test_videos
  video_names_file: /home/batman/comma2k25_compression_challenge/public_test_video_names.txt
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.06424776
  Average SegNet Distortion: 0.00381220
  Submission file size: 11580703 bytes
  Original uncompressed size: 37533786 bytes
  Compression Rate: 0.30854076
  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 8.89628579
```

## submission format and rules

A submission is a directory containing two files:

- **`archive.zip`** — your compressed data. Its size is used to compute the rate term of the score. It will be unzipped into `archive/` by the evaluation script.
- **`inflate.sh`** — a bash script that converts the extracted `archive/` contents into raw video frames.

`inflate.sh` is called with three positional arguments:

```bash
bash inflate.sh <archive_dir> <output_dir> <video_names_file>
```

- `archive_dir`: path where `archive.zip` has been extracted
- `output_dir`: path where inflated frames must be written
- `video_names_file`: text file listing video paths, one per line

For each line in `video_names_file`, `inflate.sh` must produce a raw video file at `<output_dir>/<segment_id>/video.raw`. A `.raw` file is a flat binary dump of uint8 RGB frames with shape `(N, H, W, 3)` where N is the number of frames, H and W match the original video dimensions, no header.

Open a Pull Request with your submission and follow the template instructions to be evaluated.

See [submissions/baseline/](submissions/baseline/) or [submissions/baseline_fast/](submissions/baseline_fast/) for working examples.

### evaluation

```bash
bash evaluate.sh --submission-dir ./submissions/baseline --device cpu
```

Use `--device cuda` for faster local evaluation. Official evaluation in CI uses `--device cpu`.

### rules

- External libraries and tools can be used and won't count towards compressed size, unless they use large artifacts (neural networks, meshes, point clouds, etc.).
- You can use anything for compression, including the models and the original uncompressed videos.
- You may include your compression script in the submission, but it's not required.
- `inflate.sh` should not consume anything outside of the submission directory and the extracted archive.

## going further

You can use [test_videos.zip](https://huggingface.co/datasets/commaai/comma2k19/resolve/main/compression_challenge/test_videos.zip), which is a 2.4 GB archive of 64 driving videos from the comma2k19 dataset, to test your compression strategy on more samples.

The evaluation script and the dataloader are designed to be scalable and can handle different batch sizes, sequence lengths, and video resolutions. You can modify them to fit your needs.
