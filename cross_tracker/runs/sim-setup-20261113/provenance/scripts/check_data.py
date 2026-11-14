"""Audit frozen numeric segments and independently check prepared video/window boundaries."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent)]
import av
import numpy as np
import torch
from kimodo_vla.benchmarks.arena.demos import resize_image
from kimodo_vla.data.episodes import KimodoWindowDataset
from repaired_data import checksum, read_segment


def check_prepared(folder, videos):
    dataset = KimodoWindowDataset([folder])
    results = []
    image_checks = defaultdict(lambda: defaultdict(list))
    start = 0
    for record, frames_path in zip(dataset.episodes, dataset.frame_paths):
        stem = frames_path.with_suffix("").with_suffix("")
        receipt = json.loads(stem.with_suffix(".json").read_text())
        for ext, digest in receipt["files"].items():
            if checksum(stem.with_suffix(ext)) != digest:
                raise ValueError(f"Prepared file changed: {stem}{ext}")
        images = np.load(frames_path, mmap_mode="r")
        length = len(images)
        mapping = record["source_mapping"]
        cuts = sorted({0, min(100, length-50), length-50})
        for cut in cuts:
            sample = dataset[start+cut]
            assert sample["gt_motion"].shape == (150, 417)
            assert sample["gt_mask"].sum() == min(cut, 100)+50
            assert not sample["condition_motion_mask"][100:].any()
            assert not sample["condition_motion_mask"][:max(0, 100-cut)].any()
            assert torch.equal(sample["egoview"], torch.from_numpy(images[cut, 0].copy()).permute(2, 0, 1))
        # Decode the original container sequentially, independently of ArenaDemo's seek logic.
        last_control_frame = mapping["source_frame_stop"]-mapping["source_frame_start"]-1
        wanted = {int(round(mapping["video_offset_s"]*50)) + min(round(k*50/30), last_control_frame): k
                  for k in (0, length//2, length-1)}
        for index, k in wanted.items():
            image_checks[mapping["video"]][index].append((len(results), frames_path, k))
        results.append({"episode": record["episode_id"], "model_frames": length,
                        "windows": length-49, "image_frames_checked": [],
                        "window_cuts_checked": cuts, "source_mapping": mapping})
        start += length-49
    # A shard contains many episodes: decode it once while checking every episode.
    def check_video(item):
        video, checks = item
        matched = set()
        with av.open(str(videos / video)) as container:
            for frame in container.decode(video=0):
                index = int(round(float(frame.pts*frame.time_base)*50))
                if index in checks:
                    expected = resize_image(frame.to_ndarray(format="rgb24")[None])[0]
                    for result_index, frames_path, k in checks[index]:
                        actual = np.load(frames_path, mmap_mode="r")[k, 0]
                        np.testing.assert_array_equal(actual, expected)
                        results[result_index]["image_frames_checked"].append(k)
                    matched.add(index)
                if index >= max(checks):
                    break
        assert matched == set(checks), f"Missing source image timestamps: {video}"
        print(f"IMAGES_VERIFIED {video}: {len(checks)} timestamps", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(check_video, image_checks.items()))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--videos", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest_path = ROOT / "data/manifests/balanced-v1.json"
    manifest = json.loads(manifest_path.read_text())
    assets = json.loads((ROOT / "data/manifests/assets-v1.json").read_text())
    roots = json.loads(args.paths.read_text())
    counts = {}
    for d in manifest["datasets"]:
        lengths = [len(read_segment(d, e, roots, assets)[0]) for e in d["episodes"]]
        counts[d["dataset"]] = {"episodes": len(lengths), "min_control_frames": min(lengths)}
    report = {"manifest_sha256": checksum(manifest_path), "numeric_segments": counts,
              "prepared_checks": check_prepared(args.prepared, args.videos) if args.prepared else [],
              "training_started": False, "demonstration_success": "unverified"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"segments_checked": sum(c["episodes"] for c in counts.values()),
                      "prepared_checked": len(report["prepared_checks"]), "report": str(args.report)}))
