"""Turn HumanoidToolBench demonstrations (ToolBook) into Kimodo training episodes.

For every episode of every recording under --data-dir (a local copy of the
dataset's ``raw/`` at revision 8b2cd31e): the history and the operator's commands
as Kimodo features (``kimodo_vla.data.build.training_episode``) and the head
camera, directly resized to 224x224, at every model-rate frame. Output:
``<out>/<condition>/<recording>_<episode>.pt`` and ``.frames.npy`` (see
``kimodo_vla.data.episodes``). Existing episodes are skipped.

    python scripts/prepare_toolbench.py --data-dir <ToolBook>/raw --out <prepared dir> --workers 16

By default only successful episodes are used (``episode_success`` in
``meta/episodes.jsonl``); ``--all`` keeps the rest.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import multiprocessing as mp
import time
import tempfile
from pathlib import Path

_FEATURES = None


def _init_worker() -> None:
    import torch

    global _FEATURES
    torch.set_num_threads(1)  # many workers share the host; the default oversubscribes it
    from kimodo_vla.benchmarks.toolbench.adapter import CONTROL_FPS
    from kimodo_vla.model.features import KimodoFeatures

    _FEATURES = KimodoFeatures(CONTROL_FPS)


def recordings(data_dir: Path) -> list[Path]:
    """Every LeRobot recording below ``data_dir`` (a directory with ``meta/info.json``)."""
    return sorted(p.parent.parent for p in data_dir.rglob("meta/info.json"))


def prepare_one(job) -> str:
    recording, out_dir, episode, name, reuse_prepared, refresh_current = job
    from kimodo_vla.benchmarks.toolbench import demos
    from kimodo_vla.data.build import training_episode, current_observation_fields
    from kimodo_vla.data.episodes import save_episode, validate_episode

    start = time.time()
    recording = Path(recording)
    condition = demos.canonical(recording.parent.name)
    stem = Path(out_dir) / condition / f"{name}_{episode:06d}"
    if stem.with_suffix(".frames.npy").exists() and stem.with_suffix(".pt").exists():
        import torch
        old = torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=True)
        expected = {"history_source": demos.HISTORY_SOURCE, "image_preprocessing": demos.IMAGE_PREPROCESSING,
                    "current_observation_source": demos.CURRENT_OBSERVATION_SOURCE}
        compatible = all(old.get(k) == v for k, v in expected.items())
        if not compatible:
            upgradable = (refresh_current and reuse_prepared and old.get('current_observation_source') == 'measured_at_rgb_cut_v1'
                          and all(old.get(k) == v for k, v in expected.items() if k != 'current_observation_source'))
            if not upgradable:
                raise ValueError(f"{stem}: incompatible prepared cache; use a new --out directory")
        if compatible:
            try:
                validate_episode(old, stem.with_suffix(".frames.npy"))
            except (ValueError, OSError):
                pass  # Recover a truncated image payload from the raw demonstration.
            else:
                return f"{stem.name}: exists, verified"
    if reuse_prepared:
        import torch
        import numpy as np
        source = Path(reuse_prepared) / stem.relative_to(out_dir)
        record = torch.load(source.with_suffix('.pt'), map_location='cpu', weights_only=True)
        if record.get('history_source') != demos.HISTORY_SOURCE or record.get('image_preprocessing') != demos.IMAGE_PREPROCESSING:
            raise ValueError(f'{source}: only measured-state, letterboxed caches may be reused')
        validate_episode(record, source.with_suffix('.frames.npy'))
        demo = demos.ToolBenchDemo(demos.read_recording(recording, episode), demos.instruction(recording.parent.name))
        observations = demo.observations()
        steps = np.minimum(np.round(np.arange(len(record['target_motion'])) * 50 / 30).astype(int), len(observations) - 1)
        record.update(current_observation_fields(observations, _FEATURES, steps))
        record.update(demo.preparation_metadata())
        stem.parent.mkdir(parents=True, exist_ok=True)
        image_path = stem.with_suffix('.frames.npy')
        if not image_path.exists():
            image_path.symlink_to(source.with_suffix('.frames.npy').resolve())
        with tempfile.NamedTemporaryFile(dir=stem.parent, suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            try:
                torch.save(record, f)
                f.close()
                temporary.replace(stem.with_suffix('.pt'))
            finally:
                temporary.unlink(missing_ok=True)
        return f'{condition}/{stem.name}: added exact current observations, reused verified letterbox RGB'
    info = json.loads((recording / "meta/info.json").read_text())
    video = recording / info["video_path"].format(
        episode_chunk=episode // info["chunks_size"], episode_index=episode,
        video_key="observation.images.ego_view")
    demo = demos.ToolBenchDemo(demos.read_recording(recording, episode), demos.instruction(recording.parent.name), video)
    import numpy as np
    for key, value in (("state", demo.state), ("base_pose", demo.base_pose), ("command", demo.command), ("nav", demo.nav), ("height", demo.height)):
        if not np.isfinite(value).all():
            raise ValueError(f"{recording} episode {episode}: nonfinite {key}")
    record, frames = training_episode(demo, _FEATURES, task=condition, episode_id=stem.name)
    save_episode(stem, record, frames)
    return f"{condition}/{stem.name}: {len(frames)} frames in {time.time() - start:.0f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--all", action="store_true", help="also failed episodes")
    parser.add_argument("--limit", type=int, help="at most this many episodes per recording")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--manifest", type=Path, help="pinned episode selection (data/video paths relative to data-dir parent)")
    parser.add_argument("--reuse-prepared", type=Path, help="reuse verified measured/letterbox targets and RGB; add exact current observations from raw states")
    parser.add_argument("--refresh-current-observations", action="store_true", help="upgrade the intermediate pose-only cut cache to the measured-height cut protocol")
    args = parser.parse_args()

    from kimodo_vla.benchmarks.toolbench import demos

    jobs = []
    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    if args.reuse_prepared:
        if args.reuse_prepared.resolve() == args.out.resolve():
            raise ValueError('--reuse-prepared must differ from --out; preserve previous experiments')
        previous = json.loads((args.reuse_prepared / 'preparation.json').read_text())
        expected_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest() if args.manifest else None
        if previous.get('manifest_sha256') != expected_hash:
            raise ValueError('reused prepared data must come from the same raw manifest')
    selected = {str((args.data_dir.parent / e["data"]).resolve()) for e in manifest["episodes"]} if manifest else None
    for recording in recordings(args.data_dir):
        info = json.loads((recording / "meta/info.json").read_text())
        if info["fps"] != 50:
            raise ValueError(f"{recording}: expected 50 Hz, got {info['fps']}")
        name = "_".join(recording.relative_to(args.data_dir).parts).replace("/", "_")
        kept = [e["episode_index"] for e in demos.episodes(recording) if args.all or e.get("episode_success")]
        if selected is not None:
            kept = [i for i in kept if str((recording / info["data_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i)).resolve()) in selected]
        jobs += [(str(recording), str(args.out), episode, name,
                  str(args.reuse_prepared.resolve()) if args.reuse_prepared else None,
                  args.refresh_current_observations) for episode in kept[: args.limit]]
    if selected is not None and args.limit is None and len(jobs) != len(selected):
        raise ValueError(f"manifest selects {len(selected)} episodes but metadata matched {len(jobs)}")
    args.out.mkdir(parents=True, exist_ok=True)
    settings = {
        "manifest": str(args.manifest), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest() if args.manifest else None,
        "episodes": len(jobs), "limit_per_recording": args.limit,
        "history_source": demos.HISTORY_SOURCE,
        "image_preprocessing": demos.IMAGE_PREPROCESSING,
        "history_schedule": "every measured 50 Hz control step, resampled to 30 Hz",
        "observed_hand": True,
        "include_failed": args.all,
        "current_observation_source": demos.CURRENT_OBSERVATION_SOURCE,
        "reuse_prepared": str(args.reuse_prepared.resolve()) if args.reuse_prepared else None,
    }
    settings_path = args.out / "preparation.json"
    if settings_path.exists():
        previous = json.loads(settings_path.read_text())
        previous.setdefault("include_failed", False)
        if args.refresh_current_observations and previous.get('current_observation_source') == 'measured_at_rgb_cut_v1':
            previous['current_observation_source'] = demos.CURRENT_OBSERVATION_SOURCE
        if previous != settings:
            raise ValueError(f"{args.out}: incompatible preparation settings; use a new --out directory")
    settings_path.write_text(json.dumps(settings, indent=2))
    complete = args.out / "preparation_complete.json"
    complete.unlink(missing_ok=True)
    print(f"{len(jobs)} episodes", flush=True)
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        for message in pool.imap_unordered(prepare_one, jobs, chunksize=1):
            print(message, flush=True)
    temporary = complete.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"settings_sha256": hashlib.sha256(settings_path.read_bytes()).hexdigest(),
                                     "episodes": len(jobs), "passed": True}, indent=2))
    temporary.replace(complete)


if __name__ == "__main__":
    main()
