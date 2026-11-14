"""Independent CCT worker; the three existing baseline workers are unchanged."""
import sys
import os
import json
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "scripts"), str(ROOT.parent)]

if __name__ == "__main__":
    if "--resume" in sys.argv:
        raise SystemExit("Exact context-sampler resume is not implemented; preserve this run and register a new one")
    import train
    from sampling import BalancedSampler
    from cct_data import CCTDataset
    from cct_policy import CCTPolicy
    train.KimodoWindowDataset = CCTDataset
    train.EpisodeUniformSampler = BalancedSampler
    train.KimodoPolicy = CCTPolicy
    # Diagnostic receipts come from each actual DDP rank; formal training is unchanged.
    audit = os.environ.get("CCT_DIAGNOSTIC_DIR")
    if audit:
        import torch
        original_build = train.build_model
        captured = {}

        def build(config):
            model = original_build(config)
            captured.update(model=model, initial=model.cct_encoder.queries.detach().cpu().clone())
            return model

        train.build_model = build
    train.main()
    if audit:
        model = captured["model"]
        query = model.cct_encoder.queries.detach().cpu()
        params = dict(model.cct_encoder.named_parameters())
        valid_gradients = all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values())
        changed = bool((query != captured["initial"]).any())
        rank = int(os.environ["RANK"])
        report = dict(rank=rank, world_size=int(os.environ["WORLD_SIZE"]),
                      local_rank=int(os.environ["LOCAL_RANK"]), gpu=torch.cuda.get_device_name(),
                      all_encoder_gradients_finite=bool(valid_gradients), encoder_updated=changed,
                      query_sha256=hashlib.sha256(query.numpy().tobytes()).hexdigest(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      passed=bool(valid_gradients and changed))
        (Path(audit) / f"rank-{rank}.json").write_text(json.dumps(report, indent=2)+"\n")
        if not report["passed"]:
            raise SystemExit("CCT DDP gradient/update audit failed")
