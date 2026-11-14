"""Independent CCT worker; the three existing baseline workers are unchanged."""
import sys
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
    train.main()
