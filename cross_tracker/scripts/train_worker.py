"""Use the shared DDP trainer with the repaired, source-balanced data loader."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "scripts"), str(ROOT.parent)]

if __name__ == "__main__":
    import train
    from sampling import BalancedSampler, TrainingDataset

    train.KimodoWindowDataset = TrainingDataset
    train.EpisodeUniformSampler = BalancedSampler
    train.main()
