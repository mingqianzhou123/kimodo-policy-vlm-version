"""CCT leakage, conditioning gradients, and checkpoint round-trip checks."""
import io
from pathlib import Path
import sys
import unittest
import torch
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[2])]
from cct_context import InteractionEncoder
from cct_data import select_context, combine_context


class CCTTests(unittest.TestCase):
    def test_disjoint_same_source_and_reproducible(self):
        episodes = [{"episode_id": str(i), "task": "box", "tracker": source, "split": split}
                    for i, (source, split) in enumerate([("sonic", "train")] * 4 +
                         [("twist2", "train")] * 4 + [("sonic", "validation")])]
        for target in range(8):
            a = select_context(episodes, target, torch.Generator().manual_seed(42))
            b = select_context(episodes, target, torch.Generator().manual_seed(42))
            self.assertEqual(a, b)
            self.assertEqual(len(set(a)), 3)
            self.assertNotIn(target, a)
            self.assertTrue(all(episodes[i]["tracker"] == episodes[target]["tracker"] for i in a))
            self.assertNotIn(8, a)
        with self.assertRaises(ValueError):
            select_context(episodes[:3], 0, torch.Generator())

    def test_128_aligned_samples_cover_all_trajectories(self):
        records = [{"transitions": torch.arange(43.)[:, None].expand(-1, 110),
                    "visual": torch.full((43, 2, 4), float(i))} for i in range(3)]
        out = combine_context(records, torch.ones(1, 8), torch.zeros(110), torch.ones(110))
        self.assertEqual(out["transitions"].shape, (128, 110))
        self.assertEqual(torch.bincount(out["trajectory_ids"]).tolist(), [43, 43, 42])
        for i in range(3):
            mask = out["trajectory_ids"] == i
            self.assertTrue((out["visual"][mask] == i).all())
            self.assertEqual(out["transitions"][mask, 0][[0, -1]].tolist(), [0., 42.])

    def test_visual_physical_gradients_and_reload(self):
        torch.manual_seed(42)
        args = dict(visual_dim=8, transition_dim=5, output_dim=12, width=16,
                    queries=4, layers=2, heads=4, horizon=6, patches=2)
        model = InteractionEncoder(**args)
        visual, physical = torch.randn(2, 6, 2, 8), torch.randn(2, 6, 5)
        slots = torch.tensor([[0, 0, 1, 1, 2, 2]] * 2)
        output = model(visual, physical, slots)
        output.square().mean().backward()
        for module in (model.visual, model.physical, model.cross_attention):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters()), 0.)
        self.assertFalse(torch.allclose(output, model(visual + 1, physical, slots)))
        self.assertFalse(torch.allclose(output, model(visual, physical + 1, slots)))
        data = io.BytesIO(); torch.save(model.state_dict(), data); data.seek(0)
        restored = InteractionEncoder(**args); restored.load_state_dict(torch.load(data, weights_only=True))
        torch.testing.assert_close(output, restored(visual, physical, slots), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model(visual[:, :-1], physical[:, :-1], slots[:, :-1])


    def test_metadata_and_task_tokens_survive_prefix(self):
        from cct_policy import CCTPolicy
        policy = CCTPolicy.__new__(CCTPolicy)
        torch.nn.Module.__init__(policy)
        policy.cct_encoder = InteractionEncoder(visual_dim=8, transition_dim=5, output_dim=12,
                                               width=16, queries=4, layers=1, heads=4, horizon=6, patches=2)
        context = {"visual": torch.randn(1, 6, 2, 8), "transitions": torch.randn(1, 6, 5),
                   "trajectory_ids": torch.tensor([[0, 0, 1, 1, 2, 2]]), "metadata": torch.randn(1, 1, 12)}
        prefix = policy.context_prefix(context)
        text = torch.randn(1, 1, 12)
        joined, lengths = policy.prepend(prefix, text, torch.ones(1, dtype=torch.long))
        planner_joined, _ = policy.prepend(prefix, text[0], None)
        self.assertTrue(torch.equal(joined, planner_joined))
        self.assertEqual(joined.shape, (1, 6, 12))
        self.assertEqual(lengths.tolist(), [6])
        self.assertTrue(torch.equal(joined[:, -2:-1], context["metadata"]))
        self.assertTrue(torch.equal(joined[:, -1:], text))
        changed = policy.context_prefix(dict(context, metadata=context["metadata"] + 1))
        self.assertTrue(torch.equal(changed[:, :-1], prefix[:, :-1]))
        self.assertFalse(torch.equal(changed[:, -1], prefix[:, -1]))
        with self.assertRaises(ValueError):
            policy.prepend(prefix, text, torch.zeros(1, dtype=torch.long))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
