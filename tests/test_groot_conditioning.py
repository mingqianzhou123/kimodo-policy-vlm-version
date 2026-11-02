"""Check the new architecture's initialization, padding and gradient boundaries."""

from types import SimpleNamespace
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from kimodo_vla.model.modules.backbone import TransformerEncoderBlock
from kimodo_vla.model.modules.groot_conditioning import (
    CrossAttentionResidual, GrootInjection, InjectionStack, SharedVLMAdapter,
    VLCondition, load_checkpoint_parts,
)
from kimodo_vla.model.modules.hand_control import HandTokenDiffusionDenoiser


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_real_width_zero_residual_and_gradient_activation():
    block = CrossAttentionResidual()
    query = torch.randn(1, 50, 1024, requires_grad=True)
    features = torch.randn(1, 13, 2048, requires_grad=True)
    condition = VLCondition(features, torch.ones(1, 13, dtype=torch.bool))
    update = block(query, condition)
    assert update.shape == (1, 50, 1024)
    assert torch.count_nonzero(update) == 0
    target = torch.randn_like(update)
    optimizer = torch.optim.SGD(block.parameters(), lr=0.1)
    (query + update - target).square().mean().backward()
    assert block.output_projection.weight.grad.abs().sum() > 0
    assert features.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    features.grad = None
    block(query, condition).square().mean().backward()
    assert features.grad.abs().sum() > 0


def test_injection_has_eight_independent_modules_and_one_condition():
    model = SimpleNamespace(latent_dim=1024, seqTransEncoder=SimpleNamespace(layers=range(16)))
    injection = GrootInjection(SimpleNamespace(root_model=model, body_model=model))
    condition = VLCondition(torch.randn(1, 5, 2048), torch.ones(1, 5, dtype=torch.bool))
    root, body = injection(None, condition, 150, 100)
    assert tuple(root) == tuple(body) == (3, 7, 11, 15)
    assert all(v is condition for v in [*root.values(), *body.values()])
    modules = [*injection.root_hint_fusion.blocks.values(), *injection.body_hint_fusion.blocks.values()]
    assert len({p.data_ptr() for m in modules for p in m.parameters()}) == sum(len(list(m.parameters())) for m in modules)
    with pytest.raises(ValueError, match="50 future"):
        injection(None, condition, 150, 99)


def test_backbone_injects_after_layers_only_into_future_query():
    backbone = TransformerEncoderBlock(
        input_dim=4, output_dim=4, skeleton=SimpleNamespace(nbjoints=1),
        llm_shape=[1, 8], use_text_mask=True, latent_dim=8, ff_size=16,
        num_layers=16, num_heads=2, activation="gelu", dropout=0.0, pe_dropout=0.0,
    ).eval()
    condition = VLCondition(torch.randn(1, 7, 16), torch.ones(1, 7, dtype=torch.bool))
    stack = InjectionStack(motion_dim=8, condition_dim=16, num_heads=2)
    kwargs = dict(x=torch.randn(1, 150, 4), x_pad_mask=torch.ones(1, 150, dtype=torch.bool),
                  text_feat=torch.randn(1, 1, 8), text_feat_pad_mask=torch.ones(1, 1, dtype=torch.bool),
                  timesteps=torch.tensor([4]))
    base = backbone(**kwargs)
    calls, layer_finished = [], []
    handles = [layer.register_forward_hook(lambda m, args, out, i=i: layer_finished.append(i))
               for i, layer in enumerate(backbone.seqTransEncoder.layers)]

    def fuse(layer, features, query):
        assert layer_finished[-1] == layer
        calls.append((layer, query.shape))
        return stack(layer, features, query)

    result = backbone(**kwargs, control_visual_tokens={i: condition for i in (3, 7, 11, 15)},
                      hint_fuser=fuse, future_start=100)
    torch.testing.assert_close(base, result, rtol=1e-6, atol=1e-6)
    assert calls == [(i, torch.Size([1, 50, 8])) for i in (3, 7, 11, 15)]
    for handle in handles:
        handle.remove()


def test_cross_attention_ignores_padding_and_reads_action_query():
    block = CrossAttentionResidual(8, 16, 2).eval()
    nn.init.eye_(block.output_projection.weight)
    features = torch.randn(1, 3, 16)
    query = torch.randn(1, 50, 8)
    clean = VLCondition(features, torch.ones(1, 3, dtype=torch.bool))
    padded = VLCondition(torch.cat((features, torch.randn(1, 8, 16) * 100), 1),
                         torch.tensor([[True] * 3 + [False] * 8]))
    torch.testing.assert_close(block(query, clean), block(query, padded))
    assert not torch.allclose(block(query, clean), block(torch.randn_like(query), clean))


def test_shared_adapter_never_sees_padding_and_receives_gradients():
    encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
        16, 2, 32, batch_first=True, dropout=0.0), 4, enable_nested_tensor=False)
    adapter = SharedVLMAdapter(nn.LayerNorm(16), encoder).eval()
    features = torch.randn(2, 9, 16, requires_grad=True)
    mask = torch.tensor([[False] * 4 + [True] * 5, [True] * 9])
    result = adapter(features, mask)
    expected = adapter(features[:1, 4:], torch.ones(1, 5, dtype=torch.bool))
    torch.testing.assert_close(result.tokens[:1, :5], expected.tokens)
    result.tokens.square().mean().backward()
    assert torch.count_nonzero(features.grad[0, :4]) == 0
    assert adapter.vlln.weight.grad.abs().sum() > 0


def test_hand_head_accepts_variable_2048d_tokens_and_detaches_conditions():
    head = HandTokenDiffusionDenoiser(input_dim=8, hidden_dim=16, image_token_count=None,
                                     visual_dim=32, num_layers=2, num_heads=2, ffn_dim=32).eval()
    features = torch.randn(1, 7, 32, requires_grad=True)
    body = torch.randn(1, 50, 8, requires_grad=True)
    kwargs = dict(body_future_tokens=body, noisy_future_hand=torch.randn(1, 50, 2),
                  current_hand_state=torch.zeros(1, 2), timesteps=torch.tensor([8]))
    clean = head(visual_tokens=features, **kwargs)
    padded = head(visual_tokens=torch.cat((features, torch.randn(1, 4, 32) * 100), 1),
                  visual_mask=torch.tensor([[True] * 7 + [False] * 4]), **kwargs)
    torch.testing.assert_close(clean, padded, rtol=1e-5, atol=1e-6)
    clean.square().mean().backward()
    assert features.grad is None and body.grad is None
    assert head.vision_projection[1].weight.grad.abs().sum() > 0


def test_checkpoint_prefix_load_is_strict_and_skips_action_head(tmp_path):
    original = nn.Linear(3, 4)
    save_file({**{"action_head.vlln." + k: v for k, v in original.state_dict().items()},
               "action_head.diffusion.unused": torch.randn(5)}, tmp_path / "model.safetensors")
    target = nn.Linear(3, 4)
    report = load_checkpoint_parts(str(tmp_path), {"action_head.vlln.": target})
    torch.testing.assert_close(target.weight, original.weight)
    assert report["action_head.vlln."]["tensors"] == 2
    with pytest.raises(RuntimeError, match="No checkpoint tensors"):
        load_checkpoint_parts(str(tmp_path), {"missing.": target})


def test_stage_transition_keeps_adam_moments_lr_and_frozen_adapter_checkpoint():
    from omegaconf import OmegaConf
    from kimodo_vla.model.kimodo_policy import KimodoPolicy
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train import make_optimizer, training_stage

    model = nn.Module()
    model.config = SimpleNamespace(visual_conditioning="groot_n1d7")
    model.controlnet = nn.Linear(3, 3)
    model.hand_head = nn.Linear(3, 2)
    model.image_encoder = nn.Module()
    model.image_encoder.adapter = nn.Linear(3, 3).requires_grad_(False)
    cfg = OmegaConf.create({"model": {"visual_conditioning": "groot_n1d7"}, "main": {"max_steps": 20},
        "training": {"stages": {"adapter_start": 10, "vision_start": None},
                     "scheduler": {"warmup_steps": 2, "min_lr_ratio": .1},
                     "optimizer": {"control_backbone_lr": 3e-5, "control_adapter_lr": 1e-4,
                                   "hand_lr": 1e-4, "weight_decay": .01, "betas": [.9, .999], "eps": 1e-8}}})
    assert training_stage(cfg, 9) == "injection" and training_stage(cfg, 10) == "adapter"
    groups, optimizer, scheduler = make_optimizer(model, cfg, torch.device("cpu"))
    for _ in range(10):
        optimizer.zero_grad()
        model.controlnet(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        scheduler.step()
    saved_moment = optimizer.state[model.controlnet.weight]["exp_avg"].clone()
    lr = optimizer.param_groups[0]["lr"]
    snapshot = KimodoPolicy.checkpoint_parameters(model)
    assert "image_encoder.adapter.weight" in snapshot
    model.image_encoder.adapter.requires_grad_(True)
    groups, changed, changed_schedule = make_optimizer(model, cfg, torch.device("cpu"), 10, optimizer)
    torch.testing.assert_close(changed.state[model.controlnet.weight]["exp_avg"], saved_moment)
    assert changed.param_groups[0]["lr"] == pytest.approx(lr)
    assert any(g["name"] == "vl_adapter" for g in groups)


def test_arena_resolves_simulator_task_id_to_the_training_instruction(tmp_path):
    from kimodo_vla.benchmarks.arena.policy import cached_instruction_aliases
    from kimodo_vla.benchmarks.arena.adapter import ArenaAdapter
    import numpy as np

    instruction = "Put the hammer into the basket."
    torch.save({"task_id": "HumanoidArena::TaskDoubleDesk", "task_name": "HOI_double_desk",
                "instruction": instruction, "embedding": torch.zeros(1, 4096)}, tmp_path / "task.pt")
    aliases = cached_instruction_aliases(tmp_path)
    adapter = ArenaAdapter(aliases)
    state = np.zeros((1, 64), np.float32)
    state[0, :6] = [1, 0, 0, 1, 0, 0]
    observations = adapter.to_observations({"task": "TaskDoubleDesk", "observation": {
        "state_history": state, "images": {"front": np.zeros((8, 8, 3), np.uint8)}}})
    assert observations[-1].instruction == instruction
    assert aliases["HOI_double_desk"] == aliases["HumanoidArena::TaskDoubleDesk"] == instruction


def test_distributed_launcher_and_existing_keepalive_monitor(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_groot_experiment as launcher

    command = launcher.training_command(Path("recipe.yaml"), Path("new-run"), [4, 5, 6, 7])
    assert "torch.distributed.run" in command and "--nproc_per_node=4" in command
    assert "--init-checkpoint" not in command
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="后台自启动：已启用\nxiancun RUNNING pid 23991")
    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.keepalive_receipt(Path("xiancun"))["supervised"]
    assert [command[-1] for command in calls] == ["ensure", "status"]
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="STOPPED"))
    with pytest.raises(RuntimeError, match="not enabled and monitored"):
        launcher.keepalive_receipt(Path("xiancun"))
