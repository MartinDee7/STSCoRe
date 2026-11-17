from dataclasses import asdict
from unittest.mock import patch

import pytest
import torch

from stscore.model import FusionHead
from stscore.objectives import RewardConfig, forecast_reward, parse_forecasts, policy_loss
from stscore.train import training_loss


def test_full_output_uses_exactly_two_generations_and_fusion(engine, batch):
    engine.model.set_stage("head")
    with patch.object(engine.model, "generate", wraps=engine.model.generate) as calls:
        result = engine.forecast(batch)
    assert calls.call_count == 2
    assert result.prediction.shape == (2, 12)
    assert result.second.states.shape[:2] == result.second.tokens.shape
    assert result.second.state_mask.any(-1).all()
    result.prediction.square().mean().backward()
    assert engine.model.head.regression.weight.grad.abs().sum() > 0
    assert engine.model.align[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in engine.model.llm.parameters())


def test_decoder_states_are_generated_token_states(tiny_model, batch):
    ids = torch.tensor([[5, 7, 9], [0, 6, 8]])
    mask = (ids != 0).long()
    aligned = tiny_model.encode(batch)
    result = tiny_model.generate(aligned, ids, mask, pad_id=0, eos_ids=[1], retain_states=True)
    full = torch.cat([ids, result.tokens], 1)
    full_mask = torch.cat([mask, result.mask], 1)
    inputs, attention, position = tiny_model._inputs(aligned, full, full_mask)
    with torch.no_grad():
        out = tiny_model.llm(inputs_embeds=inputs, attention_mask=attention,
                             position_ids=position, output_hidden_states=True, use_cache=False)
    # Compare to independently teacher-forced generated-token positions, not logits positions.
    expected = out.hidden_states[-1][:, aligned.shape[1]+ids.shape[1]:].float()
    torch.testing.assert_close(result.states[result.state_mask], expected[result.state_mask], atol=2e-5, rtol=2e-5)


def test_generated_scoring_excludes_prefix_prompt_and_padding(tiny_model, batch):
    ids = torch.tensor([[5, 7, 9], [0, 6, 8]])
    mask = (ids != 0).long()
    tokens = torch.tensor([[11, 12, 1], [13, 1, 0]])
    valid = torch.tensor([[True, True, True], [True, True, False]])
    aligned = tiny_model.encode(batch)
    actual = tiny_model.score(aligned, ids, mask, tokens, valid)
    inputs, attention, position = tiny_model._inputs(aligned, torch.cat([ids, tokens], 1), torch.cat([mask, valid], 1))
    out = tiny_model.llm(inputs_embeds=inputs, attention_mask=attention, position_ids=position, use_cache=False)
    expected = []
    for row in range(2):
        terms = []
        for j in range(3):
            if valid[row, j]:
                terms.append(out.logits[row, 12+3+j-1].log_softmax(-1)[tokens[row, j]])
        expected.append(torch.stack(terms).mean())
    torch.testing.assert_close(actual, torch.stack(expected))


def test_mean_policy_loss_has_no_total_advantage_clip():
    p = torch.tensor([-2.0], requires_grad=True)
    ref = torch.tensor([-3.0])
    cfg = RewardConfig(alpha=0.5, beta=0.3)
    loss, advantage = policy_loss(p, ref, torch.tensor([10.0]), torch.tensor([20.0]), cfg, 2)
    torch.testing.assert_close(advantage, torch.tensor([22.2]))
    loss.backward()
    torch.testing.assert_close(p.grad, -advantage)
    assert not advantage.requires_grad


def test_reward_and_strict_invalid_handling():
    pred, valid = parse_forecasts(["1, 2, 3", "values: 1, 2, 3", "1, 2"], 3)
    assert valid.tolist() == [True, False, False]
    cfg = RewardConfig()
    target = torch.tensor([[1., 2., 3.]]).expand(3, -1)
    reward = forecast_reward(pred, target, valid, cfg)
    torch.testing.assert_close(reward, torch.tensor([0.35, -10., -10.]))


def test_fusion_3584_width_padding_invariance_and_branch_dependence():
    torch.manual_seed(4)
    head = FusionHead(3584, 12)
    a, r, c = [torch.randn(2, n, 3584) for n in (12, 5, 7)]
    rm = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    cm = torch.ones(2, 7, dtype=torch.bool)
    last = torch.randn(2, 1)
    before = head(a, r, rm, c, cm, last)
    r2 = r.clone()
    r2[~rm] = 1e8
    torch.testing.assert_close(before, head(a, r2, rm, c, cm, last))
    assert not torch.allclose(before, head(a, -r, rm, c, cm, last))
    assert set(dict(head.named_parameters())) == {"regression.weight", "regression.bias", "residual.weight", "residual.bias"}


@pytest.mark.parametrize("stage", ["sft", "rl1", "rl2", "head", "adapt"])
def test_stage_loss_updates_only_declared_parameters(engine, batch, stage):
    model = engine.model
    model.set_stage(stage)
    model.eval()
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.001)
    loss = training_loss(engine, batch, stage, RewardConfig())
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    changed = [k for k, p in model.named_parameters() if not torch.equal(p, before[k])]
    assert changed
    assert all(dict(model.named_parameters())[k].requires_grad for k in changed)
    if stage == "adapt":
        assert all(k.startswith(("align.", "head.")) for k in changed)
    if stage in {"sft", "rl1", "rl2"}:
        assert any("lora_" in k for k in changed)


def test_checkpoint_state_never_contains_base_llm_weights(tiny_model, tmp_path):
    state = tiny_model.revision_state()
    assert any("lora_" in k for k in state)
    assert not any(k.startswith("llm.") and "lora_" not in k for k in state)
    path = tmp_path / "model.pt"
    torch.save(state, path)
    tiny_model.load_revision_state(torch.load(path, weights_only=True))
    with pytest.raises(ValueError, match="schema mismatch"):
        tiny_model.load_revision_state({})
