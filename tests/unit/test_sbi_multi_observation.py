"""Time-aware SBI context encoding."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sisyphus.regimen.types import DosingRegimen
from sisyphus.sbi.multi_observation import pack_observations


def test_pack_observations_pads_and_masks_three_slots():
    regimen = DosingRegimen.oral_repeated(30.0, 12.0, 2)
    x = pack_observations([(13.0, 0.02), (16.0, 0.01)], regimen, np.ones(12))
    assert x.shape == (24,)
    assert x[0] == pytest.approx(13.0 / 24.0)
    assert x[1] == pytest.approx(np.log10(0.02))
    assert x[2] == 1
    assert x[5] == 1
    assert np.array_equal(x[6:9], np.zeros(3))
    assert x[9:12].tolist() == pytest.approx([np.log10(30.0), 0.5, 2 / 3])


def test_pack_observations_rejects_invalid_observations():
    regimen = DosingRegimen.single_oral(30.0)
    with pytest.raises(ValueError, match="finite and positive"):
        pack_observations([(1.0, 0.0)], regimen, np.ones(12))
    with pytest.raises(ValueError, match="1–3"):
        pack_observations([], regimen, np.ones(12))
    with pytest.raises(ValueError, match="training grid"):
        pack_observations([(1.5, 0.02)], regimen, np.ones(12))
    with pytest.raises(ValueError, match="equal bolus"):
        pack_observations([(1.0, 0.02)], DosingRegimen.single_iv(30.0), np.ones(12))


def test_sbc_bounds_posterior_sampling_time():
    pytest.importorskip("torch")
    from scripts.sbi_run_sbc_multi_drug import MAX_SBC_SAMPLE_TIME_S, _coverage_and_ranks

    class StalledPosterior:
        def sample(self, shape, *, x, show_progress_bars, max_sampling_time):
            assert shape == (500,)
            assert max_sampling_time == MAX_SBC_SAMPLE_TIME_S
            raise RuntimeError("sampling timed out")

    with pytest.raises(RuntimeError, match="sampling timed out"):
        _coverage_and_ranks(
            np.zeros((1, 3)), StalledPosterior(), np.zeros((1, 13)), (90,)
        )


def test_train_script_scales_only_multi_observation_drug_features(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from scripts import sbi_train_multi_drug

    contexts = np.tile(np.arange(24, dtype=np.float32), (3, 1))
    contexts[:, -12:] += np.arange(3, dtype=np.float32)[:, None]
    data = tmp_path / "training.npz"
    out = tmp_path / "posterior.pt"
    np.savez(
        data, theta=np.zeros((3, 3), dtype=np.float32), multi_obs_x=contexts,
        prior_low=np.full(3, -1), prior_high=np.full(3, 1),
    )
    seen = {}

    def train_npe(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(training_losses=[], validation_losses=[])

    monkeypatch.setattr(sbi_train_multi_drug, "train_npe", train_npe)
    monkeypatch.setattr(sbi_train_multi_drug, "save_result", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(data), "--out", str(out)])
    sbi_train_multi_drug.main()
    aux = torch.load(out.with_suffix(".aux.pt"), weights_only=False)
    assert aux["multi_observation"] is True
    assert seen["x"].shape == (3, 24)
    np.testing.assert_array_equal(seen["x"][:, :12], contexts[:, :12])
    np.testing.assert_allclose(seen["x"][:, -12:].mean(axis=0), 0, atol=1e-6)
