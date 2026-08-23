"""Contract tests for the capstone track scaffold (`tracks/adapter.py`).

These cover the four invariants a student can neither see nor debug when they
break, because each one fails *silently* and produces a plausible number:

1. **train -> infer config parity** — a model must predict from the pipeline it
   was fit under, not from the adapter's defaults;
2. **capability declaration** — an advertised cfg option must actually do
   something on the track that offers it, and be refused where it would not;
3. **SMOTE resamples in scaled space** — "nearest minority neighbour" must be a
   physiological statement, not a contest between measurement units;
4. **sklearn meta-estimator plumbing** — nested `set_params` must reach the
   wrapped pipeline, or every `GridSearchCV` candidate is the same model.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "tracks"), os.path.join(_ROOT, "src"),
           os.path.join(_ROOT, "src", "bsp")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from adapter import (SMOTEd, ThresholdMoved, OverSampled, UnsupportedCfgKey,   # noqa: E402
                     BASE_CFG_KEYS, default_baseline)

ALL_TRACKS = ("sleep_edf:SleepEDFTrack", "ecg_cinc2017:ECGCinC2017Track",
              "har:HARTrack", "ctg_ctu_uhb:CTGTrack",
              "emg_ninapro:EMGNinaproTrack", "bci_eegmmidb:BCIEEGMMIDBTrack")


def _track(spec):
    mod, cls = spec.split(":")
    return getattr(__import__(mod), cls)


# ------------------------------------------------------------------ item 1
def test_infer_reuses_the_training_cfg_when_cfg_is_omitted():
    """The regression both review passes flagged: train under a non-default cfg,
    then infer WITHOUT repeating it. The features must come from the training
    config, not from the adapter defaults."""
    T = _track("bci_eegmmidb:BCIEEGMMIDBTrack")
    track = T()
    recs = track.smoke()
    cfg = {"spectral_method": "ar", "ar_order": 8}
    model = track.train_baseline(recs[:4], cfg=cfg)

    # the config actually travelled with the model
    assert model.cfg["spectral_method"] == "ar"
    assert model.cfg["ar_order"] == 8

    omitted = track.infer(model, recs[4])
    repeated = track.infer(model, recs[4], cfg=cfg)
    np.testing.assert_array_equal(omitted, repeated)

    # ...and it is genuinely NOT the default pipeline. This is the "before"
    # behaviour, reconstructed: features built with the adapter's default cfg.
    X_default, _, _ = track._features_for(recs[4], None)
    X_trained, _, _ = track._features_for(recs[4], model.cfg)
    assert not np.allclose(X_default, X_trained), \
        "test is vacuous unless the two configs really produce different features"
    default_preds = np.asarray(model.predict(X_default))
    assert not np.array_equal(default_preds, omitted), \
        "infer() is still building features from the adapter defaults"


def test_train_serialize_infer_parity():
    """The config must survive pickling with the model — a submission is usually
    written by a different process than the one that trained."""
    import pickle
    T = _track("sleep_edf:SleepEDFTrack")
    track = T()
    recs = track.smoke(n_subjects=3, n_epochs=30)
    cfg = {"spectral_method": "multitaper", "select": "anova", "select_k": 5}
    model = track.train_baseline(recs[:2], cfg=cfg)
    revived = pickle.loads(pickle.dumps(model))
    assert revived.cfg["spectral_method"] == "multitaper"
    assert revived.cfg["select_k"] == 5
    np.testing.assert_array_equal(track.infer(model, recs[2]),
                                  track.infer(revived, recs[2]))


def test_infer_on_a_bare_sklearn_model_still_works():
    """No `.cfg` to reuse: fall back to the adapter's own config, don't crash."""
    from sklearn.ensemble import RandomForestClassifier
    T = _track("har:HARTrack")
    track = T(cfg={"gravity": "mean"})
    recs = track.smoke()
    X, y, _ = track.build_dataset(recs[:4])
    clf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    preds = track.infer(clf, recs[4])
    assert len(preds) == len(recs[4].labels)


def test_infer_rejects_a_conflicting_cfg():
    T = _track("bci_eegmmidb:BCIEEGMMIDBTrack")
    track = T()
    recs = track.smoke()
    model = track.train_baseline(recs[:4], cfg={"spectral_method": "ar"})
    with pytest.raises(ValueError, match="TRAINED under"):
        track.infer(model, recs[4], cfg={"spectral_method": "welch"})
    # an explicit acknowledgement is allowed, and is recorded as a harness note
    track.infer(model, recs[4], cfg={"spectral_method": "welch"},
                allow_cfg_mismatch=True)
    assert any("allow_cfg_mismatch" in n for n in track.notes)


def test_infer_ignores_stage_4_5_keys_when_checking_parity():
    """`select`/`imbalance` cannot change the feature matrix, so re-stating them
    at inference time is not a conflict."""
    T = _track("har:HARTrack")
    track = T()
    recs = track.smoke()
    model = track.train_baseline(recs[:4], cfg={"gravity": "mean", "imbalance": "none"})
    a = track.infer(model, recs[4])
    b = track.infer(model, recs[4], cfg={"gravity": "mean", "imbalance": "balanced"})
    np.testing.assert_array_equal(a, b)


def test_partial_infer_cfg_inherits_the_training_cfg_not_the_adapter_defaults():
    """A partial override means "same pipeline, but this one knob".

    Trained with a NON-default feature cfg (`gravity="mean"`, default is
    `"none"`), then asked to infer with a stage-5-only override. Filling the
    omitted keys from the adapter defaults used to (a) rebuild the features under
    the default `gravity` and (b) then blame the caller for a "gravity mismatch"
    they never asked for."""
    T = _track("har:HARTrack")
    track = T()
    recs = track.smoke()
    model = track.train_baseline(recs[:4], cfg={"gravity": "mean"})
    assert track._cfg(None).get("gravity") != "mean", \
        "test is vacuous unless the training cfg is off-default"

    baseline = track.infer(model, recs[4])                     # cfg omitted entirely
    partial = track.infer(model, recs[4], cfg={"imbalance": "balanced"})   # no raise
    np.testing.assert_array_equal(partial, baseline)

    # the resolved cfg really did inherit the training value, not the default
    resolved = track._infer_cfg(model, {"imbalance": "balanced"})
    assert resolved["gravity"] == "mean"
    assert resolved["imbalance"] == "balanced"


def test_partial_infer_cfg_still_catches_a_genuine_feature_conflict():
    """The converse: the fix must not blunt the check it made less trigger-happy.
    A key the caller ACTUALLY named, that ACTUALLY differs, still raises."""
    T = _track("har:HARTrack")
    track = T()
    recs = track.smoke()
    model = track.train_baseline(recs[:4], cfg={"gravity": "mean"})
    with pytest.raises(ValueError, match="TRAINED under"):
        track.infer(model, recs[4], cfg={"gravity": "none"})
    # ...even when it is buried among harmless stage-4/5 keys
    with pytest.raises(ValueError, match="gravity"):
        track.infer(model, recs[4], cfg={"imbalance": "balanced", "gravity": "highpass"})


def test_partial_infer_cfg_is_still_capability_checked():
    """A partial override is not a way past `UnsupportedCfgKey` either."""
    T = _track("har:HARTrack")
    track = T()
    recs = track.smoke()
    model = track.train_baseline(recs[:4], cfg={"gravity": "mean"})
    with pytest.raises(UnsupportedCfgKey, match="spectral_method"):
        track.infer(model, recs[4], cfg={"spectral_method": "ar"})


def test_write_submission_accepts_a_stage_5_only_override(tmp_path):
    """`write_submission()` shares `_infer_cfg`, so it inherits the same
    partial-override semantics."""
    T = _track("har:HARTrack")
    track = T()
    recs = track.smoke()
    model = track.train_baseline(recs[:4], cfg={"gravity": "mean"})
    p = track.write_submission(recs[4:5], str(tmp_path / "p.csv"), model,
                               cfg={"imbalance": "balanced"})
    rows = [ln.split(",") for ln in open(p).read().strip().splitlines()[1:]]
    preds = np.array([r[-1] for r in rows])
    np.testing.assert_array_equal(preds, track.infer(model, recs[4]).astype(str))


def test_write_submission_inherits_the_training_cfg(tmp_path):
    T = _track("har:HARTrack")
    track = T()
    recs = track.smoke()
    model = track.train_baseline(recs[:4], cfg={"gravity": "highpass"})
    p = track.write_submission(recs[4:5], str(tmp_path / "predictions.csv"), model)
    rows = [ln.split(",") for ln in open(p).read().strip().splitlines()[1:]]
    preds = np.array([r[-1] for r in rows])
    np.testing.assert_array_equal(preds, track.infer(model, recs[4]).astype(str))
    with pytest.raises(ValueError, match="TRAINED under"):
        track.write_submission(recs[4:5], str(tmp_path / "x.csv"), model,
                               cfg={"gravity": "none"})


def test_model_cfg_cannot_be_mutated_after_training():
    """The parity guarantee is only real if the training cfg is a HISTORICAL
    FACT. Before this, `model.cfg["gravity"] = "none"` rewrote the contract and
    `infer()` silently started building different features while every number
    already reported still described the old pipeline."""
    T = _track("har:HARTrack")
    track = T()
    recs = track.smoke()
    model = track.train_baseline(recs[:4], cfg={"gravity": "mean"})
    before = track.infer(model, recs[4])

    # every in-place mutation route is refused...
    for attack in (lambda: model.cfg.__setitem__("gravity", "none"),
                   lambda: model.cfg.update({"gravity": "none"}),
                   lambda: model.cfg.setdefault("gravity", "none"),
                   lambda: model.cfg.pop("gravity"),
                   lambda: model.cfg.clear(),
                   lambda: model.cfg.__delitem__("gravity")):
        with pytest.raises(TypeError):
            attack()
    # ...and so is swapping the whole contract out
    with pytest.raises(TypeError):
        model.cfg = {"gravity": "none"}

    # the contract still says what it said, and inference still obeys it
    assert model.cfg["gravity"] == "mean"
    np.testing.assert_array_equal(track.infer(model, recs[4]), before)

    # reading / copying is untouched — the cfg is inspectable, just not editable
    assert dict(model.cfg)["gravity"] == "mean"
    assert "select_k" in model.cfg


def test_model_cfg_is_a_deep_snapshot_not_a_live_reference():
    """Sealing the top level is not enough: the caller's own dict (and any nested
    value in it) must not be a back door into the model's contract."""
    T = _track("sleep_edf:SleepEDFTrack")
    track = T()
    recs = track.smoke(n_subjects=3, n_epochs=30)
    live = {"eeg_band": [0.5, 40.0], "select": "anova", "select_k": 5}
    model = track.train_baseline(recs[:2], cfg=live)
    before = track.infer(model, recs[2])

    live["eeg_band"][1] = 8.0          # mutate the caller's nested value
    live["select_k"] = 99
    assert list(model.cfg["eeg_band"]) == [0.5, 40.0]
    assert model.cfg["select_k"] == 5
    np.testing.assert_array_equal(track.infer(model, recs[2]), before)


def test_model_cfg_survives_pickling_and_stays_frozen():
    import pickle
    T = _track("har:HARTrack")
    track = T()
    model = track.train_baseline(track.smoke()[:4], cfg={"gravity": "mean"})
    revived = pickle.loads(pickle.dumps(model))
    assert revived.cfg["gravity"] == "mean"
    with pytest.raises(TypeError):
        revived.cfg["gravity"] = "none"


def test_holdout_score_is_cfg_consistent():
    T = _track("har:HARTrack")
    track = T()
    recs = track.smoke()
    rep = track.holdout_score(recs[:4], recs[4:], cfg={"gravity": "mean"})
    assert np.isfinite(rep["accuracy"])


# ------------------------------------------------------------------ item 2
@pytest.mark.parametrize("spec", ALL_TRACKS)
def test_every_track_declares_its_capabilities(spec):
    T = _track(spec)
    keys = T.supported_cfg_keys()
    assert BASE_CFG_KEYS <= keys, "stages 4-5 are owned by the base class on every track"
    assert isinstance(T.SUPPORTED_CFG_KEYS, (set, frozenset))


@pytest.mark.parametrize("spec,key", [
    ("har:HARTrack", "spectral_method"),
    ("har:HARTrack", "preprocess"),
    ("ctg_ctu_uhb:CTGTrack", "spectral_method"),
    ("ctg_ctu_uhb:CTGTrack", "impulsive"),
    ("emg_ninapro:EMGNinaproTrack", "spectral_method"),
    ("bci_eegmmidb:BCIEEGMMIDBTrack", "preprocess"),
    ("bci_eegmmidb:BCIEEGMMIDBTrack", "powerline"),
])
def test_unsupported_cfg_keys_fail_loudly(spec, key):
    """A menu option the track's own code never reads must raise, not no-op."""
    T = _track(spec)
    with pytest.raises(UnsupportedCfgKey) as e:
        T(cfg={key: "median" if key != "spectral_method" else "ar"})
    assert key in str(e.value)
    assert "NO EFFECT" in str(e.value)
    # and the same key is refused at call time, not only at construction
    track = T()
    with pytest.raises(UnsupportedCfgKey):
        track.build_dataset(track.smoke()[:1], cfg={key: "ar"})


@pytest.mark.parametrize("spec,key,value", [
    ("sleep_edf:SleepEDFTrack", "spectral_method", "multitaper"),
    ("sleep_edf:SleepEDFTrack", "preprocess", "wavelet"),
    ("ecg_cinc2017:ECGCinC2017Track", "spectral_method", "multitaper"),
    ("bci_eegmmidb:BCIEEGMMIDBTrack", "spectral_method", "multitaper"),
    ("har:HARTrack", "gravity", "mean"),
])
def test_declared_options_actually_change_the_features(spec, key, value):
    """The converse guarantee: a menu that IS offered must move something."""
    T = _track(spec)
    recs = T().smoke()[:2]
    X0, _, _ = T().build_dataset(recs)
    X1, _, _ = T().build_dataset(recs, cfg={key: value})
    assert X0.shape == X1.shape
    assert not np.allclose(X0, X1), f"{spec} advertises {key}={value!r} but it is a no-op"


def test_declare_cfg_keys_widens_the_contract():
    T = _track("bci_eegmmidb:BCIEEGMMIDBTrack")
    track = T().declare_cfg_keys("csp_components")
    assert track.supports("csp_components")
    track.build_dataset(track.smoke()[:1], cfg={"csp_components": 4})   # no raise


@pytest.mark.parametrize("spec,key", [
    ("har:HARTrack", "spectral_method"),
    ("ctg_ctu_uhb:CTGTrack", "impulsive"),
    ("emg_ninapro:EMGNinaproTrack", "spectral_method"),
])
def test_mutating_track_cfg_after_construction_is_still_validated(spec, key):
    """Constructor-time validation alone left a second door open: `track.cfg` is a
    plain dict, so `track.cfg["spectral_method"] = "ar"` reintroduced exactly the
    silently-inert option the capability system exists to abolish. Every cfg
    resolution re-checks the adapter's own cfg."""
    T = _track(spec)
    track = T()
    track.cfg[key] = "ar" if key == "spectral_method" else "median"
    recs = track.smoke()[:1]
    with pytest.raises(UnsupportedCfgKey) as e:
        track.build_dataset(recs)
    assert key in str(e.value) and "NO EFFECT" in str(e.value)
    # the other cfg-resolving entry points are closed too
    with pytest.raises(UnsupportedCfgKey):
        track.run_smoke()
    with pytest.raises(UnsupportedCfgKey):
        track.train_baseline(recs)


def test_declare_cfg_keys_reopens_a_post_construction_mutation():
    """The re-validation is a capability check, not a freeze: declare the key and
    the same mutation becomes legal."""
    T = _track("bci_eegmmidb:BCIEEGMMIDBTrack")
    track = T()
    track.cfg["csp_components"] = 4
    with pytest.raises(UnsupportedCfgKey):
        track.build_dataset(track.smoke()[:1])
    track.declare_cfg_keys("csp_components")
    track.build_dataset(track.smoke()[:1])              # no raise


def test_notebook_only_rebuild_key_is_still_refused():
    """`rebuild` is the A/B cell's own flag; it must be popped, never forwarded."""
    T = _track("har:HARTrack")
    with pytest.raises(UnsupportedCfgKey, match="rebuild"):
        T(cfg={"rebuild": True})


# ------------------------------------------------------------------ item 3/4
def _imbalanced(n=140, scale=1.0, seed=0):
    """Two informative columns plus one huge-magnitude pure-noise column.

    `scale` multiplies every column by a different constant, so a distance-based
    neighbour search that runs BEFORE standardisation sees a different geometry
    while a search that runs AFTER standardisation sees an identical one."""
    rng = np.random.default_rng(seed)
    y = np.array(["maj"] * (n - 30) + ["min"] * 30)
    X = rng.normal(size=(n, 3))
    X[y == "min", 0] += 1.4
    X[y == "min", 1] -= 1.1
    X[:, 2] = rng.normal(scale=50.0, size=n)          # loud, uninformative
    return X * np.array([1.0, 1.0, 1.0]) * scale, y


def test_smote_resamples_after_scaling_and_is_therefore_unit_invariant():
    """SMOTE's neighbour search must happen in standardised space.

    Multiplying one raw column by 1000 cannot change which patients get
    synthesised if the scaler ran first — StandardScaler maps both versions to
    the same matrix. If the sampler ran first (the bug), the noise column would
    dominate the k-NN distance and the synthetic minority set would change."""
    X, y = _imbalanced()
    Xte = np.random.default_rng(7).normal(size=(40, 3)) * np.array([1.0, 1.0, 50.0])

    a = SMOTEd(estimator=default_baseline(imbalance="none"), random_state=0).fit(X, y)
    Xs = X.copy()
    Xs[:, 2] *= 1000.0                                # same data, different units
    b = SMOTEd(estimator=default_baseline(imbalance="none"), random_state=0).fit(Xs, y)

    Xte_s = Xte.copy()
    Xte_s[:, 2] *= 1000.0
    np.testing.assert_array_equal(a.predict(Xte), b.predict(Xte_s))


def test_smote_pipeline_puts_the_sampler_after_the_scaler():
    X, y = _imbalanced()
    m = SMOTEd(estimator=default_baseline(imbalance="none"), random_state=0).fit(X, y)
    names = [n for n, _ in m.estimator_.steps]
    assert names.index("scale") < names.index("resample") < names.index("clf")


def test_smote_stays_fold_safe_nothing_is_fit_outside_fit():
    """The sampler must be a fit-time-only step: predict() must not resample and
    must not need y."""
    X, y = _imbalanced()
    m = SMOTEd(estimator=default_baseline(imbalance="none"), random_state=0).fit(X, y)
    p1 = m.predict(X[:20])
    p2 = m.predict(X[:20])
    np.testing.assert_array_equal(p1, p2)
    assert len(p1) == 20


def test_smote_falls_back_loudly_when_the_minority_is_a_singleton():
    X = np.random.default_rng(0).normal(size=(40, 3))
    y = np.array(["maj"] * 39 + ["min"])
    with pytest.warns(RuntimeWarning, match="too few to interpolate"):
        m = SMOTEd(estimator=default_baseline(imbalance="none"), random_state=0).fit(X, y)
    assert m.predict(X[:5]).shape == (5,)


def test_smote_with_a_balanced_fold_is_a_no_op():
    X = np.random.default_rng(0).normal(size=(40, 3))
    y = np.array(["a", "b"] * 20)
    m = SMOTEd(estimator=default_baseline(imbalance="none"), random_state=0).fit(X, y)
    assert m.predict(X[:5]).shape == (5,)


# ------------------------------------------------------------------ item 5
@pytest.mark.parametrize("make", [
    lambda: OverSampled(estimator=default_baseline(imbalance="none")),
    lambda: SMOTEd(estimator=default_baseline(imbalance="none")),
    lambda: ThresholdMoved(estimator=default_baseline(imbalance="none")),
])
def test_nested_set_params_reaches_the_wrapped_pipeline(make):
    from sklearn.base import clone
    est = make()
    est.set_params(estimator__clf__max_depth=5)
    assert est.estimator.named_steps["clf"].max_depth == 5
    assert not hasattr(est, "estimator__clf__max_depth"), \
        "shallow set_params created a useless top-level attribute"
    assert est.get_params(deep=True)["estimator__clf__max_depth"] == 5

    fresh = clone(est)
    assert fresh.estimator.named_steps["clf"].max_depth == 5
    fresh.set_params(estimator__clf__max_depth=2)
    assert fresh.estimator.named_steps["clf"].max_depth == 2
    assert est.estimator.named_steps["clf"].max_depth == 5, "clone() shared state"


def test_grid_search_over_a_wrapped_estimator_actually_varies_the_model():
    """The consequence of the bug: without proper delegation every candidate is
    the same model, so every candidate scores identically."""
    from sklearn.model_selection import GridSearchCV
    X, y = _imbalanced(n=120)
    gs = GridSearchCV(
        SMOTEd(estimator=default_baseline(imbalance="none"), random_state=0),
        {"estimator__clf__max_depth": [1, 2, None]}, cv=3, scoring="balanced_accuracy")
    gs.fit(X, y)
    scores = gs.cv_results_["mean_test_score"]
    assert len(set(np.round(scores, 6))) > 1, \
        "every GridSearchCV candidate scored identically — set_params did not propagate"


# ------------------------------------------------------------------ smoke
@pytest.mark.parametrize("spec", ALL_TRACKS)
def test_run_smoke_still_green(spec):
    rep = _track(spec)().run_smoke()
    assert np.isfinite(rep["accuracy"])
    assert rep["spread_unit"] in ("subject", "record", "patient", "session", "fold")
