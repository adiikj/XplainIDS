"""Evidence per alert type and the phi_hat inverse-scaling round trip."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evidence import build_record, decode_value, known_evidence, novel_evidence  # noqa: E402

NAMES = ['duration', 'flag', 'src_bytes', 'serror_rate', 'count', 'service']
CATS = {'flag': ['SF', 'S0', 'REJ'], 'service': ['http', 'private']}


def test_decode_value():
    assert decode_value('flag', 1.0, CATS) == 'S0'
    assert decode_value('flag', -1.0, CATS) == 'unseen'
    assert decode_value('count', 511.0, CATS) == 511
    assert decode_value('serror_rate', 0.25, CATS) == 0.25


def test_known_evidence_positive_only_and_sorted():
    phi = np.array([-0.5, 0.3, 0.05, 0.4, -0.2, 0.0])
    ev = known_evidence(phi, np.array([0, 1, 10, 1.0, 511, 0]), NAMES, CATS)
    assert [e['name'] for e in ev] == ['serror_rate', 'flag']    # benign-pushing, zero and < 20% of the top left out
    assert ev[1]['raw_value'] == 'S0'


def test_novel_evidence_largest_deviation():
    phi = np.array([0.04, 0.30, 0.0, 0.02, -0.20, 0.0])
    phi_hat = np.array([0.01, 0.02, 0.0, 0.02, 0.05, 0.0])
    ev = novel_evidence(phi, phi_hat, np.zeros(6), NAMES, CATS)
    assert [e['name'] for e in ev] == ['flag', 'count']                 # duration (|0.03| < 20% of 0.28) dropped
    assert ev[0]['deviation'] == pytest.approx(0.28) and ev[1]['deviation'] == pytest.approx(-0.25)
    assert 'more' in ev[0]['direction'] and 'less' in ev[1]['direction']


def test_build_record_types():
    phi, ph, x = np.full(6, 0.1), np.zeros(6), np.zeros(6)
    assert build_record(1, 1, .9, .1, .5, .2, phi, None, x, NAMES, CATS)['alert_type'] == 'KNOWN'
    r = build_record(1, 2, .1, .3, .5, .2, phi, ph, x, NAMES, CATS)
    assert r['alert_type'] == 'NOVEL' and 'phi_hat' in r['evidence'][0] and len(r['evidence']) == 5
    with pytest.raises(ValueError):
        build_record(1, 2, .1, .3, .5, .2, phi, None, x, NAMES, CATS)
    with pytest.raises(ValueError):
        build_record(1, 0, .1, .1, .5, .2, phi, ph, x, NAMES, CATS)


def test_inverse_shap_round_trip():
    pytest.importorskip('torch')
    from detectors import InputTransform
    rng = np.random.default_rng(0)
    X, S = rng.lognormal(size=(500, 6)), rng.normal(scale=0.1, size=(500, 6))
    for kind in ('shap', 'concat'):
        T = InputTransform(kind).fit(X, S)
        np.testing.assert_allclose(T.inverse_shap(T.transform(X, S)), S, atol=1e-5)
