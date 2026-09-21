"""
Phase 1 합성 생성기 검증.

핵심은 순환검증 방지(R1/R2)와 내적 타당도:
  - 라벨 토큰 누출 0
  - states에 cluster_id 부재, narrator가 cluster_id 거부
  - 결정론성(같은 시드 → 동일 문장)
  - delta=0 무주입 대조군은 텍스트→라벨 예측이 chance 수준,
    delta=1은 신호가 회수되어 예측 정확도 상승
  - 생성물 저장(관측/정답/로그)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synth import generate  # noqa: E402
from synth.config import SynthConfig  # noqa: E402
from synth.narrator import TemplateNarrator  # noqa: E402
from synth.physical import build_physical_states  # noqa: E402


def test_states에_cluster_id_없음():
    cfg = SynthConfig(n_records=100, n_clusters=4, seed=1)
    states, truth = build_physical_states(cfg)
    assert "cluster_id" not in states.columns          # 관측 물리상태에 라벨 없음
    assert "cluster_id" in truth.columns               # 정답 파일에만 라벨
    assert len(states) == 100 == len(truth)


def test_narrator가_cluster_id_거부():
    nar = TemplateNarrator(seed=0)
    with pytest.raises(ValueError):
        nar.render({"part": "구동모터", "symptom_code": "과열",
                    "environment": "고온", "cluster_id": 2})


def test_라벨토큰_누출_0():
    cfg = SynthConfig(n_records=200, n_clusters=4, delta=0.7, seed=3)
    res = generate(cfg, save=False)
    assert res["leakage"]["token_leakage_rate"] == 0.0


def test_결정론성_같은시드_동일문장():
    cfg = SynthConfig(n_records=120, n_clusters=3, delta=0.6, seed=7)
    a = generate(cfg, save=False)["obs"]["고장증상"].tolist()
    b = generate(cfg, save=False)["obs"]["고장증상"].tolist()
    assert a == b


def test_delta0_대조군_vs_delta1_신호회수():
    base = dict(n_records=260, n_clusters=4, noise_ratio=0.2,
                expression_variance=0.5, seed=11)
    acc0 = generate(SynthConfig(delta=0.0, **base), save=False)["leakage"]["tfidf_logreg"]
    acc1 = generate(SynthConfig(delta=1.0, **base), save=False)["leakage"]["tfidf_logreg"]

    # 무주입은 chance 근방 (약간의 여유 허용)
    assert acc0["accuracy_mean"] <= acc0["chance"] + 0.20
    # 강한 주입은 신호가 회수되어 무주입보다 뚜렷이 높음
    assert acc1["accuracy_mean"] >= acc0["accuracy_mean"] + 0.10


def test_effect_class_생성되고_라벨의_결정론적함수가_아님():
    """
    effect_class는 증상(물리상태)에서 확률적으로 파생된다(Phase 3 β·α 근사용).
    군집 라벨의 결정론적 함수가 되면 순환성이 생기므로, 같은 군집 안에서도
    EFF/NEFF가 섞여야 한다(R1).
    """
    res = generate(
        SynthConfig(n_records=300, n_clusters=3, delta=1.0, noise_ratio=0.1, seed=2),
        save=False,
    )
    obs, truth = res["obs"], res["truth"]
    assert "effect_class" in obs.columns
    assert set(obs["effect_class"]) <= {"EFF", "NEFF"}

    # 라벨→effect_class가 1:1이면 결정론적 누출
    섞인_군집 = 0
    for c in sorted(set(truth["cluster_id"])):
        if c == -1:
            continue
        vals = set(obs.loc[truth["cluster_id"] == c, "effect_class"])
        if len(vals) > 1:
            섞인_군집 += 1
    assert 섞인_군집 >= 1


def test_생성물_저장(tmp_path):
    cfg = SynthConfig(n_records=80, n_clusters=3, delta=0.5, seed=0)
    res = generate(cfg, out_dir=tmp_path, save=True)

    for key in ("obs", "truth", "log"):
        assert Path(res["paths"][key]).exists()

    obs = res["obs"]
    assert len(obs) == 80
    # 관측 데이터에 잠재 물리상태가 노출되지 않아야 함
    assert "symptom_code" not in obs.columns
    assert "environment" not in obs.columns
    assert res["counts"]["n_records"] == 80
