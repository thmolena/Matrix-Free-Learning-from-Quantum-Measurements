import numpy as np
from ctpcpinn.adaptive_ramsey import run_study


def test_nested_gate_and_confirmation_result():
    result=run_study(); assert result["dataset"]["observations"]==40787; assert result["dataset"]["traces"]==177
    assert result["protocol"]["candidates_per_trace"]==29
    assert result["overall"]["adaptive_mean_rmse"] < result["overall"]["linear_mean_rmse"]
    assert result["overall"]["spectral_selected"]==91


def test_gate_uses_disjoint_index_classes():
    index=np.arange(64); outer=index%4!=0; development=outer&(index%8==3); inner=outer&~development
    assert not np.any(development & inner); assert not np.any((~outer)&development)
