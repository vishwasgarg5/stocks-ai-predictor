import pandas as pd

from src.data_snapshot import canonicalize


def _frame():
    idx = pd.to_datetime(["2026-09-03", "2026-09-04", "2026-09-07", "2026-09-08"])
    return pd.DataFrame({"Open":[1,2,3,4],"High":[2,3,4,5],"Low":[0.5,1.5,2.5,3.5],"Close":[1.5,2.5,3.5,4.5],"Volume":[10,10,10,10]}, index=idx)


def test_canonicalize_never_leaks_future_rows():
    out = canonicalize(_frame(), "2026-09-07")
    assert str(out.index.max().date()) == "2026-09-07"
    assert len(out) == 3


def test_canonicalize_sorts_and_deduplicates():
    x = _frame().iloc[[2, 1, 1, 0]]
    out = canonicalize(x, "2026-09-07")
    assert out.index.is_monotonic_increasing
    assert not out.index.duplicated().any()
