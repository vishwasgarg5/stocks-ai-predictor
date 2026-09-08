import numpy as np

import src.market_data as md


def test_market_snapshot_contains_broad_and_sector_indices(monkeypatch):
    def fake_snapshot(symbol, period="3mo", cutoff=None):
        return {
            "Close": 100.0,
            "Change1D": 1.0,
            "Open": 99.0,
            "High": 101.0,
            "Low": 98.0,
            "High52W": 120.0,
            "Low52W": 80.0,
            "Source": symbol,
            "DataDate": "2026-09-07",
        }

    monkeypatch.setattr(md, "_index_snapshot", fake_snapshot)
    snap = md.get_market_snapshot({}, "2026-09-07")

    for key in [
        "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "VIX",
        "NIFTYNEXT50", "NIFTY100", "NIFTY200", "NIFTY500",
        "MIDCAP150", "SMALLCAP250", "IT", "AUTO", "PHARMA",
        "METAL", "ENERGY", "REALTY", "PSUBANK", "Breadth",
    ]:
        assert key in snap
        if key != "Breadth":
            assert np.isfinite(snap[key]["Close"])


def test_finnifty_has_fallback_candidates():
    candidates = md.INDEX_SYMBOL_FALLBACKS["FINN"]
    assert "NIFTY_FIN_SERVICE.NS" in candidates
    assert "^CNXFINANCE" in candidates
