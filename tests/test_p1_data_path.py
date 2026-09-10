import pandas as pd


def _valid_frame(rows=30):
    idx = pd.date_range("2026-01-01", periods=rows, freq="D")
    return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=idx)


def test_canonical_price_bucket_boundaries():
    from src.utils import price_bucket
    expected = {
        9.99: "<10", 10: "10-49", 49.99: "10-49", 50: "50-99",
        99.99: "50-99", 100: "100-249", 249.99: "100-249",
        250: "250-499", 499.99: "250-499", 500: "500-999",
        999.99: "500-999", 1000: "1000-2499", 2499.99: "1000-2499",
        2500: ">2500",
    }
    for value, label in expected.items():
        assert price_bucket(value) == label


def test_universe_cap_is_applied_before_download(monkeypatch):
    import src.market_data as md
    import src
    monkeypatch.setattr(src, "MAX_UNIVERSE", 3)
    calls = []

    def fake_download(symbol, period=None, retries=2):
        calls.append(symbol)
        return _valid_frame()

    monkeypatch.setattr(src, "_incremental_download_symbol", fake_download)
    result = src._bounded_download_many(["A", "B", "C", "D", "E"], workers=1)
    assert calls == ["A", "B", "C"]
    assert set(result) == {"A", "B", "C"}
    assert md.download_many is src._bounded_download_many


def test_five_year_cache_path_requests_only_missing_ranges(monkeypatch):
    import src
    import src.market_data as md
    monkeypatch.setattr(md, "_read_cached_ohlcv", lambda symbol: _valid_frame())
    monkeypatch.setattr(md, "_save_cached_ohlcv", lambda symbol, df: None)
    ranges = []

    def fake_range(ticker, start=None, end=None, period=None, retries=2):
        ranges.append((start, end, period, retries))
        return _valid_frame()

    monkeypatch.setattr(md, "_download_range", fake_range)
    src._incremental_download_symbol("TEST", "5y", retries=0)
    assert ranges
    assert all(period is None for _, _, period, _ in ranges)
    assert any(start is not None for start, _, _, _ in ranges)
