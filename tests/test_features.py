import polars as pl
from src.data.features import compute_features


def test_feature_columns():
    df = pl.DataFrame({
        "stock_id": ["S"],
        "episode": [0],
        "time_step": [0],
        "bid_price1": [100.0],
        "ask_price1": [100.1],
        "bid_size1": [200],
        "ask_size1": [180],
        "bid_price2": [99.9],
        "ask_price2": [100.2],
        "bid_size2": [150],
        "ask_size2": [140],
        "bid_price3": [99.8],
        "ask_price3": [100.3],
        "bid_size3": [120],
        "ask_size3": [110],
        "trade_price": [100.05],
        "trade_size": [50],
        "trade_dir": [1],
        "cumulative_volume": [50],
    })
    lf = compute_features(df.lazy())
    cols = lf.collect().columns
    for col in [
        "mid_price",
        "spread_bps",
        "queue_imbalance",
        "depth_ratio_3l",
        "weighted_mid",
        "realized_vol",
        "ofi",
        "kyle_lambda",
        "amihud",
        "tick_dir",
        "volume_clock",
        "mom_5",
        "mom_10",
        "mom_30",
    ]:
        assert col in cols
