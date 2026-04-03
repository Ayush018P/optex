from src.backtesting.engine import BacktestEngine

def test_backtest_runs():
    engine = BacktestEngine(episodes=2)
    metrics = engine.run()
    assert not metrics.is_empty()
