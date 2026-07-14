# Institutional Trend Engine v2.0

A professional intraday trend trading system for SPX/SPY, built around the concept of **Turning Point Detection** using multi-factor analysis.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo with synthetic data
python main.py demo

# Run backtest with your own data
python main.py backtest --data your_data.csv

# Run Swing Low reversal strategy (buy at every swing low)
python main.py swing
python main.py swing --data your_data.csv
python main.py swing --bars 1000 --stop-atr 1.0 --tp-atr 2.0

# Run Swing Low Elevated strategy (底部抬高: buy only when current swing low > previous 2 lows)
python main.py swing2
python main.py swing2 --data your_data.csv
python main.py swing2 --bars 1000 --stop-atr 1.0 --tp-atr 2.0 --prev-lows 2

# Run Momentum Acceleration strategy (动量突破+加速度交易)
python main.py momentum
python main.py momentum --data your_data.csv
python main.py momentum --bars 1000 --stop-atr 0.5 --tp-atr 1.0 --max-hold 20

# Run live with Schwab API
cp .env.example .env  # Fill in your credentials
python main.py live --symbol SPY
```

## Architecture

```
institutional_trend_engine/
├── main.py                     # Entry point (backtest / live / demo / swing)
├── config.py                   # Global configuration
├── requirements.txt
│
├── data/                       # Data Layer
│   ├── market_data.py          # Real-time streaming (schwabdev)
│   └── historical_data.py      # Historical bar fetching
│
├── indicators/                 # Indicator Layer
│   ├── atr.py                  # Average True Range
│   ├── ema.py                  # Exponential Moving Average
│   ├── vwap.py                 # Volume-Weighted Average Price
│   ├── swing.py                # Swing High/Low detector
│   ├── momentum.py             # Velocity / Acceleration / Curvature / Slope
│   ├── volume_delta.py         # Volume Delta / CVD
│   ├── realized_vol.py         # Realized Volatility
│   ├── market_structure.py     # ★ Market Structure Engine (BOS/CHoCH/Sweep)
│   └── trend_score.py          # ★ Turning Point Score (multi-factor)
│
├── strategy/                   # Strategy Layer
│   ├── signal_engine.py        # Master signal orchestrator
│   ├── state_machine.py        # Market regime classifier
│   ├── breakout.py             # Early entry + breakout strategy
│   ├── swing_low_strategy.py   # Swing Low reversal strategy (buy every swing low)
│   ├── swing_low_elevated.py   # Swing Low Elevated strategy (bottom rising filter)
│   ├── momentum_acceleration.py # Momentum Breakout + Acceleration Trading
│   ├── stoploss.py             # Multi-stage stop-loss (initial/BE/trailing)
│   └── take_profit.py          # Scaled take-profit
│
├── risk/                       # Risk Layer
│   ├── position.py             # ATR-based position sizing
│   └── money_management.py     # Daily loss limits / circuit breakers
│
├── execution/                  # Execution Layer
│   ├── broker.py               # Schwab API adapter
│   └── order_manager.py        # Order lifecycle management
│
├── backtest/                   # Backtest Layer
│   ├── engine.py               # Event-driven backtest engine
│   ├── swing_engine.py         # Dedicated Swing Low backtest engine
│   ├── swing_elevated_engine.py # Dedicated Swing Low Elevated backtest engine
│   ├── momentum_engine.py      # Dedicated Momentum Acceleration backtest engine
│   └── report.py               # Performance report generator
│
├── visualization/              # Visualization
│   ├── chart.py                # TradingView-style charts
│   ├── swing_chart.py          # Swing Low strategy charts
│   ├── swing_elevated_chart.py # Swing Low Elevated strategy charts
│   └── momentum_chart.py       # Momentum Acceleration strategy charts
│
└── utils/                      # Utilities
    ├── logger.py               # Logging setup
    └── math_utils.py           # Math helpers
```

## Core Concepts

### Turning Point Detection (Turning Score)

The system doesn't predict tops/bottoms — it identifies when a turning point has *already begun* using a weighted multi-factor score:

| Factor | Weight | Description |
|--------|--------|-------------|
| Acceleration zero-crossing | 25 | Second derivative changes sign |
| OFI reversal | 20 | Order flow direction flips |
| Curvature increase | 15 | Price curve bending sharply |
| Slope change | 15 | Linear regression slope flips |
| VWAP reclaim | 10 | Price retakes VWAP |
| Volume Delta | 5 | Net buying pressure |
| RV expansion | 5 | Volatility release |
| GEX support | 5 | Gamma exposure alignment |

When score ≥ 80, a turning point is confirmed.

### Market Structure Engine

ICT/SMC-style structure recognition:
- **Swing High/Low** — dynamic fractal-based detection
- **HH/HL/LH/LL** — structure classification
- **BOS** (Break of Structure) — trend continuation
- **CHoCH** (Change of Character) — trend reversal
- **Liquidity Sweep** — stop-hunt detection
- **False Break** — failed breakout detection

### Early Entry Algorithm

Instead of waiting for a breakout confirmation, the system enters early when:
1. Price is within 0.2 ATR of a swing high/low
2. Velocity is increasing
3. Acceleration is increasing
4. OFI is increasing
5. VWAP slope is increasing

Then adds to the position when the actual breakout occurs.

### Risk Management

- **Initial stop**: 1.5 × ATR from entry
- **Break-even**: Move stop to entry at 1R profit
- **ATR trailing**: 2.0 × ATR trailing stop
- **Take profit**: 2R target (with 50% partial at 1R)
- **Daily circuit breaker**: 3% max daily loss
- **Consecutive loss limit**: 3 losses → halt

## Schwab API Setup

1. Register at [developer.schwab.com](https://developer.schwab.com/)
2. Create an app to get `app_key` and `app_secret`
3. Copy `.env.example` to `.env` and fill in credentials
4. Run `python main.py live --symbol SPY`

## Data Format

CSV files for backtest should have columns:
```
datetime,open,high,low,close,volume
2025-01-13 09:30:00,6000.0,6002.0,5998.0,6001.0,150000
...
```

## Output

After a backtest, the following files are generated in `output/`:
- `backtest_report.txt` — Full performance report (breakout strategy)
- `trades.csv` — Trade log (breakout strategy)
- `chart.png` — Visual chart with signals and structure
- `equity_curve.csv` — Equity over time

After a swing strategy run, the following files are generated in `output/`:
- `swing_report.txt` — Swing Low performance report
- `swing_trades.csv` — Trade log
- `swing_signals.csv` — BUY signals
- `swing_equity.csv` — Equity curve
- `swing_chart.png` — Visual chart with swing lows, signals, SL/TP, equity, and R-multiples

After a swing2 (elevated) strategy run, the following files are generated in `output/`:
- `swing2_report.txt` — Elevated Swing Low performance report
- `swing2_trades.csv` — Trade log
- `swing2_all_signals.csv` — All signals (BUY + REJECT)
- `swing2_buy_signals.csv` — BUY signals only
- `swing2_equity.csv` — Equity curve
- `swing2_chart.png` — Visual chart with swing lows, signals, SL/TP, equity, and R-multiples

After a momentum strategy run, the following files are generated in `output/`:
- `momentum_report.txt` — Momentum Acceleration performance report
- `momentum_trades.csv` — Trade log
- `momentum_signals.csv` — LONG/SHORT signals
- `momentum_zones.csv` — Detected consolidation zones
- `momentum_equity.csv` — Equity curve
- `momentum_chart.png` — Visual chart with consolidation zones, signals, SL/TP, velocity, acceleration, and equity

## License

MIT
