import time
from dataclasses import dataclass
from typing import Optional

import ccxt
import numpy as np
import pandas as pd


# =========================
#  CONFIG
# =========================
SYMBOL = "BTC/USDT"     # будемо брати свічки з Binance (spot), рахуємо як ф’ючерси
TIMEFRAME = "1h"        # HTF
WINDOW = 300            # кількість свічок в історії
SLEEP_SECONDS = 60      # пауза між циклами


# =========================
#  STRATEGY CONFIG
# =========================
@dataclass
class CupWithHandleConfig:
    ema_fast: int = 20
    ema_trend: int = 100

    cup_lookback: int = 80
    min_cup_depth: float = 0.08
    max_cup_depth: float = 0.35

    handle_lookback: int = 15
    max_handle_depth_factor: float = 0.4
    level_tolerance: float = 0.02

    tp_r_multiple: float = 2.0
    sl_buffer: float = 0.001

    use_short: bool = False   # поки тільки long


class CupWithHandleStrategy:
    def __init__(self, cfg: CupWithHandleConfig):
        self.cfg = cfg

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Додає в df колонки signal, sl_price, tp_price."""
        if df.empty:
            return df.copy()

        data = df.copy()

        # EMA як фільтр тренду
        data["ema_fast"] = data["close"].ewm(
            span=self.cfg.ema_fast, adjust=False
        ).mean()
        data["ema_trend"] = data["close"].ewm(
            span=self.cfg.ema_trend, adjust=False
        ).mean()

        data["up_trend"] = (data["close"] > data["ema_trend"]) & \
                           (data["ema_fast"] > data["ema_trend"])

        data["signal"] = 0
        data["sl_price"] = np.nan
        data["tp_price"] = np.nan

        n = len(data)
        cup_lb = self.cfg.cup_lookback
        handle_lb = self.cfg.handle_lookback

        if n < cup_lb + handle_lb + 1:
            return data

        for i in range(cup_lb + handle_lb, n):
            if not data["up_trend"].iloc[i]:
                continue

            # вікно чашки
            start_cup = i - cup_lb
            end_cup = i - 1
            cup = data.iloc[start_cup:end_cup]

            cup_high = cup["close"].max()
            cup_low = cup["close"].min()
            if cup_high <= 0:
                continue

            depth = (cup_high - cup_low) / cup_high
            if not (self.cfg.min_cup_depth <= depth <= self.cfg.max_cup_depth):
                continue

            # вікно ручки
            start_handle = i - handle_lb
            handle = data.iloc[start_handle:i]

            handle_high = handle["close"].max()
            handle_low = handle["close"].min()
            last_h = handle["close"].iloc[-1]

            lvl_diff = abs(last_h - cup_high) / cup_high
            if lvl_diff > self.cfg.level_tolerance:
                continue

            handle_depth = (handle_high - handle_low) / cup_high
            if handle_depth > self.cfg.max_handle_depth_factor * depth:
                continue

            close = data["close"].iloc[i]

            # пробій ручки вгору
            if close > handle_high:
                data.iloc[i, data.columns.get_loc("signal")] = 1

                sl = handle_low * (1 - self.cfg.sl_buffer)
                risk = close - sl
                tp = close + self.cfg.tp_r_multiple * risk

                data.iloc[i, data.columns.get_loc("sl_price")] = sl
                data.iloc[i, data.columns.get_loc("tp_price")] = tp

        return data


# =========================
#  PAPER TRADER
# =========================
@dataclass
class Position:
    side: int = 0   # 1 = long, -1 = short
    qty: float = 0.0
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0


class Trader:
    def __init__(self, initial_equity: float = 1000.0, risk_per_trade: float = 0.01):
        self.equity = initial_equity
        self.risk_per_trade = risk_per_trade
        self.pos = Position()

    def on_bar(self, row: pd.Series):
        ts = row["timestamp"]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        sig = int(row.get("signal", 0))
        sl = row.get("sl_price", np.nan)
        tp = row.get("tp_price", np.nan)

        # EXIT для long
        if self.pos.side == 1:
            exit_p: Optional[float] = None
            reason: Optional[str] = None

            if l <= self.pos.sl:
                exit_p = self.pos.sl
                reason = "SL"
            elif h >= self.pos.tp:
                exit_p = self.pos.tp
                reason = "TP"

            if exit_p is not None:
                pnl = (exit_p - self.pos.entry) * self.pos.qty
                self.equity += pnl
                print(f"[{ts}] EXIT {reason} @ {exit_p:.2f}, pnl={pnl:.2f}, equity={self.equity:.2f}")
                self.pos = Position()
                return

        # ENTRY
        if self.pos.side == 0 and sig == 1:
            if pd.isna(sl) or pd.isna(tp):
                return

            risk_money = self.equity * self.risk_per_trade
            risk_unit = c - sl
            if risk_unit <= 0:
                return

            qty = risk_money / risk_unit
            if qty <= 0:
                return

            self.pos = Position(side=1, qty=qty, entry=c, sl=sl, tp=tp)
            print(
                f"[{ts}] ENTRY LONG @ {c:.2f}, SL={sl:.2f}, TP={tp:.2f}, "
                f"qty={qty:.4f}, equity={self.equity:.2f}"
            )


# =========================
#  EXCHANGE / DATA
# =========================
def create_binance_client():
    exchange = ccxt.binance({
        "enableRateLimit": True,
    })
    return exchange


def ohlcv_to_df(ohlcv):
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)
    return df


# =========================
#  MAIN LOOP
# =========================
def main():
    print("=== Binance Futures cup-with-handle paper bot (1h) стартував ===")

    ex = create_binance_client()
    strat = CupWithHandleStrategy(CupWithHandleConfig())
    trader = Trader(initial_equity=1000.0, risk_per_trade=0.01)

    while True:
        try:
            # тягнемо останні свічки
            ohlcv = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=WINDOW)
            if not ohlcv:
                print("Binance не повернув даних")
                time.sleep(SLEEP_SECONDS)
                continue

            df = ohlcv_to_df(ohlcv)

            if df.empty or len(df) < 2:
                print("df порожній або замало свічок")
                time.sleep(SLEEP_SECONDS)
                continue

            # працюємо тільки по ЗАКРИТИХ свічках
            df_closed = df.iloc[:-1].copy()

            df_sig = strat.generate(df_closed)
            if df_sig.empty:
                print("df_sig порожній")
                time.sleep(SLEEP_SECONDS)
                continue

            last = df_sig.iloc[-1]
            trader.on_bar(last)

        except Exception as e:
            print("Помилка в main loop:", e)

        # лог, щоб бачити, що бот живий
        print("Цикл завершено, чекаю", SLEEP_SECONDS, "секунд...\n")
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
