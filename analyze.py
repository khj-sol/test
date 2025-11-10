"""기술적 지표를 이용해 주식을 분석하는 간단한 CLI 스크립트."""

from __future__ import annotations

import sys
from typing import Tuple

import pandas as pd
import yfinance as yf


RSI_COLUMN = "RSI_14"
SMA20_COLUMN = "SMA_20"
SMA50_COLUMN = "SMA_50"
MACD_COLUMN = "MACD_12_26_9"
MACD_SIGNAL_COLUMN = "MACDs_12_26_9"


def calculate_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """수익률을 기반으로 RSI(Relative Strength Index)를 계산합니다."""

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.rolling(window=length, min_periods=length).mean()
    average_loss = losses.rolling(window=length, min_periods=length).mean()

    zero_loss = average_loss == 0
    average_loss = average_loss.replace(0, pd.NA)
    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(~zero_loss, 100)
    return rsi


def calculate_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series]:
    """MACD와 시그널 선을 계산합니다."""

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def append_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """주가 데이터프레임에 주요 기술적 지표 열을 추가합니다."""

    data = data.copy()
    close = data["Close"]

    data[RSI_COLUMN] = calculate_rsi(close)
    data[SMA20_COLUMN] = close.rolling(window=20, min_periods=20).mean()
    data[SMA50_COLUMN] = close.rolling(window=50, min_periods=50).mean()

    macd_line, signal_line = calculate_macd(close)
    data[MACD_COLUMN] = macd_line
    data[MACD_SIGNAL_COLUMN] = signal_line
    return data


def analyze_stock(ticker: str) -> None:
    """주어진 티커에 대한 기술적 분석을 수행하고 결과를 출력합니다."""

    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if data.empty:
            print(f"오류: '{ticker}'에 대한 데이터를 찾을 수 없습니다. 티커를 확인해 주세요.")
            return

        data = append_indicators(data)
        latest_data = data.iloc[-1]

        print("---" * 15)
        print(
            f"📊 {ticker} 기술적 분석 결과 (최근 거래일: "
            f"{latest_data.name.strftime('%Y-%m-%d')})"
        )
        print("---" * 15)

        print(f"종가: ${latest_data['Close']:.2f}")
        print("\n--- 주요 지표 ---")

        rsi_14 = latest_data.get(RSI_COLUMN)
        if pd.notna(rsi_14):
            print(f"RSI (14일): {rsi_14:.2f}")
            if rsi_14 > 70:
                print("  -> 📈 상태: 과매수 구간 (과열)")
            elif rsi_14 < 30:
                print("  -> 📉 상태: 과매도 구간 (침체)")
            else:
                print("  -> 📊 상태: 중립 구간")
        else:
            print("RSI (14일): 계산되지 않았습니다.")

        sma_20 = latest_data.get(SMA20_COLUMN)
        sma_50 = latest_data.get(SMA50_COLUMN)
        print("\n이동평균선 (SMA):")
        if pd.notna(sma_20):
            print(f"  - 20일선: ${sma_20:.2f}")
        else:
            print("  - 20일선: 계산되지 않았습니다.")
        if pd.notna(sma_50):
            print(f"  - 50일선: ${sma_50:.2f}")
        else:
            print("  - 50일선: 계산되지 않았습니다.")
        if pd.notna(sma_20) and pd.notna(sma_50):
            if sma_20 > sma_50:
                print("  -> 📈 상태: 단기 골든 크로스 (상승 추세)")
            else:
                print("  -> 📉 상태: 단기 데드 크로스 (하락 추세)")

        macd_line = latest_data.get(MACD_COLUMN)
        signal_line = latest_data.get(MACD_SIGNAL_COLUMN)
        print("\nMACD (12, 26, 9):")
        if pd.notna(macd_line):
            print(f"  - MACD 선: {macd_line:.2f}")
        else:
            print("  - MACD 선: 계산되지 않았습니다.")
        if pd.notna(signal_line):
            print(f"  - 시그널 선: {signal_line:.2f}")
        else:
            print("  - 시그널 선: 계산되지 않았습니다.")
        if pd.notna(macd_line) and pd.notna(signal_line):
            if macd_line > signal_line:
                print("  -> 📈 상태: 매수 신호 (상승 모멘텀)")
            else:
                print("  -> 📉 상태: 매도 신호 (하락 모멘텀)")

    except Exception as error:
        print(f"분석 중 오류 발생: {error}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python analyze.py [주식코드]")
        print("예시 (애플): python analyze.py AAPL")
        print("예시 (삼성전자): python analyze.py 005930.KS")
    else:
        stock_ticker = sys.argv[1].upper()
        analyze_stock(stock_ticker)
