"""Command line tool for basic technical and fundamental analysis of equities."""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Any

import pandas as pd
import pandas_ta as ta
import yfinance as yf

# 보조 지표 컬럼 이름을 상수로 정의
RSI_COLUMN = "RSI_14"
SMA20_COLUMN = "SMA_20"
SMA50_COLUMN = "SMA_50"
MACD_COLUMN = "MACD_12_26_9"
MACD_SIGNAL_COLUMN = "MACDs_12_26_9"


def download_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download historical stock data for the given ticker."""

    # auto_adjust=False로 설정하여 'Close'와 'Adj Close'를 모두 받습니다.
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False  # 수정 종가를 따로 받기 위해 False 유지
    )

    if data.empty:
        raise ValueError(
            f"'{ticker}'에 대한 데이터를 찾을 수 없습니다. 티커와 기간/간격을 확인해 주세요."
        )

    # yfinance가 ('Open', 'PLTR') 같은 MultiIndex를 반환할 경우,
    # level 1 (티커)을 삭제하고 level 0 (지표)만 남깁니다.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1) # 0이 아닌 1을 삭제

    # 모든 컬럼 이름을 소문자로 변경합니다.
    data.columns = data.columns.str.lower()

    # 'adj close' (수정 종가)가 있는지 확인합니다.
    if 'adj close' in data.columns:
        # 'close' (실제 종가)가 있다면, 분석에 필요 없으므로 삭제합니다.
        if 'close' in data.columns:
            data = data.drop(columns=['close'])
        
        # 'adj close' (수정 종가)를 'close'로 이름을 변경하여 분석에 사용합니다.
        data.rename(columns={'adj close': 'close'}, inplace=True)
        
    elif 'close' in data.columns:
        # 'adj close'가 없고 'close'만 있는 경우 (예: 암호화폐)
        # 'close'를 그대로 사용합니다.
        pass
    else:
        # 둘 다 없는 경우 오류 발생
        raise ValueError(f"데이터에 'close' 또는 'adj close' 컬럼이 없습니다. 사용 가능한 컬럼: {list(data.columns)}")

    return data


# [!!! 신규 함수 !!!]
def get_fundamental_data(ticker_str: str) -> dict[str, Any]:
    """Get key fundamental metrics for the given ticker."""
    try:
        stock = yf.Ticker(ticker_str)
        info = stock.info
        
        # .info 딕셔너리에서 주요 지표 추출
        # .get()을 사용하면 해당 키가 없을 때 오류 대신 None을 반환
        fundamentals = {
            'per': info.get('trailingPE'),      # PER (과거 12개월)
            'forward_per': info.get('forwardPE'),  # Forward PER (향후 12개월 예상)
            'pbr': info.get('priceToBook'),      # PBR
            'roe': info.get('returnOnEquity'),   # ROE
            'margin': info.get('profitMargins')  # 순이익률
        }
        return fundamentals
    except Exception as e:
        print(f"\n[경고] 기본적 분석 데이터 가져오기 실패: (일부 티커는 지원 안됨) {e}")
        return {} # 실패 시 빈 딕셔너리 반환


def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Append technical indicators to the provided dataframe."""

    data = data.copy()
    
    # 'close' 컬럼 (수정 종가)을 명시적으로 지정합니다.
    try:
        data.ta.rsi(close='close', length=14, append=True)
        data.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
        data.ta.sma(close='close', length=20, append=True)
        data.ta.sma(close='close', length=50, append=True)
    except Exception as e:
        print(f"보조 지표 계산 중 오류 발생 (데이터 컬럼 확인 필요): {e}")
        pass 

    return data


def format_float(value: float) -> str:
    """Format a float value consistently for CLI output."""

    return f"{value:.2f}"


def analyze_stock(
    ticker: str,
    *,
    period: str = "1y",
    interval: str = "1d",
    export_path: Optional[str] = None,
) -> bool:
    """Perform technical analysis for the given ticker.
    Returns ``True`` if the analysis succeeded; otherwise ``False``.
    """
    
    # --- [데이터 수집] ---
    try:
        data = download_stock_data(ticker, period, interval)
        # [!!! 수정 !!!] 기본적 분석 데이터도 함께 가져옴
        fundamentals = get_fundamental_data(ticker) 
    except ValueError as error:
        print(f"오류: {error}")
        return False
    except Exception as error: 
        print(f"데이터 다운로드 중 예상치 못한 오류 발생: {error}")
        return False

    try:
        enriched_data = compute_indicators(data)
    except Exception as error: 
        print(f"보조 지표 계산 중 오류 발생: {error}")
        return False

    try:
        # --- [기술적 분석 결과 출력] ---
        latest_data = enriched_data.iloc[-1]

        print("---" * 15)
        print(
            f"📊 {ticker} 기술적 분석 결과 (최근 거래일: "
            f"{latest_data.name.strftime('%Y-%m-%d')})"
        )
        print("---" * 15)

        if 'close' in latest_data and pd.notna(latest_data['close']):
             print(f"종가 (수정 종가 기준): ${format_float(latest_data['close'])}")
        else:
            print("종가: (데이터 없음)")

        print("\n--- 📈 기술적 지표 ---")

        # RSI 분석
        rsi_14 = latest_data.get(RSI_COLUMN)
        if pd.notna(rsi_14):
            print(f"RSI (14일): {format_float(rsi_14)}")
            if rsi_14 > 70:
                print("  -> 📈 상태: 과매수 구간 (과열)")
            elif rsi_14 < 30:
                print("  -> 📉 상태: 과매도 구간 (침체)")
            else:
                print("  -> 📊 상태: 중립 구간")
        else:
            print("RSI (14일): 계산되지 않았습니다.")

        # SMA 분석
        sma_20 = latest_data.get(SMA20_COLUMN)
        sma_50 = latest_data.get(SMA50_COLUMN)
        print("\n이동평균선 (SMA):")
        if pd.notna(sma_20):
            print(f"  - 20일선: ${format_float(sma_20)}")
        else:
            print("  - 20일선: 계산되지 않았습니다.")
        if pd.notna(sma_50):
            print(f"  - 50일선: ${format_float(sma_50)}")
        else:
            print("  - 50일선: 계산되지 않았습니다.")
            
        if pd.notna(sma_20) and pd.notna(sma_50):
            if sma_20 > sma_50:
                print("  -> 📈 상태: 단기 골든 크로스 (상승 추세)")
            else:
                print("  -> 📉 상태: 단기 데드 크로스 (하락 추세)")

        # MACD 분석
        macd_line = latest_data.get(MACD_COLUMN)
        signal_line = latest_data.get(MACD_SIGNAL_COLUMN)
        print("\nMACD (12, 26, 9):")
        if pd.notna(macd_line):
            print(f"  - MACD 선: {format_float(macd_line)}")
        else:
            print("  - MACD 선: 계산되지 않았습니다.")
        if pd.notna(signal_line):
            print(f"  - 시그널 선: {format_float(signal_line)}")
        else:
            print("  - 시그널 선: 계산되지 않았습니다.")
            
        if pd.notna(macd_line) and pd.notna(signal_line):
            if macd_line > signal_line:
                print("  -> 📈 상태: 매수 신호 (상승 모멘텀)")
            else:
                print("  -> 📉 상태: 매도 신호 (하락 모멘텀)")
        

        # --- [!!! 신규 추가: 기본적 분석 결과 출력 !!!] ---
        print("\n--- 💼 기본적 분석 지표 (참고용) ---")
        print("   (경고: 산업별, 시장별 기준이 다르므로 단순 참고만 하세요.)")

        if not fundamentals:
            print("  (기본적 분석 데이터를 가져오는 데 실패했습니다.)")
        else:
            # PER 평가
            per = fundamentals.get('per')
            if per and pd.notna(per):
                print(f"\nPER (주가수익비율): {format_float(per)}")
                if per > 0 and per < 15:
                    print("  -> 📊 상태: (전통적) 저평가 구간")
                elif per > 0 and per < 30:
                    print("  -> 📊 상태: (일반적) 적정 수준")
                elif per > 0:
                    print("  -> 📈 상태: 고평가 또는 성장주")
                else:
                    print("  -> 📉 상태: 적자 기업 (수익 없음)")
            else:
                print("\nPER: N/A (데이터 없음)")

            # PBR 평가
            pbr = fundamentals.get('pbr')
            if pbr and pd.notna(pbr):
                print(f"\nPBR (주가순자산비율): {format_float(pbr)}")
                if pbr < 1:
                    print("  -> 📊 상태: 저평가 (자산 가치 대비 주가 낮음)")
                elif pbr < 2:
                    print("  -> 📊 상태: 양호")
                else:
                    print("  -> 📈 상태: 고평가 (자산 가치 대비 주가 높음)")
            else:
                print("\nPBR: N/A (데이터 없음)")
            
            # ROE 평가 (백분율로 변환)
            roe = fundamentals.get('roe')
            if roe and pd.notna(roe):
                print(f"\nROE (자기자본이익률): {roe * 100:.2f}%")
                if roe > 0.15: # 15% 이상
                    print("  -> 📈 상태: 우수 (자본 효율성 매우 높음)")
                elif roe > 0.05: # 5% 이상
                    print("  -> 📊 상태: 양호 (수익 발생 중)")
                else:
                    print("  -> 📉 상태: 비효율 또는 적자")
            else:
                print("\nROE: N/A (데이터 없음)")
        # --- [기본적 분석 끝] ---
        

        all_metrics_valid = (
            pd.notna(sma_20) and pd.notna(sma_50) and
            pd.notna(macd_line) and pd.notna(signal_line) and pd.notna(rsi_14)
        )

        if all_metrics_valid:
            is_sma_bullish = sma_20 > sma_50
            is_macd_bullish = macd_line > signal_line
            is_not_overbought = rsi_14 < 70
            is_not_oversold = rsi_14 > 30

            if is_sma_bullish and is_macd_bullish and is_not_overbought:
                print("\n💡 신호: 긍정적 (강력 매수 고려)")
                print("   (이유: 추세 상승 + 모멘텀 상승 + 과매수 아님)")

            elif (not is_sma_bullish) and (not is_macd_bullish) and is_not_oversold:
                print("\n💡 신호: 부정적 (매도 또는 관망 고려)")
                print("   (이유: 추세 하락 + 모멘텀 하락 + 과매도 아님)")
                
            else:
                print("\n💡 신호: 🚦 중립 (신호 엇갈림)")
                print("   (이유: 지표들이 서로 다른 방향을 가리키고 있습니다.)")

        else:
            print("\n💡 신호: (데이터 부족으로 신호를 생성할 수 없습니다.)")
        # --- [신규 로직 끝] ---

    except Exception as error:
        print(f"분석 중 오류 발생: {error}")
        return False # 분석 중 에러가 나도 실패로 간주

    if export_path:
        try:
            enriched_data.to_csv(export_path)
            print(f"\n데이터가 '{export_path}' 파일로 저장되었습니다.")
        except Exception as error: 
            print(f"CSV 저장 중 오류 발생: {error}")

    return True


def build_argument_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the CLI interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Yahoo Finance 데이터를 이용해 주식의 기술적/기본적 지표를 계산하고 출력합니다."
        )
    )
    parser.add_argument("ticker", help="분석할 주식 코드 (예: AAPL, 005930.KS)")
    parser.add_argument(
        "--period",
        default="1y",
        help="데이터 기간 (예: 1mo, 6mo, 1y, 5y). 기본값은 1y 입니다.",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="데이터 간격 (예: 1d, 1h, 30m). 기본값은 1d 입니다.",
    )
    parser.add_argument(
        "--export",
        help="보조 지표가 포함된 전체 데이터를 CSV 파일로 저장합니다.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point used by the command line script."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    success = analyze_stock(
        args.ticker.upper(), period=args.period, interval=args.interval, export_path=args.export
    )
    return 0 if (success) else 1


if __name__ == "__main__":
    sys.exit(main())