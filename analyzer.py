"""
分析模块 - 负责股票筛选和分析逻辑
"""
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

from config import config
from utils import logger, format_market_cap
from data_fetcher import data_fetcher


class StockAnalyzer:
    """股票分析器"""
    
    def __init__(self):
        self.filter_config = config.filter
    
    def filter_by_fundamentals(self, stock_list: pd.DataFrame) -> pd.DataFrame:
        """根据基本面筛选股票"""
        if stock_list.empty:
            return stock_list
        
        filtered_stocks = []
        
        for _, row in stock_list.iterrows():
            ts_code = row.get("ts_code") or row.get("symbol")
            if not ts_code:
                continue
            
            try:
                fina_data = data_fetcher.get_fundamental_data(ts_code)
                if fina_data.empty:
                    continue
                
                # 获取关键指标
                roe = self._get_value(fina_data, ["ind_roe", "roe"], 0)
                profit_growth = self._get_value(fina_data, ["prime_net_prof_pcom_yoy", "net_prof_pcom_yoy", "profit_growth_rate"], 0)
                debt_ratio = self._get_value(fina_data, ["ind_ast_liab_rate", "ast_liab_rate", "debt_to_assets_ratio"], 100)
                market_cap = self._get_value(fina_data, ["val_mkt_cap", "mkt_cap", "total_market_cap"], 0)
                
                # 应用筛选条件
                if market_cap > self.filter_config.max_market_cap:
                    continue
                if roe < self.filter_config.min_roe:
                    continue
                if profit_growth < self.filter_config.min_profit_growth:
                    continue
                if debt_ratio > self.filter_config.max_debt_ratio:
                    continue
                
                row["roe"] = roe
                row["profit_growth"] = profit_growth
                row["debt_ratio"] = debt_ratio
                row["market_cap"] = market_cap
                filtered_stocks.append(row)
                
            except Exception as e:
                logger.debug(f"筛选股票 {ts_code} 失败: {e}")
                continue
        
        if not filtered_stocks:
            return pd.DataFrame()
        
        result = pd.DataFrame(filtered_stocks)
        # 按ROE排序，取前N只
        result = result.sort_values("roe", ascending=False).head(self.filter_config.max_stocks)
        return result
    
    def _get_value(self, df: pd.DataFrame, columns: List[str], default: float = 0) -> float:
        """从DataFrame中获取值，尝试多个列名"""
        for col in columns:
            if col in df.columns:
                val = df[col].iloc[0]
                if pd.notna(val):
                    return float(val)
        return default
    
    def analyze_technical(self, ts_code: str) -> Dict:
        """技术分析"""
        daily, current_price = data_fetcher.get_technical_data(ts_code)
        
        if daily.empty:
            return {}
        
        latest = daily.iloc[-1] if not daily.empty else None
        
        # 计算技术指标
        analysis = {
            "current_price": current_price,
            "support_price": latest.get("low") if latest is not None else None,
            "resistance_price": latest.get("high") if latest is not None else None,
            "ma20_position": None,
            "volume_trend": None,
            "rsi": None,
        }
        
        if latest is not None and "close" in latest and "ma20" in latest:
            if pd.notna(latest["close"]) and pd.notna(latest["ma20"]):
                analysis["ma20_position"] = "above" if latest["close"] > latest["ma20"] else "below"
        
        if len(daily) >= 2 and latest is not None and "volume" in latest:
            prev_volume = daily.iloc[-2].get("volume")
            if prev_volume is not None and pd.notna(latest["volume"]) and pd.notna(prev_volume):
                analysis["volume_trend"] = "up" if latest["volume"] > prev_volume else "down"
        
        if "rsi" in daily.columns and not daily.empty:
            rsi_val = daily["rsi"].iloc[-1]
            if pd.notna(rsi_val):
                analysis["rsi"] = float(rsi_val)
        
        return analysis
    
    def generate_trading_signal(self, ts_code: str) -> Dict:
        """生成交易信号"""
        # 获取基本面数据
        fina_data = data_fetcher.get_fundamental_data(ts_code)
        
        # 获取技术分析
        tech_analysis = self.analyze_technical(ts_code)
        
        # 综合评分
        score = 0
        signals = []
        
        # 基本面评分
        if not fina_data.empty:
            roe = self._get_value(fina_data, ["ind_roe", "roe"], 0)
            if roe > 0.15:
                score += 30
                signals.append("ROE优秀")
            elif roe > 0.12:
                score += 20
                signals.append("ROE良好")
            
            profit_growth = self._get_value(fina_data, ["prime_net_prof_pcom_yoy", "profit_growth_rate"], 0)
            if profit_growth > 0.20:
                score += 25
                signals.append("利润增长强劲")
            elif profit_growth > 0.15:
                score += 15
                signals.append("利润增长稳定")
        
        # 技术面评分
        if tech_analysis.get("ma20_position") == "above":
            score += 20
            signals.append("站上MA20")
        
        if tech_analysis.get("volume_trend") == "up":
            score += 15
            signals.append("放量上涨")
        
        rsi = tech_analysis.get("rsi")
        if rsi is not None:
            if 30 < rsi < 70:
                score += 10
                signals.append("RSI正常")
            elif rsi <= 30:
                score += 5
                signals.append("RSI超卖")
        
        # 确定交易建议
        if score >= 70:
            recommendation = "强烈买入"
        elif score >= 50:
            recommendation = "买入"
        elif score >= 30:
            recommendation = "观望"
        else:
            recommendation = "回避"
        
        return {
            "ts_code": ts_code,
            "score": score,
            "recommendation": recommendation,
            "signals": signals,
            "technical": tech_analysis,
            "fundamental": fina_data.head(1).to_dict("records")[0] if not fina_data.empty else {},
        }
    
    def get_top_stocks(self, n: int = 10) -> List[Dict]:
        """获取推荐股票列表"""
        # 获取股票列表
        stock_list = data_fetcher.get_stock_list()
        if stock_list.empty:
            logger.error("获取股票列表失败")
            return []
        
        # 基本面筛选
        filtered = self.filter_by_fundamentals(stock_list)
        if filtered.empty:
            logger.warning("没有股票通过基本面筛选")
            return []
        
        # 生成分析报告
        results = []
        for _, row in filtered.iterrows():
            ts_code = row.get("ts_code") or row.get("symbol")
            analysis = self.generate_trading_signal(ts_code)
            if analysis:
                analysis["name"] = row.get("name") or row.get("sec_name")
                analysis["market_cap"] = row.get("market_cap", 0)
                results.append(analysis)
        
        # 按评分排序
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:n]


# 全局分析器实例
analyzer = StockAnalyzer()
