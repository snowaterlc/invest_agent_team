"""
工具函数模块 - 通用工具函数
"""
import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Dict, List, Any, Union

from config import config


# 设置日志
logger = logging.getLogger(__name__)


class APIError(Exception):
    """API调用错误"""
    pass


class DataFetchError(Exception):
    """数据获取错误"""
    pass


class CacheError(Exception):
    """缓存错误"""
    pass


def retry_on_error(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    重试装饰器
    :param max_retries: 最大重试次数
    :param delay: 初始延迟时间（秒）
    :param backoff: 延迟时间倍数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import requests
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.Timeout, requests.ConnectionError) as e:
                    last_exception = e
                    logger.warning(f"{func.__name__} 第{attempt + 1}次尝试失败: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
                except Exception as e:
                    logger.error(f"{func.__name__} 发生不可重试错误: {e}")
                    raise
            
            logger.error(f"{func.__name__} 重试{max_retries}次后仍失败")
            raise APIError(f"重试{max_retries}次后仍失败: {last_exception}")
        return wrapper
    return decorator


def is_cache_valid(cache_path: str) -> bool:
    """检查缓存是否有效（未过期）"""
    if not os.path.exists(cache_path):
        return False
    file_time = os.path.getmtime(cache_path)
    return (time.time() - file_time) < config.cache.expire_seconds


def safe_get_value(df, column: str, default=None):
    """安全获取DataFrame中的值"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return default
    if column not in df.columns:
        return default
    value = df[column].iloc[0]
    return value if pd.notna(value) else default


def safe_get_values(df, columns: List[str], defaults: Any = None) -> Dict:
    """安全批量获取DataFrame中的值"""
    result = {}
    defaults_list = [defaults] * len(columns) if not isinstance(defaults, list) else defaults
    
    for i, column in enumerate(columns):
        result[column] = safe_get_value(df, column, defaults_list[i] if i < len(defaults_list) else None)
    return result


def convert_ts_code(ts_code: str) -> str:
    """将标准股票代码转换为掘金格式"""
    if not ts_code:
        return ""
    ts_code = ts_code.strip().upper()
    if "." not in ts_code:
        if ts_code.startswith("6"):
            return f"SHSE.{ts_code}"
        elif ts_code.startswith(("0", "3")):
            return f"SZSE.{ts_code}"
    return ts_code


def normalize_stock_code(ts_code: str) -> str:
    """标准化股票代码格式"""
    if not ts_code:
        return ""
    ts_code = str(ts_code).strip().upper()
    import re
    ts_code = re.sub(r"\.SH|\.SZ", "", ts_code)
    ts_code = ts_code.zfill(6)
    if ts_code.startswith("6"):
        return f"SHSE.{ts_code}"
    elif ts_code.startswith(("0", "3")):
        return f"SZSE.{ts_code}"
    return ts_code


def extract_symbol_code(ts_code: str) -> str:
    """从掘金格式提取纯数字代码"""
    if not ts_code:
        return ""
    return ts_code.split(".")[-1]


def format_market_cap(value: float) -> str:
    """格式化市值显示"""
    if value >= 1e8:
        return f"{value/1e8:.2f}亿"
    elif value >= 1e4:
        return f"{value/1e4:.2f}万"
    return f"{value:.2f}"


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """计算RSI指标"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_ma(prices: pd.Series, window: int) -> pd.Series:
    """计算移动平均线"""
    return prices.rolling(window=window, min_periods=1).mean()


def get_cache_path(ts_code: str) -> str:
    """获取缓存文件路径"""
    cache_file = ts_code.replace(".", "_") + ".json"
    return os.path.join(config.cache.cache_dir, cache_file)
