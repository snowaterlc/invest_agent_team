"""
数据获取模块 - 负责从各种数据源获取股票数据
"""
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple

import akshare as ak
from gm.api import (
    set_token,
    get_symbols,
    get_history_symbol,
    get_symbol_infos,
    current,
    stk_get_fundamentals_balance_pt,
    stk_get_fundamentals_income_pt,
    stk_get_finance_deriv_pt,
    stk_get_finance_prime_pt,
    stk_get_daily_valuation_pt,
    get_instrumentinfos,
    get_history_instruments,
    get_previous_trading_date,
    SEC_TYPE_STOCK,
)

from config import config
from utils import (
    logger, retry_on_error, is_cache_valid, safe_get_value,
    normalize_stock_code, extract_symbol_code, get_cache_path,
    calculate_rsi, calculate_ma
)


class StockDataFetcher:
    """股票数据获取器"""
    
    def __init__(self):
        self._init_gm_api()
    
    def _init_gm_api(self):
        """初始化掘金量化API"""
        if config.api.gm_api_token:
            try:
                set_token(config.api.gm_api_token)
                logger.info("掘金量化API已初始化")
            except Exception as e:
                logger.warning(f"掘金量化API初始化失败: {e}")
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表 - 直接使用snowater的get_normal_stocks函数"""
        stock_list = []
        
        # 1. 直接使用snowater的get_normal_stocks函数
        try:
            from snowater.core.data import get_normal_stocks
            
            logger.info("正在使用snowater获取股票列表...")
            
            # 获取今天日期
            today = datetime.now().strftime("%Y-%m-%d")
            
            # 调用get_normal_stocks获取股票列表
            all_stocks, all_stocks_str = get_normal_stocks(date=today, new_days=365)
            
            if all_stocks:
                logger.info(f"snowater获取到 {len(all_stocks)} 只股票")
                
                # 获取股票名称
                try:
                    df_code = get_instrumentinfos(
                        sec_types=SEC_TYPE_STOCK, 
                        fields="symbol, sec_name", 
                        df=True
                    )
                    
                    for symbol in all_stocks:
                        row = df_code[df_code["symbol"] == symbol]
                        name = row.iloc[0].get("sec_name", "") if not row.empty else ""
                        code = symbol.split(".")[-1]
                        stock_list.append({
                            'code': code,
                            'name': name,
                            'symbol': symbol,
                            'ts_code': symbol,
                            'sec_name': name
                        })
                except:
                    # 如果获取名称失败，只返回代码
                    for symbol in all_stocks:
                        code = symbol.split(".")[-1]
                        stock_list.append({
                            'code': code,
                            'name': '',
                            'symbol': symbol,
                            'ts_code': symbol,
                            'sec_name': ''
                        })
                
                if stock_list:
                    return pd.DataFrame(stock_list)
                    
        except Exception as e:
            logger.debug(f"snowater获取股票列表失败: {e}")
        
        # 2. 回退到get_symbols
        try:
            logger.info("正在从掘金量化get_symbols获取股票列表...")
            stock_df = get_symbols(
                sec_type1=1010,
                sec_type2=101001,
                exchanges="SHSE,SZSE",
                skip_suspended=True,
                skip_st=True,
                df=True,
            )
            if stock_df is not None and not stock_df.empty:
                logger.info(f"掘金量化get_symbols获取到 {len(stock_df)} 只股票")
                stock_df["ts_code"] = stock_df["symbol"]
                stock_df["name"] = stock_df["sec_name"]
                stock_df["code"] = stock_df["symbol"].apply(lambda x: x.split(".")[-1])
                return stock_df
        except Exception as e:
            logger.debug(f"掘金量化get_symbols获取股票列表失败: {e}")
        
        # 3. 回退到akshare获取股票列表
        import concurrent.futures
        
        logger.info("正在从akshare获取股票列表...")
        
        def fetch_with_timeout(func, timeout=30):
            """带超时的函数执行"""
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(func)
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"获取股票列表超时({timeout}秒)")
                    return None
        
        # 尝试获取深交所股票
        def get_sz_stocks():
            try:
                return ak.stock_info_sz_name_code(symbol="A股列表")
            except:
                return None
        
        # 尝试获取上交所股票
        def get_sh_stocks():
            try:
                return ak.stock_info_sh_name_code()
            except:
                return None
        
        # 并行获取两个交易所的股票
        sz_df = fetch_with_timeout(get_sz_stocks, timeout=30)
        sh_df = fetch_with_timeout(get_sh_stocks, timeout=30)
        
        # 处理深交所股票
        if sz_df is not None and not sz_df.empty:
            try:
                for _, row in sz_df.iterrows():
                    code = str(row.get('A股代码', '')).strip()
                    name = str(row.get('A股简称', '')).strip()
                    if code and name and not any(x in name for x in ['ST', '退', '*']):
                        stock_list.append({
                            'code': code,
                            'name': name,
                            'symbol': f'SZSE.{code}',
                            'ts_code': f'SZSE.{code}',
                            'sec_name': name
                        })
            except Exception as e:
                logger.debug(f"处理深交所股票数据失败: {e}")
        
        # 处理上交所股票
        if sh_df is not None and not sh_df.empty:
            try:
                for _, row in sh_df.iterrows():
                    code = str(row.get('证券代码', '')).strip()
                    name = str(row.get('证券简称', '')).strip()
                    if code and name and not any(x in name for x in ['ST', '退', '*']):
                        stock_list.append({
                            'code': code,
                            'name': name,
                            'symbol': f'SHSE.{code}',
                            'ts_code': f'SHSE.{code}',
                            'sec_name': name
                        })
            except Exception as e:
                logger.debug(f"处理上交所股票数据失败: {e}")
        
        if not stock_list:
            logger.error("无法获取股票列表")
            return pd.DataFrame()
        
        logger.info(f"成功获取 {len(stock_list)} 只股票")
        return pd.DataFrame(stock_list)
    
    def get_fundamental_data(self, ts_code: str) -> pd.DataFrame:
        """获取基本面数据"""
        gm_symbol = normalize_stock_code(ts_code)
        fina_data = pd.DataFrame()
        
        try:
            # 尝试掘金量化API
            balance_data = stk_get_fundamentals_balance_pt(
                symbols=gm_symbol,
                date=datetime.now().strftime("%Y-%m-%d"),
                fields="ttl_ast,mny_cptl,ttl_cur_ast,ttl_ncur_ast,ttl_liab,ttl_eqy",
            )
            income_data = stk_get_fundamentals_income_pt(
                symbols=gm_symbol,
                date=datetime.now().strftime("%Y-%m-%d"),
                fields="inc_oper,net_prof,oper_prof,ttl_prof,biz_tax_sur,exp_sell,exp_adm,exp_rd,exp_fin",
            )
            indicator_data = stk_get_finance_deriv_pt(
                symbols=gm_symbol,
                date=datetime.now().strftime("%Y-%m-%d"),
                fields="roe,roe_weight,roe_avg,roe_cut",
            )
            prime_data = stk_get_finance_prime_pt(
                symbols=gm_symbol,
                date=datetime.now().strftime("%Y-%m-%d"),
                fields="eps_basic,eps_dil,bps_pcom_ps,net_prof_pcom_yoy",
            )
            valuation_data = stk_get_daily_valuation_pt(
                symbols=gm_symbol,
                fields="pb_lyr,pe_ttm,ps_ttm,pcf_ttm_oper,dy_ttm",
            )
            
            # 合并数据
            for data, prefix in [
                (balance_data, "bal_"), (income_data, "inc_"),
                (indicator_data, "ind_"), (prime_data, "prime_"),
                (valuation_data, "val_")
            ]:
                if isinstance(data, pd.DataFrame) and not data.empty:
                    data = data.add_prefix(prefix)
                    fina_data = pd.concat([fina_data, data], axis=1) if not fina_data.empty else data
            
            if not fina_data.empty:
                return fina_data
                
        except Exception as e:
            logger.debug(f"掘金量化基本面数据获取失败: {e}")
        
        # 回退到akshare - 使用多个备选接口
        symbol_clean = extract_symbol_code(ts_code)
        
        # 尝试获取主要财务指标
        try:
            fina_data = ak.stock_financial_analysis_indicator(symbol=symbol_clean)
            if not fina_data.empty:
                fina_data["ts_code"] = ts_code
                return fina_data
        except:
            pass
        
        # 尝试获取财务摘要
        try:
            fina_data = ak.stock_financial_abstract_ths(symbol=symbol_clean)
            if not fina_data.empty:
                fina_data["ts_code"] = ts_code
                return fina_data
        except:
            pass
        
        # 如果所有接口都失败，返回空DataFrame
        logger.debug(f"无法获取 {ts_code} 的基本面数据")
        return pd.DataFrame()
    
    def get_technical_data(self, ts_code: str) -> Tuple[pd.DataFrame, Optional[float]]:
        """获取技术面数据，返回(日数据, 当前价格)"""
        gm_symbol = normalize_stock_code(ts_code)
        daily = pd.DataFrame()
        current_price = None
        
        try:
            # 尝试掘金量化API
            daily = get_history_symbol(
                symbol=gm_symbol,
                start_date=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                end_date=datetime.now().strftime("%Y-%m-%d"),
                df=True,
            )
            daily = daily.rename(columns={"trade_date": "date"})
            
            # 获取实时价格
            try:
                current_data = current(symbols=gm_symbol)
                if current_data and len(current_data) > 0:
                    tick = current_data[0]
                    current_price = tick.price if hasattr(tick, 'price') else tick.get('price')
            except:
                pass
            
            if not daily.empty:
                return daily, current_price
                
        except Exception as e:
            logger.debug(f"掘金量化技术面数据获取失败: {e}")
        
        # 回退到akshare
        try:
            symbol_clean = extract_symbol_code(ts_code)
            daily = ak.stock_zh_a_hist(
                symbol=symbol_clean,
                period="daily",
                start_date=(datetime.now() - timedelta(days=60)).strftime("%Y%m%d"),
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="",
            )
            if not daily.empty:
                daily = daily.rename(columns={
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low", "成交量": "volume",
                })
                if "close" in daily.columns:
                    current_price = daily["close"].iloc[-1]
                return daily, current_price
        except Exception as e:
            logger.warning(f"akshare技术面数据获取失败: {e}")
        
        return pd.DataFrame(), None
    
    def get_stock_name(self, ts_code: str) -> Optional[str]:
        """获取股票名称"""
        gm_symbol = normalize_stock_code(ts_code)
        
        try:
            stock_info = get_symbol_infos(sec_type1=1010, symbols=gm_symbol)
            if stock_info and len(stock_info) > 0 and "sec_name" in stock_info[0]:
                return stock_info[0]["sec_name"]
        except:
            pass
        
        try:
            symbol_clean = extract_symbol_code(ts_code)
            stock_info = ak.stock_info_a_code_name()
            if isinstance(stock_info, pd.DataFrame) and "code" in stock_info.columns:
                filtered = stock_info[stock_info["code"] == symbol_clean]
                if not filtered.empty:
                    return filtered["name"].iloc[0]
        except:
            pass
        
        return None
    
    def get_all_data(self, ts_code: str, use_cache: bool = True) -> Dict:
        """获取股票的所有数据"""
        # 检查缓存
        cache_path = get_cache_path(ts_code)
        if use_cache and is_cache_valid(cache_path):
            try:
                import json
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        # 获取数据
        fina_data = self.get_fundamental_data(ts_code)
        daily, current_price = self.get_technical_data(ts_code)
        stock_name = self.get_stock_name(ts_code)
        
        # 计算技术指标
        if not daily.empty and "close" in daily.columns:
            daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
            daily["ma5"] = calculate_ma(daily["close"], 5)
            daily["ma20"] = calculate_ma(daily["close"], 20)
            daily["rsi"] = calculate_rsi(daily["close"])
        
        # 构建结果
        result = {
            "ts_code": ts_code,
            "name": stock_name,
            "fundamental": fina_data.head(1).to_dict("records")[0] if not fina_data.empty else {},
            "technical": daily.to_dict("records") if not daily.empty else [],
            "current_price": current_price,
        }
        
        # 保存缓存
        if use_cache:
            try:
                import json
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, default=str)
            except:
                pass
        
        return result


# 全局数据获取器实例
data_fetcher = StockDataFetcher()
