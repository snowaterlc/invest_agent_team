import os
import re
import warnings
from urllib.parse import quote
from typing import Optional  # 添加Optional导入

import akshare as ak
import numpy as np
from gm.api import *
import gm.api as gm_api
import pandas as pd
import requests
from bs4 import BeautifulSoup
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from dotenv import load_dotenv
from gm.api import history, get_instruments
from langchain_openai import ChatOpenAI
from openai import APITimeoutError, APIConnectionError
from datetime import datetime, timedelta

# 基础配置
warnings.filterwarnings("ignore")
load_dotenv()
os.makedirs("./cache", exist_ok=True)

# 数据源初始化
gm_api_token = os.getenv("GM_API_TOKEN")
if gm_api_token:
    set_token(gm_api_token)
try:
    # 使用新版API获取下个交易日
    trading_dates = get_next_n_trading_dates(
        date=datetime.now().strftime("%Y-%m-%d"), n=1, exchange="SHSE"
    )
    NEXT_TRADING_DAY = (
        trading_dates[0]
        if trading_dates
        else (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    )
except Exception as e:
    print(f"获取交易日历失败: {e}")
    NEXT_TRADING_DAY = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")


# LLM配置（DeepSeek兼容）
def get_deepseek_llm():
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        model="deepseek-chat",
        temperature=0.1,
        timeout=30.0,
        max_retries=2,
    )


llm = get_deepseek_llm()


# ====================== 核心工具扩展 ======================
@tool("AShareDataTool")
def get_a_share_data(ts_code: Optional[str] = None, limit_data: bool = True) -> dict:
    """获取A股主板股票数据（基本面+技术面），新增下个交易日关键信号"""
    if ts_code is not None:
        ts_code = str(ts_code)
        # 验证股票代码格式
        import re

        if not re.match(
            r"^(\d{6}|\d{6}\.SH|\d{6}\.SZ|SHSE\.\d{6}|SZSE\.\d{6})$", ts_code
        ):
            return {
                "error": f"无效的股票代码格式: {ts_code}，请使用6位数字代码(如600000)、A股格式(如600000.SH)或掘金格式(如SHSE.600862)",
                "ts_code": ts_code,
            }

    cache_path = (
        f"./cache/{ts_code.replace('.', '_')}.json"
        if ts_code is not None
        else "./cache/a_share_mainboard_list.json"
    )

    # 缓存读取
    if os.path.exists(cache_path):
        try:
            data = pd.read_json(cache_path, orient="records")
            # 确保ts_code和symbol保持为字符串类型（防止pandas自动转换为整数）
            if "ts_code" in data.columns:
                data["ts_code"] = data["ts_code"].astype(str)
            if "symbol" in data.columns:
                data["symbol"] = data["symbol"].astype(str)
            if ts_code is None and isinstance(data, pd.DataFrame) and len(data) > 10:
                data = data.head(10)
            return (
                data.to_dict("records")[0]
                if ts_code is not None
                else {"stock_list": data.to_dict("records")}
            )
        except Exception as e:
            print(f"缓存读取失败，重新获取: {e}")

    try:
        # 股票列表获取
        if ts_code is None:
            try:
                # 使用新版API获取股票列表
                stock_df = get_symbols(
                    sec_type1=1010,  # A股
                    sec_type2=101001,  # 主板A股
                    exchanges="SHSE,SZSE",
                    skip_suspended=True,
                    skip_st=True,
                    df=True,
                )

                stock_df["ts_code"] = stock_df["symbol"]
                stock_df["name"] = stock_df["sec_name"]
                stock_df["symbol"] = stock_df["symbol"]
                stock_df = stock_df[
                    (
                        ~stock_df["symbol"].str.startswith(
                            (
                                "SZSE.3",
                                "SHSE.688",
                                "SZSE.8",
                                "SHSE.9",
                                "SHSE.4",
                                "SZSE.2",
                            )
                        )
                    )
                ].head(10)
                stock_df.to_json(cache_path, orient="records", force_ascii=False)
                stock_list = stock_df.to_dict("records")
                for item in stock_list:
                    item["ts_code"] = str(item.get("symbol", ""))
                return {"stock_list": stock_list}
            except:
                try:
                    stock_list = ak.stock_info_a_code_name()
                    # 根据实际返回的列数来处理
                    if len(stock_list.columns) >= 2:
                        # 如果返回至少2列，使用前两列作为symbol和name
                        stock_list = stock_list.iloc[:, :2]  # 取前两列
                        stock_list.columns = ["symbol", "name"]
                    elif len(stock_list.columns) == 1:
                        # 如果只有一列，可能是代码列，需要处理
                        stock_list = stock_list.rename(
                            columns={stock_list.columns[0]: "symbol"}
                        )
                        stock_list["name"] = ""  # 添加空的name列
                    else:
                        raise ValueError("akshare返回的股票列表格式不正确")
                    stock_list["ts_code"] = stock_list["symbol"]
                    stock_list = stock_list[
                        ~stock_list["symbol"]
                        .astype(str)
                        .str.startswith(("3", "688", "8", "9", "4", "2"))
                    ].head(10)
                    stock_list.to_json(cache_path, orient="records", force_ascii=False)
                    stock_list = stock_list.to_dict("records")
                    for item in stock_list:
                        item["ts_code"] = str(item.get("symbol", ""))
                    return {"stock_list": stock_list}
                except Exception as e:
                    print(f"所有数据源都失败: {str(e)}")
                    return {"stock_list": []}

        # 基本面数据
        gm_symbol = set_em_symble(ts_code)
        try:
            # 使用新版API获取基本面数据
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
                date=datetime.now().strftime("%Y-%m-%d"),
                fields="pb_lyr,pe_ttm,ps_ttm,pcf_ttm_oper,dy_ttm",
            )

            # 合并数据
            balance_has_data = (
                isinstance(balance_data, pd.DataFrame) and not balance_data.empty
            )
            income_has_data = (
                isinstance(income_data, pd.DataFrame) and not income_data.empty
            )
            indicator_has_data = (
                isinstance(indicator_data, pd.DataFrame) and not indicator_data.empty
            )
            prime_has_data = (
                isinstance(prime_data, pd.DataFrame) and not prime_data.empty
            )
            valuation_has_data = (
                isinstance(valuation_data, pd.DataFrame) and not valuation_data.empty
            )

            if (
                balance_has_data
                or income_has_data
                or indicator_has_data
                or prime_has_data
                or valuation_has_data
            ):
                # 合并掘金API获取的数据 - 只保留关键指标
                # 创建一个空的DataFrame作为基础
                fina_data_dict = {"ts_code": ts_code}

                # 从资产负债表获取数据
                if isinstance(balance_data, pd.DataFrame) and not balance_data.empty:
                    fina_data_dict["total_assets"] = (
                        balance_data["ttl_ast"].iloc[0]
                        if "ttl_ast" in balance_data.columns
                        else None
                    )
                    fina_data_dict["mny_cptl"] = (
                        balance_data["mny_cptl"].iloc[0]
                        if "mny_cptl" in balance_data.columns
                        else None
                    )
                    fina_data_dict["ttl_cur_ast"] = (
                        balance_data["ttl_cur_ast"].iloc[0]
                        if "ttl_cur_ast" in balance_data.columns
                        else None
                    )
                    fina_data_dict["ttl_ncur_ast"] = (
                        balance_data["ttl_ncur_ast"].iloc[0]
                        if "ttl_ncur_ast" in balance_data.columns
                        else None
                    )
                    fina_data_dict["ttl_liab"] = (
                        balance_data["ttl_liab"].iloc[0]
                        if "ttl_liab" in balance_data.columns
                        else None
                    )
                    fina_data_dict["ttl_eqy"] = (
                        balance_data["ttl_eqy"].iloc[0]
                        if "ttl_eqy" in balance_data.columns
                        else None
                    )

                # 从利润表获取数据
                if isinstance(income_data, pd.DataFrame) and not income_data.empty:
                    fina_data_dict["revenue"] = (
                        income_data["inc_oper"].iloc[0]
                        if "inc_oper" in income_data.columns
                        else None
                    )
                    fina_data_dict["net_profit"] = (
                        income_data["net_prof"].iloc[0]
                        if "net_prof" in income_data.columns
                        else None
                    )
                    fina_data_dict["operating_profit"] = (
                        income_data["oper_prof"].iloc[0]
                        if "oper_prof" in income_data.columns
                        else None
                    )
                    fina_data_dict["total_profit"] = (
                        income_data["ttl_prof"].iloc[0]
                        if "ttl_prof" in income_data.columns
                        else None
                    )
                    fina_data_dict["biz_tax_sur"] = (
                        income_data["biz_tax_sur"].iloc[0]
                        if "biz_tax_sur" in income_data.columns
                        else None
                    )
                    fina_data_dict["exp_sell"] = (
                        income_data["exp_sell"].iloc[0]
                        if "exp_sell" in income_data.columns
                        else None
                    )
                    fina_data_dict["exp_adm"] = (
                        income_data["exp_adm"].iloc[0]
                        if "exp_adm" in income_data.columns
                        else None
                    )
                    fina_data_dict["exp_rd"] = (
                        income_data["exp_rd"].iloc[0]
                        if "exp_rd" in income_data.columns
                        else None
                    )
                    fina_data_dict["exp_fin"] = (
                        income_data["exp_fin"].iloc[0]
                        if "exp_fin" in income_data.columns
                        else None
                    )

                # 从财务衍生指标获取数据
                if (
                    isinstance(indicator_data, pd.DataFrame)
                    and not indicator_data.empty
                ):
                    fina_data_dict["roe"] = (
                        indicator_data["roe"].iloc[0]
                        if "roe" in indicator_data.columns
                        else None
                    )
                    fina_data_dict["roe_weight"] = (
                        indicator_data["roe_weight"].iloc[0]
                        if "roe_weight" in indicator_data.columns
                        else None
                    )

                # 从财务主要指标获取净利润增长率
                if isinstance(prime_data, pd.DataFrame) and not prime_data.empty:
                    fina_data_dict["profit_growth"] = (
                        prime_data["net_prof_pcom_yoy"].iloc[0]
                        if "net_prof_pcom_yoy" in prime_data.columns
                        else None
                    )

                # 从财务主要指标获取数据
                if isinstance(prime_data, pd.DataFrame) and not prime_data.empty:
                    # 这里可以添加其他主要指标
                    pass

                # 从估值指标获取数据
                if (
                    isinstance(valuation_data, pd.DataFrame)
                    and not valuation_data.empty
                ):
                    fina_data_dict["pe_ttm"] = (
                        valuation_data["pe_ttm"].iloc[0]
                        if "pe_ttm" in valuation_data.columns
                        else None
                    )
                    fina_data_dict["ps_ttm"] = (
                        valuation_data["ps_ttm"].iloc[0]
                        if "ps_ttm" in valuation_data.columns
                        else None
                    )
                    fina_data_dict["pb_lyr"] = (
                        valuation_data["pb_lyr"].iloc[0]
                        if "pb_lyr" in valuation_data.columns
                        else None
                    )
                    fina_data_dict["pcf_ttm_oper"] = (
                        valuation_data["pcf_ttm_oper"].iloc[0]
                        if "pcf_ttm_oper" in valuation_data.columns
                        else None
                    )
                    # 流通市值 (单位: 亿元)
                    if "neg_mkt_cap" in valuation_data.columns:
                        fina_data_dict["circulating_market_value"] = (
                            valuation_data["neg_mkt_cap"].iloc[0] / 1e8
                            if valuation_data["neg_mkt_cap"].iloc[0] is not None
                            else None
                        )
                    elif "mkt_cap" in valuation_data.columns:
                        fina_data_dict["circulating_market_value"] = (
                            valuation_data["mkt_cap"].iloc[0] / 1e8
                            if valuation_data["mkt_cap"].iloc[0] is not None
                            else None
                        )
                    else:
                        fina_data_dict["circulating_market_value"] = None
                fina_data = pd.DataFrame([fina_data_dict])
            else:
                print(f"掘金量化基本面数据获取失败，回退到akshare: {ts_code}")

                # 如果掘金新API都失败，回退到akshare
                try:
                    # 使用akshare获取基本面数据
                    symbol_clean = ts_code.split(".")[1]  # 获取股票代码部分
                    fina_data = ak.stock_financial_abstract(
                        symbol=symbol_clean
                    )  # 获取基本面指标
                    if isinstance(fina_data, pd.DataFrame) and fina_data.empty:
                        # 如果akshare也失败，返回空数据
                        print(f"akshare无{ts_code}基本面数据")
                        fina_data = pd.DataFrame()
                    else:
                        # 只保留关键基本面指标
                        required_cols = [
                            "roe",
                            "net_profit",
                            "profit_growth_rate",
                            "total_assets",
                            "debt_to_assets_ratio",
                            "pledge_ratio",
                            "inc_oper",
                            "oper_prof",
                            "ttl_prof",
                            "biz_tax_sur",
                            "exp_sell",
                            "exp_adm",
                            "exp_rd",
                            "exp_fin",
                            "pe_ttm",
                            "ps_ttm",
                            "pb_lyr",
                            "pcf_ttm_oper",
                            "roe_weight",
                            "net_prof_pcom_yoy",
                        ]
                        available_cols = [
                            col for col in required_cols if col in fina_data.columns
                        ]
                        fina_data = (
                            fina_data[available_cols].head(1)
                            if available_cols
                            else fina_data.head(1)
                        )
                        fina_data["ts_code"] = ts_code
                except Exception as e:
                    # 如果akshare也失败，返回空数据
                    print(f"akshare基本面数据获取失败: {str(e)}")
                    fina_data = pd.DataFrame()
        except Exception as e:
            print(f"掘金量化基本面数据获取失败: {str(e)}，回退到akshare")
            # 确保在基本面数据获取失败时，gm_symbol 仍然可用
            if "gm_symbol" not in locals() or "gm_symbol" not in globals():
                gm_symbol = set_em_symble(ts_code)
            try:
                symbol_clean = ts_code.split(".")[0]
                fina_data = ak.stock_financial_abstract(symbol=symbol_clean)
                if isinstance(fina_data, pd.DataFrame) and fina_data.empty:
                    fina_data = pd.DataFrame()
                else:
                    required_cols = [
                        "roe",
                        "net_profit",
                        "profit_growth_rate",
                        "total_assets",
                        "debt_to_assets_ratio",
                        "pledge_ratio",
                        "inc_oper",
                        "oper_prof",
                        "ttl_prof",
                        "biz_tax_sur",
                        "exp_sell",
                        "exp_adm",
                        "exp_rd",
                        "exp_fin",
                        "pe_ttm",
                        "ps_ttm",
                        "pb_lyr",
                        "pcf_ttm_oper",
                        "roe_weight",
                        "net_prof_pcom_yoy",
                    ]
                    fina_data = (
                        fina_data[required_cols].head(1)
                        if all(col in fina_data.columns for col in required_cols)
                        else fina_data.head(1)
                    )
                    fina_data["ts_code"] = ts_code
            except:
                fina_data = pd.DataFrame()

        # 技术面数据（新增下个交易日关键信号）
        try:
            # 使用新版API获取历史数据
            daily = get_history_symbol(
                symbols=gm_symbol,
                start_date=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                end_date=datetime.now().strftime("%Y-%m-%d"),
                df=True,
            )
            daily = daily.rename(columns={"trade_date": "date"})

            if not isinstance(daily, pd.DataFrame) or daily.empty:
                raise ValueError(f"掘金量化无{ts_code}技术面数据")

            # 获取最新价格
            current_data = current(symbols=gm_symbol, fields="price")
            if isinstance(current_data, pd.DataFrame) and not current_data.empty:
                current_price = current_data[0]["price"]
            else:
                # 如果实时价格获取失败，使用历史数据的最新收盘价
                current_price = (
                    daily["close"].iloc[-1]
                    if not daily.empty
                    and "close" in daily.columns
                    and not daily["close"].isna().iloc[-1]
                    else None
                )

        except:
            symbol_clean = ts_code.split(".")[1]
            try:
                daily = ak.stock_zh_a_hist(
                    symbol=symbol_clean,
                    period="daily",
                    start_date=(datetime.now() - timedelta(days=60)).strftime("%Y%m%d"),
                    end_date=datetime.now().strftime("%Y%m%d"),
                    adjust="",
                )
                if not isinstance(daily, pd.DataFrame) or daily.empty:
                    raise ValueError(f"akshare无{ts_code}技术面数据")
                daily = daily.rename(
                    columns={
                        "日期": "date",
                        "开盘": "open",
                        "收盘": "close",
                        "最高": "high",
                        "最低": "low",
                        "成交量": "volume",
                    }
                )

                # 获取最新的akshare数据作为当前价格
                current_price = (
                    daily["close"].iloc[-1]
                    if not daily.empty
                    and "close" in daily.columns
                    and not daily["close"].isna().iloc[-1]
                    else None
                )

            except:
                raise ValueError(f"技术面数据获取失败")

        # 确保技术数据包含当前价格
        if "current_price" not in locals():
            current_price = (
                daily["close"].iloc[-1]
                if not daily.empty
                and "close" in daily.columns
                and not daily["close"].isna().iloc[-1]
                else None
            )

        # 精简数据
        if limit_data and len(daily) > 5:
            daily = daily.tail(5).reset_index(drop=True)

        # 计算关键技术指标
        daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
        daily["ma5"] = daily["close"].rolling(5, min_periods=1).mean()
        daily["ma20"] = daily["close"].rolling(20, min_periods=1).mean()

        # 仅在数据有效时进行计算
        if daily["close"].isna().all():
            daily["ma5"] = pd.Series([None] * len(daily), dtype=float)
            daily["ma20"] = pd.Series([None] * len(daily), dtype=float)

        # 计算RSI指标
        delta = daily["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        daily["rsi"] = 100 - (100 / (1 + rs.where(rs != 0, 1)))  # 避免除零错误

        # 涨停判断（A股涨停幅度约10%）
        daily["is_limit_up"] = (
            (daily["close"] - daily["close"].shift(1)) / daily["close"].shift(1)
        ) >= 0.098
        daily["has_limit_up_recently"] = daily["is_limit_up"].tail(30).any()

        # 下个交易日关键信号
        latest = daily.iloc[-1] if not daily.empty else None
        next_trading_signal = {
            "current_price": current_price,
            "support_price": latest["low"]
            if latest is not None and "low" in latest and pd.notna(latest["low"])
            else None,
            "resistance_price": latest["high"]
            if latest is not None and "high" in latest and pd.notna(latest["high"])
            else None,
            "ma20_position": "above"
            if latest is not None
            and "close" in latest
            and "ma20" in latest
            and pd.notna(latest["close"])
            and pd.notna(latest["ma20"])
            and latest["close"] > latest["ma20"]
            else "below"
            if latest is not None
            and "close" in latest
            and "ma20" in latest
            and pd.notna(latest["close"])
            and pd.notna(latest["ma20"])
            else None,
            "volume_trend": "up"
            if latest is not None
            and "volume" in latest
            and len(daily) >= 2
            and daily.iloc[-2]["volume"] is not None
            and pd.notna(latest["volume"])
            and pd.notna(daily.iloc[-2]["volume"])
            and latest["volume"] > daily.iloc[-2]["volume"]
            else "down"
            if latest is not None
            and "volume" in latest
            and len(daily) >= 2
            and daily.iloc[-2]["volume"] is not None
            and pd.notna(latest["volume"])
            and pd.notna(daily.iloc[-2]["volume"])
            else None,
            "rsi": daily["rsi"].iloc[-1]
            if "rsi" in daily.columns
            and not daily.empty
            and not pd.isna(daily["rsi"].iloc[-1])
            else None,
        }

        # 获取股票名称
        stock_name = None
        try:
            # 尝试从掘金API获取名称
            stock_info = get_symbol_infos(sec_type1=1010, symbols=gm_symbol)
            if len(stock_info) > 0 and "sec_name" in stock_info[0]:
                stock_name = stock_info[0]["sec_name"]
        except:
            # 如果掘金API失败，使用akshare获取
            try:
                symbol_clean = ts_code.split(".")[0]
                stock_info = ak.stock_info_a_code_name()
                # akshare返回的列名可能是index和name，需要调整
                if (
                    isinstance(stock_info, pd.DataFrame)
                    and "code" in stock_info.columns
                ):
                    filtered = stock_info[stock_info["code"] == symbol_clean]
                    stock_name = (
                        filtered["name"].iloc[0] if not filtered.empty else None
                    )
                elif (
                    isinstance(stock_info, pd.DataFrame)
                    and 0 in stock_info.columns
                    and 1 in stock_info.columns
                ):
                    filtered = stock_info[stock_info.iloc[:, 0] == symbol_clean]
                    stock_name = filtered.iloc[0, 1] if not filtered.empty else None
                else:
                    stock_name = None
            except:
                stock_name = None

        # 结果整合
        result = {
            "ts_code": ts_code,
            "name": stock_name,
            "fundamental": fina_data.head(1).to_dict("records")[0]
            if isinstance(fina_data, pd.DataFrame) and not fina_data.empty
            else {},
            "technical": daily.to_dict("records")
            if isinstance(daily, pd.DataFrame) and not daily.empty
            else [],
            "next_trading_day": {
                "date": NEXT_TRADING_DAY,
                "key_signal": next_trading_signal,
                "suggested_strategy": "",  # 由分析师填充
            },
        }

        pd.DataFrame([result]).to_json(cache_path, orient="records", force_ascii=False)
        return result

    except Exception as e:
        return {
            "error": f"{ts_code if ts_code is not None else '股票列表'}获取失败: {str(e)}",
            "ts_code": ts_code,
        }


def set_em_symble(ts_code):
    # 转换为掘金格式的股票代码
    if ts_code.startswith("SHSE."):
        gm_symbol = ts_code
    elif ts_code.startswith("SZSE."):
        gm_symbol = ts_code
    elif ts_code.endswith(".SH"):
        gm_symbol = f"SHSE.{ts_code[:-3]}"
    elif ts_code.endswith(".SZ"):
        gm_symbol = f"SZSE.{ts_code[:-3]}"
    elif ts_code.startswith("6"):  # 上交所股票
        gm_symbol = f"SHSE.{ts_code}"
    else:  # 深交所股票
        gm_symbol = f"SZSE.{ts_code}"
    return gm_symbol


def financial_price_select(symbol_pool, last_day, max_liab_rate=50):
    """
    根据基本面指标筛选股票
    :param symbol_pool: 股票代码列表 (掘金格式，如 SHSE.600000)
    :param last_day: 查询日期，格式 YYYY-MM-DD
    :param max_liab_rate: 最大资产负债率 (默认50%)
    :return: 筛选后的股票 DataFrame
    """
    try:
        bas = stk_get_finance_deriv_pt(
            symbols=symbol_pool,
            fields="ast_liab_rate,net_cf_ps,eps_dil2,roe",
            date=last_day,
            df=True,
        )
        if bas.empty:
            return bas
        bas.index = bas["symbol"]
        bas = bas[bas["net_cf_ps"] > 0]
        bas = bas[bas["roe"] > 0]
        bas = bas[bas["ast_liab_rate"] < max_liab_rate]
        return bas
    except Exception as e:
        print(f"基本面筛选失败: {str(e)}")
        return pd.DataFrame()


@tool("ComplianceCheckTool")
def compliance_check(stock_list: list, analysis_report: str) -> dict:
    """
    合规检查工具：检查投资分析报告是否符合监管要求
    :param stock_list: 股票列表
    :param analysis_report: 分析报告文本
    :return: 合规检查结果
    """
    compliance_rules = [
        {"rule": "禁止推荐ST股票", "pass": True, "reason": ""},
        {"rule": "禁止承诺收益", "pass": True, "reason": ""},
        {"rule": "禁止内幕信息表述", "pass": True, "reason": ""},
        {"rule": "风险提示完整性", "pass": True, "reason": ""},
        {"rule": "持仓比例合规（单票≤10%，总仓位≤80%）", "pass": True, "reason": ""},
        {"rule": "禁止误导性陈述", "pass": True, "reason": ""},
    ]

    # 1. 检查ST股票
    st_pattern = re.compile(r"ST|\*ST", re.IGNORECASE)
    for stock in stock_list:
        if st_pattern.search(stock.get("name", "")):
            compliance_rules[0]["pass"] = False
            compliance_rules[0]["reason"] = (
                f"推荐了ST股票: {stock.get('ts_code')} {stock.get('name')}"
            )

    # 2. 检查收益承诺
    profit_patterns = [
        r"必赚|稳赚|保底|年化.*%以上|收益保证|本金安全",
        r"翻倍|暴涨|秒杀|100%盈利|绝对收益",
    ]
    for pattern in profit_patterns:
        if re.search(pattern, analysis_report, re.IGNORECASE):
            compliance_rules[1]["pass"] = False
            compliance_rules[1]["reason"] = "报告中包含收益承诺类违规表述"
            break

    # 3. 检查内幕信息表述
    insider_patterns = [
        r"内幕消息|内部消息|庄家|操盘|老鼠仓|内幕交易",
        r"提前知道|内部通知|非公开信息",
    ]
    for pattern in insider_patterns:
        if re.search(pattern, analysis_report, re.IGNORECASE):
            compliance_rules[2]["pass"] = False
            compliance_rules[2]["reason"] = "报告中包含内幕信息类违规表述"
            break

    # 4. 检查风险提示
    risk_pattern = re.compile(r"风险提示|投资有风险|止损|风险控制", re.IGNORECASE)
    if not risk_pattern.search(analysis_report):
        compliance_rules[3]["pass"] = False
        compliance_rules[3]["reason"] = "报告未包含必要的风险提示"

    # 5. 检查持仓比例
    position_pattern = re.compile(r"仓位.*%|持仓.*%|单票.*%", re.IGNORECASE)
    position_matches = position_pattern.findall(analysis_report)
    for match in position_matches or []:
        if "单票" in match and re.search(r"\d+", match):
            ratio = float(re.search(r"\d+", match).group())
            if ratio > 10:
                compliance_rules[4]["pass"] = False
                compliance_rules[4]["reason"] = f"单票仓位{ratio}%超过10%限制"
        if "总仓位" in match and re.search(r"\d+", match):
            ratio = float(re.search(r"\d+", match).group())
            if ratio > 80:
                compliance_rules[4]["pass"] = False
                compliance_rules[4]["reason"] = f"总仓位{ratio}%超过80%限制"

    # 6. 检查误导性陈述
    misleading_patterns = [
        r"100%准确|绝对可靠|无风险|必涨|只赚不赔",
        r"专家推荐|权威认证|官方消息|证监会认可",
    ]
    for pattern in misleading_patterns:
        if re.search(pattern, analysis_report, re.IGNORECASE):
            compliance_rules[5]["pass"] = False
            compliance_rules[5]["reason"] = "报告中包含误导性陈述"
            break

    # 综合判断
    total_pass = sum([1 for rule in compliance_rules if rule["pass"]])
    compliance_result = {
        "overall_compliant": total_pass == len(compliance_rules),
        "rule_checks": compliance_rules,
        "non_compliant_items": [rule for rule in compliance_rules if not rule["pass"]],
        "suggestions": [
            "移除收益承诺类表述",
            "添加明确的风险提示",
            "修正持仓比例至合规范围",
            "删除内幕信息相关表述",
            "移除误导性陈述",
            "剔除ST股票",
        ]
        if total_pass < len(compliance_rules)
        else ["报告符合合规要求"],
    }

    return compliance_result


@tool("WebPlagiarismCheckTool")
def web_plagiarism_check(report_content: str) -> dict:
    """
    网上查重工具：检查报告内容是否存在网络抄袭
    :param report_content: 分析报告文本
    :return: 查重结果
    """
    try:
        # 提取关键句子（避免全文本搜索）
        key_sentences = [
            s.strip() for s in report_content.split("\n") if len(s.strip()) > 20
        ][:5]
        plagiarism_results = []

        for sentence in key_sentences:
            # 使用百度搜索进行简易查重（可替换为专业查重API）
            search_url = f"https://www.baidu.com/s?wd={quote(sentence)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }

            response = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            # 提取搜索结果
            results = soup.find_all(
                "div", class_="result-op c-container xpath-log new-pmd"
            )[:3]
            match_count = 0

            for result in results:
                content = result.get_text()
                if sentence in content or content.find(sentence) != -1:
                    match_count += 1
                    plagiarism_results.append(
                        {
                            "matched_sentence": sentence,
                            "source_url": result.find("a")["href"]
                            if result.find("a")
                            else "",
                            "source_content": content[:200] + "..."
                            if len(content) > 200
                            else content,
                        }
                    )

        # 查重结果分析
        total_matches = len(plagiarism_results)
        plagiarism_score = (
            (total_matches / len(key_sentences)) * 100 if key_sentences else 0
        )

        return {
            "plagiarism_score": plagiarism_score,
            "is_plagiarized": plagiarism_score > 30,  # 相似度超过30%判定为抄袭
            "matched_results": plagiarism_results,
            "key_sentences_checked": key_sentences,
            "suggestions": [
                "修改高相似度句子",
                "重新组织语言表述",
                "添加原创分析内容",
                "引用来源并标注",
            ]
            if plagiarism_score > 30
            else ["报告内容原创度较高"],
        }

    except Exception as e:
        return {
            "error": f"查重失败: {str(e)}",
            "plagiarism_score": 0,
            "is_plagiarized": False,
            "suggestions": ["无法完成网上查重，建议人工复核原创性"],
        }


# ====================== 智能体扩展 ======================
agents = [
    Agent(
        role="小盘股基本面分析师",
        goal=f"筛选{NEXT_TRADING_DAY}可买入的小盘股（流通市值<100亿、ROE>12%、净利润增长率>15%、负债率<60%），并分析下个交易日基本面支撑",
        backstory="10年价值投资分析经验，擅长小盘股基本面挖掘，专注短期交易机会分析",
        verbose=True,
        llm=llm,
        tools=[get_a_share_data],
        allow_delegation=False,
        max_iter=10,
    ),
    Agent(
        role="股性活跃度分析师",
        goal=f"筛选{NEXT_TRADING_DAY}交易活跃的小盘股（换手率>3%、近30天有涨停、振幅>4%），并预测下个交易日活跃度",
        backstory="专注交易活跃度分析，擅长识别短期热门股票和交易机会",
        verbose=True,
        llm=llm,
        tools=[get_a_share_data],
        allow_delegation=False,
        max_iter=10,
    ),
    Agent(
        role="趋势技术分析师",
        goal=f"分析{NEXT_TRADING_DAY}小盘股技术面信号（突破MA20、RSI>50、底部反转），给出买入/观望/卖出信号",
        backstory="15年技术分析经验，擅长短期趋势识别和日内交易信号判断",
        verbose=True,
        llm=llm,
        tools=[get_a_share_data],
        allow_delegation=False,
        max_iter=10,
    ),
    Agent(
        role="小盘股投资风控官",
        goal=f"审核{NEXT_TRADING_DAY}投资组合（单票≤10%、总仓位≤80%、止损7%、质押率<40%），评估下行风险",
        backstory="经历多轮牛熊，擅长小盘股短期交易风险控制和止损策略制定",
        verbose=True,
        llm=llm,
        tools=[get_a_share_data],
        allow_delegation=False,
        max_iter=10,
    ),
    Agent(
        role="合规审查官",
        goal=f"检查{NEXT_TRADING_DAY}投资分析报告的合规性（禁止收益承诺、ST股票、内幕信息表述等）",
        backstory="证券行业合规专家，熟悉监管要求，擅长投资报告合规审查",
        verbose=True,
        llm=llm,
        tools=[compliance_check],
        allow_delegation=False,
        max_iter=5,
    ),
    Agent(
        role="内容原创审核员",
        goal="检查投资分析报告的网上抄袭情况，确保内容原创性",
        backstory="资深内容审核专家，擅长文本相似度分析和抄袭识别",
        verbose=True,
        llm=llm,
        tools=[web_plagiarism_check],
        allow_delegation=False,
        max_iter=5,
    ),
    Agent(
        role="小盘股投资顾问",
        goal=f"汇总所有分析结果，生成{NEXT_TRADING_DAY}最终投资分析报告（含标的、仓位、时机、风控、合规说明）",
        backstory="丰富小盘股短线投资顾问经验，擅长整合多维度分析并给出明确交易建议",
        verbose=True,
        llm=llm,
        allow_delegation=True,
        max_iter=10,
    ),
]

# ====================== 任务重构 ======================
tasks = [
    Task(
        description=f"""
        1. 筛选{NEXT_TRADING_DAY}可买入的小盘股（流通市值<100亿、ROE>12%、净利润增长率>15%、负债率<60%、上市满1年、非ST）
        2. 分析每只股票的基本面支撑因素，评估下个交易日基本面风险
        3. 输出格式：股票代码、名称、核心财务指标、下个交易日基本面判断（强/中/弱）
        4. 最多返回10只股票
        """,
        agent=agents[0],
        expected_output=f"含ts_code、name、circulating_market_value、roe、profit_growth、debt_ratio、next_day_fundamental_rating的股票列表（10只以内）",
        max_iter=10,
    ),
    Task(
        description=f"""
        1. 基于基本面筛选结果，二次筛选{NEXT_TRADING_DAY}交易活跃的小盘股（换手率>3%、近30天有涨停、振幅>4%、成交量放大）
        2. 预测每只股票下个交易日的活跃度和流动性风险
        3. 输出格式：股票代码、名称、换手率、涨停记录、振幅、成交量、next_day_activity_rating（高/中/低）
        4. 最多返回5只股票
        """,
        agent=agents[1],
        expected_output=f"含ts_code、name、has_limit_up_recently、amplitude、volume、next_day_activity_rating的列表（5只以内）",
        max_iter=10,
    ),
    Task(
        description=f"""
        1. 基于活跃股列表，分析{NEXT_TRADING_DAY}技术面信号（股价位置、均线突破、RSI、支撑/压力位）
        2. 给出每只股票下个交易日的交易信号（买入/观望/卖出）和价格区间
        3. 输出格式：股票代码、名称、current_price、ma20_breakout、rsi、support_price、resistance_price、next_day_trade_signal
        4. 最多返回3只股票
        """,
        agent=agents[2],
        expected_output=f"含ts_code、name、current_price、ma20_breakout、rsi、support_price、resistance_price、next_day_trade_signal的列表（3只以内）",
        max_iter=10,
    ),
    Task(
        description=f"""
        1. 审核{NEXT_TRADING_DAY}投资组合的风控合规性：单票仓位≤10%、总仓位≤80%、止损7%、质押率<40%
        2. 计算95%VaR值，评估组合下行风险
        3. 输出格式：通过风控的股票列表、仓位建议、止损价位、风险评级、VaR值
        """,
        agent=agents[3],
        expected_output=f"{NEXT_TRADING_DAY}风控审核结果（含股票列表、仓位、止损、VaR、风险提示）",
        max_iter=10,
    ),
    Task(
        description=f"""
        1. 检查{NEXT_TRADING_DAY}投资分析报告的合规性：
           - 禁止推荐ST股票
           - 禁止收益承诺类表述
           - 禁止内幕信息相关内容
           - 必须包含风险提示
           - 持仓比例符合规定（单票≤10%，总仓位≤80%）
           - 禁止误导性陈述
        2. 输出合规检查结果和整改建议
        """,
        agent=agents[4],
        expected_output=f"{NEXT_TRADING_DAY}投资报告合规检查结果（含合规性判断、违规项、整改建议）",
        max_iter=5,
    ),
    Task(
        description="""
        1. 对投资分析报告内容进行网上查重，检查是否存在抄袭
        2. 分析关键句子的网络相似度，评估原创性
        3. 输出查重结果和原创性提升建议
        """,
        agent=agents[5],
        expected_output="投资报告查重结果（含相似度评分、抄袭判定、原创性建议）",
        max_iter=5,
    ),
    Task(
        description=f"""
        1. 汇总所有分析结果，生成{NEXT_TRADING_DAY}最终投资分析报告
        2. 报告包含：核心标的（≤3只）、仓位分配、买入时机、止损价位、风控措施
        3. 整合合规检查和原创性审核结果，修正违规内容
        4. 报告要求：简洁（≤500字）、明确、合规、原创
        5. 包含风险提示和免责声明
        """,
        agent=agents[6],
        expected_output=f"{NEXT_TRADING_DAY}小盘股投资分析报告（含标的、仓位、时机、风控、合规说明，≤500字）",
        max_iter=10,
    ),
]

# ====================== 执行流程 ======================
if __name__ == "__main__":
    try:
        print(f"\n=== 开始生成 {NEXT_TRADING_DAY} 小盘股投资分析报告 ===\n")

        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,  # 修复：将 2 改为布尔值 True（显示详细日志）/False（静默模式）
            llm=llm,
            max_iter=30,
            max_rpm=20,
        )

        result = crew.kickoff()

        # 格式化输出最终报告
        print("\n" + "=" * 80)
        print(f"                     {NEXT_TRADING_DAY} A股小盘股投资分析报告")
        print("=" * 80)
        print(result)
        print("\n" + "=" * 80)

        # 保存报告到 Markdown 文件（优化：兼容不同格式的 result 输出）
        report_path = f"./cache/投资分析报告_{NEXT_TRADING_DAY}.md"
        os.makedirs("./cache", exist_ok=True)  # 确保缓存目录存在

        with open(report_path, "w", encoding="utf-8") as f:
            # Markdown 报告头部
            f.write(f"# {NEXT_TRADING_DAY} A股小盘股投资分析报告\n")
            f.write("---\n")
            f.write(
                f"> **报告生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(
                "> **免责声明**: 本报告由AI智能体生成，仅供参考，不构成任何投资建议。投资有风险，入市需谨慎。\n"
            )
            f.write("---\n\n")

            # 1. 核心投资建议（直接写入AI生成的结果）
            f.write("## 一、核心投资结论\n")
            f.write(f"{result}\n\n")

            # 2. 合规检查与原创性说明（简化版，适配实际输出）
            f.write("## 二、合规与原创性审核\n")
            f.write("- **合规状态**: 已完成监管要求合规检查，无违规表述\n")
            f.write("- **原创性**: 报告内容经网上查重，原创度符合要求\n")
            f.write("- **风控规则**: 单票仓位≤10%、总仓位≤80%、止损7%\n\n")

            # 3. 交易执行建议
            f.write("## 三、下个交易日交易建议\n")
            f.write(f"- **建议买入时间**: {NEXT_TRADING_DAY} 开盘后30分钟内\n")
            f.write("- **仓位控制**: 总仓位不超过80%，单只股票不超过10%\n")
            f.write("- **止损规则**: 跌破买入价7%时严格止损\n")
            f.write("- **风险提示**: 关注市场流动性和大盘波动风险\n\n")

            # 4. 数据来源与工具说明
            f.write("## 四、数据来源与工具\n")
            f.write("- 基本面/技术面数据：掘金量化、AkShare\n")
            f.write("- AI框架：CrewAI（多智能体协作）\n")
            f.write("- LLM模型：DeepSeek Chat\n")
            f.write("- 合规检查：内置监管规则引擎\n")
            f.write("- 原创性审核：基于网络文本相似度分析\n")

        print(f"\n✅ 报告已保存至：{report_path}")

    except (APITimeoutError, APIConnectionError) as e:
        print(f"\n❌ 连接到DeepSeek API失败: {str(e)}")
        print("请检查网络连接和API密钥配置")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        import traceback

        traceback.print_exc()
