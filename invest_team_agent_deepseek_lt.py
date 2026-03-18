import os
import re
import warnings
from urllib.parse import quote
from typing import Optional

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


# LLM配置（Kimi2兼容OpenAI API）
def get_kimi_llm():
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.moonshot.cn/v1"),
        model="moonshot-v1-32k",
        temperature=0.1,
        timeout=30.0,
        max_retries=2,
    )


llm = get_kimi_llm()


def get_clean_code(ts_code):
    """提取6位数字股票代码"""
    if not ts_code:
        return ""
    match = re.search(r"\d{6}", str(ts_code))
    return match.group() if match else str(ts_code)


# ====================== 核心工具扩展 ======================
@tool("AShareDataTool")
def get_a_share_data(ts_code: Optional[str] = None, limit_data: bool = True) -> dict:
    """获取A股主板股票数据（基本面+技术面），新增下个交易日关键信号"""
    if ts_code is not None:
        ts_code = str(ts_code)
        # 验证股票代码格式
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
                # 过滤北交所和创业板/科创板，保留主板
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
                    if len(stock_list.columns) >= 2:
                        stock_list = stock_list.iloc[:, :2]
                        stock_list.columns = ["symbol", "name"]
                    elif len(stock_list.columns) == 1:
                        stock_list = stock_list.rename(
                            columns={stock_list.columns[0]: "symbol"}
                        )
                        stock_list["name"] = ""
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
            current_data = current(symbols=gm_symbol)
            if current_data and len(current_data) > 0:
                tick = current_data[0]
                current_price = tick.price if hasattr(tick, 'price') else (tick.get('price') if isinstance(tick, dict) else None)
            else:
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
        return {"error": str(e), "ts_code": ts_code}


def set_em_symble(ts_code):
    if ts_code.startswith("SHSE.") or ts_code.startswith("SZSE."):
        return ts_code
    clean = get_clean_code(ts_code)
    if clean.startswith("6"):
        return f"SHSE.{clean}"
    else:
        return f"SZSE.{clean}"


@tool("ComplianceCheckTool")
def compliance_check(stock_list: list, analysis_report: str) -> dict:
    """合规检查工具：检查投资分析报告是否符合监管要求"""
    compliance_rules = [
        {"rule": "禁止推荐ST股票", "pass": True, "reason": ""},
        {"rule": "禁止承诺收益", "pass": True, "reason": ""},
        {"rule": "禁止内幕信息表述", "pass": True, "reason": ""},
        {"rule": "风险提示完整性", "pass": True, "reason": ""},
        {"rule": "持仓比例合规", "pass": True, "reason": ""},
    ]

    # 1. ST
    st_pattern = re.compile(r"ST|\*ST", re.IGNORECASE)
    for stock in stock_list:
        if st_pattern.search(str(stock.get("name", ""))):
            compliance_rules[0]["pass"] = False
            compliance_rules[0]["reason"] = f"推荐了ST股票: {stock.get('ts_code')}"

    # 2. 收益承诺
    if re.search(r"必赚|稳赚|保底|收益保证|100%盈利|翻倍|暴涨", analysis_report):
        compliance_rules[1]["pass"] = False
        compliance_rules[1]["reason"] = "包含收益承诺违规表述"

    # 3. 内幕
    if re.search(r"内幕消息|内部消息|庄家操盘|内部通知", analysis_report):
        compliance_rules[2]["pass"] = False
        compliance_rules[2]["reason"] = "包含内幕信息相关表述"

    # 4. 风险提示
    if not re.search(r"风险提示|投资有风险|止损|风控", analysis_report):
        compliance_rules[3]["pass"] = False
        compliance_rules[3]["reason"] = "缺失风险提示内容"

    # 5. 持仓比例
    pos_matches = re.finditer(r"(?:仓位|持仓|单票)\s*(\d+(?:\.\d+)?)%", analysis_report)
    for m in pos_matches:
        val = float(m.group(1))
        txt = m.group(0)
        if "单票" in txt and val > 10:
            compliance_rules[4]["pass"] = False
            compliance_rules[4]["reason"] = f"单票仓位{val}%超限"
        elif val > 80:
            compliance_rules[4]["pass"] = False
            compliance_rules[4]["reason"] = f"总仓位{val}%超限"

    total_pass = sum([1 for r in compliance_rules if r["pass"]])
    return {
        "overall_compliant": total_pass == len(compliance_rules),
        "checks": compliance_rules,
        "suggestions": ["修正违规内容", "补充风险提示"]
        if total_pass < len(compliance_rules)
        else ["合规"],
    }


@tool("WebPlagiarismCheckTool")
def web_plagiarism_check(report_content: str) -> dict:
    """网上查重工具"""
    try:
        sentences = [
            s.strip() for s in report_content.split("\n") if len(s.strip()) > 20
        ][:3]
        if not sentences:
            return {"plagiarism_score": 0, "is_plagiarized": False}

        matches = 0
        for s in sentences:
            try:
                resp = requests.get(f"https://www.baidu.com/s?wd={quote(s)}", timeout=5)
                if s in resp.text:
                    matches += 1
            except:
                pass

        score = (matches / len(sentences)) * 100
        return {"plagiarism_score": score, "is_plagiarized": score > 30}
    except:
        return {"error": "查重服务暂不可用"}


# ====================== 智能体定义 ======================
agents = [
    Agent(
        role="小盘股基本面分析师",
        goal=f"筛选{NEXT_TRADING_DAY}可买入的小盘股（流通市值<100亿、ROE>12%、净利润增长率>15%、负债率<60%）",
        backstory="擅长挖掘具有高成长潜力的小市值标的，注重财务质量和护城河。",
        verbose=True,
        llm=llm,
        tools=[get_a_share_data],
        max_iter=10,
    ),
    Agent(
        role="股性活跃度分析师",
        goal=f"筛选{NEXT_TRADING_DAY}交易活跃的股票（换手率>3%、近期有涨停、振幅大）",
        backstory="追踪市场热点和资金流向，擅长识别短期爆发力强的牛股。",
        verbose=True,
        llm=llm,
        tools=[get_a_share_data],
        max_iter=10,
    ),
    Agent(
        role="趋势技术分析师",
        goal=f"分析{NEXT_TRADING_DAY}技术面信号（MA20、RSI、支撑压力位），给出买卖建议",
        backstory="精通K线形态和各种技术指标，精准捕捉买卖点。",
        verbose=True,
        llm=llm,
        tools=[get_a_share_data],
        max_iter=10,
    ),
    Agent(
        role="小盘股投资风控官",
        goal=f"审核投资组合的风控指标（单票≤10%、总仓位≤80%、止损7%）",
        backstory="纪律严明，始终将风险控制放在首位，确保资产安全。",
        verbose=True,
        llm=llm,
        tools=[get_a_share_data],
        max_iter=10,
    ),
    Agent(
        role="合规审查官",
        goal=f"检查报告的合规性，剔除ST和误导性表述",
        backstory="熟悉法律法规，确保所有建议符合行业规范。",
        verbose=True,
        llm=llm,
        tools=[compliance_check],
        max_iter=5,
    ),
    Agent(
        role="内容原创审核员",
        goal="确保报告内容的原创性，防止抄袭",
        backstory="严格审核每一份文稿，维护分析工作的独立性。",
        verbose=True,
        llm=llm,
        tools=[web_plagiarism_check],
        max_iter=5,
    ),
    Agent(
        role="小盘股投资顾问",
        goal=f"汇总分析，生成最终报告",
        backstory="整合各方观点，给出简洁明了的行动指南。",
        verbose=True,
        llm=llm,
        allow_delegation=True,
        max_iter=10,
    ),
]

tasks = [
    Task(
        description=f"基于基本面筛选10只以内的优质小盘股。输出：ts_code, name, roe, profit_growth, debt_ratio, circulating_market_value。",
        agent=agents[0],
        expected_output="基本面优质股票列表",
    ),
    Task(
        description=f"在基本面列表中筛选活跃度高的股票。输出：ts_code, name, volume_trend, has_limit_up_recently。",
        agent=agents[1],
        expected_output="高活跃度股票列表",
    ),
    Task(
        description=f"进行技术分析，评估买入点。输出：current_price, ma20_position, support_price, resistance_price。",
        agent=agents[2],
        expected_output="技术面分析报告",
    ),
    Task(
        description="审核风控指标，给出仓位建议和止损位。",
        agent=agents[3],
        expected_output="风控审核意见",
    ),
    Task(
        description="合规审查，确保无误导表述。",
        agent=agents[4],
        expected_output="合规报告",
    ),
    Task(description="原创性审核。", agent=agents[5], expected_output="查重报告"),
    Task(
        description=f"生成最终报告。包含：核心标的（≤3只）、仓位、买入价、止损价。要求合规、原创、简洁（500字内）。",
        agent=agents[6],
        expected_output="最终投资报告",
    ),
]

if __name__ == "__main__":
    try:
        crew = Crew(
            agents=agents, tasks=tasks, process=Process.sequential, verbose=True
        )
        result = crew.kickoff()
        print("\n" + "=" * 50 + "\n最终决策报告\n" + "=" * 50)
        print(result)

        path = f"./cache/SmallCapReport_{NEXT_TRADING_DAY}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {NEXT_TRADING_DAY} 小盘股投资报告\n\n{str(result)}")
    except Exception as e:
        print(f"Error: {e}")
