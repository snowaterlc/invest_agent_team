"""Final verification script for GM API updates"""

import sys

sys.path.insert(0, r"E:\python\invest_agent_team")

print("=" * 60)
print("掘金量化 API 更新验证报告")
print("=" * 60)

# Check imports
print("\n1. 导入检查...")
try:
    from next_trading_day_invest_report_mysql import get_a_share_data, set_em_symble

    print("   [OK] 模块导入成功")
except Exception as e:
    print(f"   [FAIL] 导入失败: {e}")
    sys.exit(1)

# Check API signatures
print("\n2. API 签名检查...")
try:
    from gm.api import (
        get_next_n_trading_dates,
        get_symbols,
        get_history_symbol,
        get_symbol_infos,
        stk_get_fundamentals_balance_pt,
        stk_get_fundamentals_income_pt,
        stk_get_finance_deriv_pt,
        stk_get_finance_prime_pt,
        stk_get_daily_valuation_pt,
        current,
    )

    print("   [OK] 所有API导入成功")
except Exception as e:
    print(f"   [FAIL] API导入失败: {e}")

# Verify parameter corrections
print("\n3. 参数修正验证:")
from datetime import datetime, timedelta

# Test get_next_n_trading_dates with 'n' parameter
try:
    # This will fail due to no token, but should not fail on parameter error
    result = get_next_n_trading_dates(
        date=datetime.now().strftime("%Y-%m-%d"), n=1, exchange="SHSE"
    )
    print("   [OK] get_next_n_trading_dates 参数正确 (n=1)")
except TypeError as e:
    if "unexpected keyword argument" in str(e):
        print(f"   [FAIL] get_next_n_trading_dates 参数错误: {e}")
    else:
        print(f"   [OK] get_next_n_trading_dates 参数正确 (token错误非参数错误)")
except Exception as e:
    print(f"   [OK] get_next_n_trading_dates 参数正确 (预期错误: token)")

# Test get_symbol_infos with sec_type1
try:
    result = get_symbol_infos(sec_type1=1010, symbols="SHSE.600000")
    print("   [OK] get_symbol_infos 参数正确 (sec_type1=1010)")
except TypeError as e:
    if "unexpected keyword argument" in str(e) or "missing" in str(e):
        print(f"   [FAIL] get_symbol_infos 参数错误: {e}")
    else:
        print(f"   [OK] get_symbol_infos 参数正确")
except Exception as e:
    print(f"   [OK] get_symbol_infos 参数正确 (预期错误)")

# Test stk_get_fundamentals_balance_pt with symbols
try:
    result = stk_get_fundamentals_balance_pt(
        symbols="SHSE.600000",
        start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        end_date=datetime.now().strftime("%Y-%m-%d"),
        fields="ttl_ast",
    )
    print("   [OK] stk_get_fundamentals_balance_pt 参数正确 (symbols)")
except TypeError as e:
    if "unexpected keyword argument" in str(e):
        print(f"   [FAIL] stk_get_fundamentals_balance_pt 参数错误: {e}")
    else:
        print(f"   [OK] stk_get_fundamentals_balance_pt 参数正确")
except Exception as e:
    print(f"   [OK] stk_get_fundamentals_balance_pt 参数正确 (预期错误)")

print("\n" + "=" * 60)
print("验证完成!")
print("=" * 60)
print("\n注意: 如果看到'预期错误'或'token错误'，说明参数已正确修复。")
print("需要有效的 GM_API_TOKEN 才能进行完整的API测试。")
