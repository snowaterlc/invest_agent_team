"""Simple test script to verify GM API changes"""

import sys
import os

# Add project path
sys.path.insert(0, r"E:\python\invest_agent_team")

from datetime import datetime, timedelta


def test_gm_api():
    """Test GM API calls"""
    print("Testing GM API...")

    # Test 1: get_next_n_trading_dates
    print("\n1. Testing get_next_n_trading_dates...")
    try:
        from gm.api import get_next_n_trading_dates

        # Try with 'n' parameter instead of 'count'
        result = get_next_n_trading_dates(
            date=datetime.now().strftime("%Y-%m-%d"), n=1, exchange="SHSE"
        )
        print(f"   Result: {result}")
        print("   OK: get_next_n_trading_dates works!")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 2: get_symbols
    print("\n2. Testing get_symbols...")
    try:
        from gm.api import get_symbols

        result = get_symbols(
            sec_type1=1010,
            sec_type2=101001,
            exchanges="SHSE,SZSE",
            skip_suspended=True,
            skip_st=True,
            df=True,
        )
        print(
            f"   Result shape: {result.shape if hasattr(result, 'shape') else len(result)}"
        )
        print("   OK: get_symbols works!")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 3: get_history_symbol
    print("\n3. Testing get_history_symbol...")
    try:
        from gm.api import get_history_symbol

        result = get_history_symbol(
            symbol="SHSE.600000",
            start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            df=True,
        )
        print(
            f"   Result shape: {result.shape if hasattr(result, 'shape') else len(result)}"
        )
        print("   OK: get_history_symbol works!")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 4: get_symbol_infos
    print("\n4. Testing get_symbol_infos...")
    try:
        from gm.api import get_symbol_infos

        result = get_symbol_infos(sec_type1=1010, symbols="SHSE.600000")
        print(f"   Result: {len(result)} record(s)")
        print("   OK: get_symbol_infos works!")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 5: stk_get_fundamentals_balance_pt
    print("\n5. Testing stk_get_fundamentals_balance_pt...")
    try:
        from gm.api import stk_get_fundamentals_balance_pt

        result = stk_get_fundamentals_balance_pt(
            symbols="SHSE.600000",
            start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            fields="ttl_ast,mny_cptl",
        )
        print(f"   Result: {len(result)} record(s)")
        print("   OK: stk_get_fundamentals_balance_pt works!")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 5: stk_get_fundamentals_balance_pt
    print("\n5. Testing stk_get_fundamentals_balance_pt...")
    try:
        from gm.api import stk_get_fundamentals_balance_pt

        result = stk_get_fundamentals_balance_pt(
            symbol="SHSE.600000",
            start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            end_date=datetime.now().strftime("%Y-%m-%d"),
            fields="ttl_ast,mny_cptl",
        )
        print(f"   Result: {len(result)} record(s)")
        print("   OK: stk_get_fundamentals_balance_pt works!")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_gm_api()
