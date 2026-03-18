"""
A股小盘股投资分析报告生成器 - 重构版

使用方法:
    python main.py

环境变量配置 (.env文件):
    OPENAI_API_KEY=your_api_key
    OPENAI_BASE_URL=https://api.deepseek.com/v1
    GM_API_TOKEN=your_gm_token
    DB_HOST=localhost
    DB_PORT=3306
    DB_USER=root
    DB_PASSWORD=your_password
    DB_NAME=stock_base
"""
import os
import sys
import warnings
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 忽略警告
warnings.filterwarnings("ignore")

# 配置日志
from config import config
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入模块
from utils import format_market_cap
from data_fetcher import data_fetcher
from analyzer import analyzer


def generate_report() -> str:
    """生成投资分析报告"""
    logger.info("开始生成投资分析报告...")
    
    # 获取推荐股票
    top_stocks = analyzer.get_top_stocks(n=10)
    
    if not top_stocks:
        return "未能获取到符合条件的股票数据"
    
    # 生成报告内容
    report_lines = [
        "# A股小盘股投资分析报告",
        f"\n**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## 推荐股票列表\n"
    ]
    
    for i, stock in enumerate(top_stocks, 1):
        ts_code = stock.get("ts_code", "")
        name = stock.get("name", "")
        score = stock.get("score", 0)
        recommendation = stock.get("recommendation", "")
        signals = stock.get("signals", [])
        market_cap = stock.get("market_cap", 0)
        
        report_lines.append(f"\n### {i}. {name} ({ts_code})")
        report_lines.append(f"- **综合评分**: {score}/100")
        report_lines.append(f"- **投资建议**: {recommendation}")
        report_lines.append(f"- **流通市值**: {format_market_cap(market_cap)}")
        report_lines.append(f"- **积极信号**: {', '.join(signals) if signals else '无'}")
        
        # 技术指标
        technical = stock.get("technical", {})
        if technical:
            report_lines.append(f"- **当前价格**: {technical.get('current_price', 'N/A')}")
            report_lines.append(f"- **支撑位**: {technical.get('support_price', 'N/A')}")
            report_lines.append(f"- **阻力位**: {technical.get('resistance_price', 'N/A')}")
            report_lines.append(f"- **RSI**: {technical.get('rsi', 'N/A')}")
    
    # 添加风险提示
    report_lines.extend([
        "\n## 风险提示",
        "\n1. 本报告仅供参考，不构成投资建议",
        "2. 股市有风险，投资需谨慎",
        "3. 过往业绩不代表未来表现",
        "4. 请根据自身风险承受能力做出投资决策"
    ])
    
    report = "\n".join(report_lines)
    
    # 保存报告
    report_dir = "./reports"
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(
        report_dir,
        f"invest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"报告已保存至: {report_file}")
    return report


def main():
    """主函数"""
    try:
        report = generate_report()
        print("\n" + "="*80)
        print(report)
        print("="*80 + "\n")
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
