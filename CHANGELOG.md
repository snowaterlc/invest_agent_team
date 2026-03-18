# 更新日志

## [2026-03-16] - 重大更新

### 主要变更

#### 1. AI模型迁移
- **从 DeepSeek 迁移至 Kimi2**
  - API地址: `https://api.deepseek.com/v1` → `https://api.moonshot.cn/v1`
  - 模型名称: `deepseek-chat` → `moonshot-v1-32k`
  - 函数名: `get_deepseek_llm()` → `get_kimi_llm()`
  - 错误提示: `连接到DeepSeek API失败` → `连接到Kimi API失败`
  - 报告显示: `DeepSeek Chat` → `Kimi2 (moonshot-v1-32k)`

#### 2. 代码重构
- **新增模块化文件**
  - `config.py` - 集中管理所有配置参数
  - `utils.py` - 通用工具函数
  - `data_fetcher.py` - 数据获取逻辑
  - `analyzer.py` - 股票分析逻辑
  - `main.py` - 简化后的主程序入口

#### 3. Bug修复
- **缓存机制修复**
  - 添加缓存过期检查 `is_cache_valid()`
  - 默认缓存有效期24小时
  - 解决价格数据不更新的问题

- **数据库修复**
  - 修复 `create_mysql_engine` 函数定义顺序问题
  - 修复 `Base` 未定义问题
  - 修复 `InvestmentReport` 和 `SelectedStock` 模型类定义顺序问题
  - 修复 `CrewOutput` 对象处理逻辑

- **API参数修复**
  - 修复 `get_history_symbol()` 参数 `symbols` → `symbol`
  - 修复股票代码提取逻辑 `split(".")[0]` → `split(".")[-1]`
  - 修复 `stk_get_daily_valuation_pt()` 参数错误

#### 4. 功能增强
- **股票列表获取优化**
  - 优先使用掘金量化API `get_symbols()`
  - 参考 snowater 包的 `get_normal_stocks()` 实现
  - 过滤ST股、停牌股、次新股

- **价格获取优化**
  - 优先使用实时数据 `current()`
  - 失败时回退到历史数据收盘价
  - 添加详细的日志记录

### 修改的文件

| 文件 | 修改类型 |
|------|----------|
| `.env` | 修改 - 更新API密钥和地址 |
| `next_trading_day_invest_report_mysql.py` | 修改 - 修复所有Bug |
| `invest_team_agent_deepseek_lt.py` | 修改 - DeepSeek → Kimi2 |
| `invest_team_agent_deepseek.py` | 修改 - DeepSeek → Kimi2 |
| `next_trading_day_invest_report.py` | 修改 - DeepSeek → Kimi2 |
| `config.py` | 新增 - 配置模块 |
| `utils.py` | 新增 - 工具函数模块 |
| `data_fetcher.py` | 新增 - 数据获取模块 |
| `analyzer.py` | 新增 - 分析模块 |
| `main.py` | 新增 - 主程序入口 |

### 删除的文件
- `test_db_final.py`
- `test_db_simple.py`
- `test_gm_api.py`

## [历史版本]

### 早期版本
- 初始版本基于 DeepSeek API
- 支持掘金量化和akshare数据源
- 使用CrewAI多智能体框架
- 支持MySQL数据库存储

---

## 使用说明

### 环境配置
```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
# 编辑 .env 文件
OPENAI_API_KEY=your_kimi_api_key
OPENAI_BASE_URL=https://api.moonshot.cn/v1
GM_API_TOKEN=your_gm_token
```

### 运行程序
```bash
# 运行主程序
python next_trading_day_invest_report_mysql.py

# 或运行重构版本
python main.py
```

### 查看报告
生成的报告保存在 `./cache/` 目录下，文件名格式：`投资分析报告_YYYY-MM-DD.md`

---

## 技术栈

- **AI模型**: Kimi2 (moonshot-v1-32k)
- **数据源**: 掘金量化、AkShare
- **数据库**: MySQL
- **框架**: CrewAI（多智能体协作）
- **语言**: Python 3.13

---

## 注意事项

1. **API密钥**: 确保 `.env` 文件中的 `OPENAI_API_KEY` 已更新为Kimi的API密钥
2. **缓存清理**: 如需获取最新数据，可删除 `./cache/` 目录下的缓存文件
3. **数据库**: 确保MySQL服务已启动，且配置正确
4. **网络**: 确保能够访问 `https://api.moonshot.cn/v1`

---

## 联系方式

如有问题，请提交Issue或联系开发团队。
