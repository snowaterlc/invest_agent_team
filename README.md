# AIGC 项目使用手册

本项目基于 Python，使用 CrewAI 多智能体框架和 Kimi2 AI 模型，支持掘金量化和 AkShare 数据源，生成 A 股小盘股投资分析报告。

## 项目简介

- **AI 模型**: Kimi2 (moonshot-v1-32k)
- **数据源**: 掘金量化、AkShare
- **数据库**: MySQL
- **框架**: CrewAI（多智能体协作）
- **语言**: Python 3.13

本文档旨在帮助开发者快速搭建环境、配置依赖、运行程序以及常见维护工作。

目录
- 环境要求
- 快速开始
- 环境变量与配置
- 运行入口定位
- 数据库与迁移
- 测试
- 部署与运维
- 维护与贡献
- 变更日志

## 1. 环境要求
- Python 版本：建议 Python 3.8 及以上（以项目实际需求为准）
- 操作系统：Windows、Linux、macOS 均可
- 网络：可访问 PyPI 以安装依赖，确保对外 API（如 OpenAI）及数据库服务可用

## 2. 快速开始
### 2.1 设置虚拟环境
Windows PowerShell:
```
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
```
注意：在某些系统中执行策略可能阻止脚本运行。若遇到权限问题，请临时放宽策略后再激活：
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

Linux/macOS Bash:
```
python3 -m venv venv
source venv/bin/activate
```

注意：以上激活命令需要与实际路径匹配，推荐直接在命令中使用 `.
\venv\Scripts\activate`。

Linux/macOS Bash:
```
python3 -m venv venv
source venv/bin/activate
```

### 2.2 安装依赖
```
pip install -r requirements.txt
```
如网络受限，可使用国内镜像源，例如：
```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.3 配置环境变量

本项目使用 `.env` 文件管理环境变量，请按以下格式配置：

```
# Kimi API 配置（必需）
OPENAI_API_KEY=your_kimi_api_key
OPENAI_BASE_URL=https://api.moonshot.cn/v1

# 掘金量化 Token（必需）
GM_API_TOKEN=your_gm_token

# MySQL 数据库配置（可选，默认使用本地数据库）
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=stock_base

# 日志等级（可选）
LOG_LEVEL=INFO
```

**注意**: 
- 请确保 `OPENAI_API_KEY` 是 Kimi 的 API 密钥
- 如需使用 DeepSeek，请修改 `OPENAI_BASE_URL` 为 `https://api.deepseek.com/v1`

### 2.4 运行验证
进入项目根目录后，尝试启动入口脚本以确认环境就绪。
```
python next_trading_day_invest_report_mysql.py
```
若入口不同，请以实际入口为准。跑起来后检查控制台输出与日志。

## 3. 环境变量与配置（深入）
- 变量命名和来源请以代码中读取方式为准，默认优先级通常是环境变量 > .env 文件 > 代码内默认值。
- 如使用 Alembic 进行数据库迁移，请确保数据库连接可用，迁移脚本位置与执行命令按项目约定执行。

## 4. 运行入口定位
- 若仓库中存在 `if __name__ == "__main__":` 的入口块，通常对应命令行启动脚本。
- 常见入口点：
  - main.py → `python next_trading_day_invest_report_mysql.py`
- 若不确定入口，请搜索项目中的入口模式：
  - 使用文本搜索查找 `if __name__ ==` 或 `def main(` 等关键词。

## 5. 数据库与迁移
- 数据库连接字符串通常放在 `DATABASE_URL` 环境变量中，例如：`mysql+pymysql://user:password@host:3306/dbname`。
- 如使用 SQLAlchemy：创建引擎、Session、以及 ORM 模型。
- 迁移工具（如 Alembic）请按项目已有的迁移方案执行初始化与迁移。

## 6. 测试
- 测试框架：若使用 pytest，请在虚拟环境中执行：
```
pytest
```
- 确保测试数据库/外部服务的访问凭证已正确配置，测试环境独立于生产环境。

## 7. 部署与运维
- 本地开发：使用虚拟环境，确保依赖锁定，配置完毕后启动应用。
- 生产部署：可考虑 Docker/容器化部署、或在云服务器直接部署。确保日志、健康检查、以及数据库连接池配置合理。
- 依赖管理：尽量锁定版本（如 requirements.txt），并定期在 CI 中执行依赖更新与测试。

## 8. 维护与贡献
- 代码风格遵循项目现有规范，提交前请运行测试。
- 如需开发新功能，请更新本手册以覆盖新的运行方式与依赖。

## 9. 变更日志

### 2026-03-16 更新
- AI 模型从 DeepSeek 迁移至 Kimi2
- 修复多个 Bug（缓存过期、数据库连接等）
- 新增模块化代码结构（config.py、utils.py、data_fetcher.py、analyzer.py）

### 历史版本
- 初始版本基于 DeepSeek API

如需调整或扩展，请告诉我你希望加入的入口点信息、部署方式或 CI/CD 工作流，我可以追加到此 README 中。

## 10. 掘金量化安装与 Token 获取
### 10.1 安装
请确保安装掘金量化的最新SDK版本：
```
python.exe -m pip install gm -i https://mirrors.aliyun.com/pypi/simple/ -U
```

**注意：** 掘金量化已于2024年9月30日正式下线老版数据API，请确保安装 gm>=3.0.148 版本以使用新版API。

### 10.2 获取 Token
1) 登陆掘金量化客户端。
2) 进入系统设置。
3) 找到秘钥管理（Tokens）。
4) 复制生成的 Token，妥善保存。
5) 将 Token 配置到环境变量中，例如在 .env：
```
GM_API_TOKEN=your_token_here
```
6) 在代码中读取 Token，例如：
```python
import os
token = os.getenv('GM_API_TOKEN')
```

### 10.3 新版API说明
掘金量化已更新至新版API，主要变更：

| 旧API（已下线） | 新API |
|---|---|
| `get_instruments` | `get_symbols` |
| `get_next_trading_date` | `get_next_n_trading_dates` |
| `history_n` | `get_history_symbol` |
| `stk_get_fundamentals_balance` | `stk_get_fundamentals_balance_pt` |
| `stk_get_fundamentals_income` | `stk_get_fundamentals_income_pt` |
| `stk_get_finance_deriv` | `stk_get_finance_deriv_pt` |
| `stk_get_finance_prime` | `stk_get_finance_prime_pt` |
| `stk_get_daily_valuation` | `stk_get_daily_valuation_pt` |

### 10.4 使用
- 运行项目必须保证掘金量化客户端已登录，仿真股票账户已连接。
- 确保使用新版API调用方式。

### 10.5 常见问题
- 安装失败：检查 Python 版本、网络、以及代理设置。
- 掘金量化相关API调用失败：检查 Token 是否正确配置，以及账户是否有足够权限。
- 确保安装了最新版本的 gm SDK（>=3.0.148）。

如需更详细的API使用示例，请参考掘金量化官方文档：https://www.myquant.cn/docs2/sdk/python/
