# AIGC 项目使用手册

本项目基于 Python，依赖项来自 requirements.txt，包含 crewai、crewa i-tools、langchain、langchain-openai、akshare、gm、python-dotenv、pandas、numpy、pymysql、sqlalchemy 等。本文档旨在帮助开发者快速搭建环境、配置依赖、运行程序以及常见维护工作。

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
常用变量示例：
- OPENAI_API_KEY：OpenAI 密钥
- DATABASE_URL：数据库连接字符串，例如 `mysql+pymysql://user:password@host:port/dbname`
- LOG_LEVEL：日志等级
- APP_HOST、APP_PORT：应用绑定地址与端口

可以将变量写入 `.env` 文件，项目若使用 python-dotenv 或自定义加载，将自动读取。示例：
```
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/aigc
LOG_LEVEL=INFO
APP_HOST=0.0.0.0
APP_PORT=8000
```

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
- 初版手册，基于当前仓库依赖与常见用法编写。

如需调整或扩展，请告诉我你希望加入的入口点信息、部署方式或 CI/CD 工作流，我可以追加到此 README 中。

## 10. 掘金量化安装与 Token 获取
### 10.1 安装
请将实际的 Python 包名替换为你们使用的包名，下面给出通用示例：
```
python.exe -m pip install gm -i https://mirrors.aliyun.com/pypi/simple/ -U
```

### 10.2 获取 Token
1) 登陆掘金量化客户端。
2) 进入系统设置。
3) 找到秘钥管理（Tokens）。
4) 复制生成的 Token，妥善保存。
5) 将 Token 配置到环境变量中，例如在 .env：
```
JUEJIN_TOKEN=your_token_here
```
6) 在代码中读取 Token，例如：
```python
import os
token = os.getenv('JUEJIN_TOKEN')
```

### 10.3 使用
- 运行项目必须保证掘金量化客户端已登录，仿真股票账户已连接。

### 10.4 常见问题
- 安装失败：检查 Python 版本、网络、以及代理设置。
- 掘金量化相关api 调用失败：检查 Token 是否正确配置，以及账户是否有足够权限。
- 检查掘金量化api是否最新版本。

如需，我可以把实际包名替换为你仓库中实际使用的包名，并补充更具体的调用示例。
