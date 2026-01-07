# 数据库功能说明

## 功能概述

[next_trading_day_invest_report.py](file:///E:/python/lianghua/AI/AIGC/next_trading_day_invest_report.py) 脚本现在支持将生成的投资分析报告和选中的股票信息保存到MySQL数据库中。

## 数据库表结构

### 1. investment_reports (投资报告表)
- `id`: 主键，自增ID
- `report_date`: 报告日期
- `report_content`: 报告内容
- `created_at`: 创建时间
- `updated_at`: 更新时间

### 2. selected_stocks (选中股票表)
- `id`: 主键，自增ID
- `report_date`: 报告日期
- `stock_code`: 股票代码
- `stock_name`: 股票名称
- `buy_price`: 买入价格
- `sell_price`: 卖出价格
- `buy_date`: 买入日期
- `created_at`: 创建时间

## 配置说明

### 1. 环境变量配置
在 `.env` 文件中配置数据库连接信息：

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=db_quant
```

### 2. MySQL数据库设置
请参考 [DB_SETUP.md](file:///E:/python/lianghua/AI/AIGC/DB_SETUP.md) 文件中的详细配置说明。

### 3. 依赖安装
运行以下命令安装MySQL相关依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

1. 确保MySQL服务已启动
2. 根据需要修改 `.env` 文件中的数据库连接信息
3. 运行 [next_trading_day_invest_report.py](file:///E:/python/lianghua/AI/AIGC/next_trading_day_invest_report.py) 脚本
4. 程序会自动创建数据库表（如果不存在）
5. 生成的投资报告和选中的股票信息将自动保存到数据库

## 数据提取逻辑

程序会从AI生成的投资报告文本中自动提取：
- 股票代码和名称
- 买入价格（通过正则表达式匹配"买入价"、"目标价"等关键词）
- 卖出价格（通过正则表达式匹配"卖出价"、"目标卖价"等关键词）

## 错误处理

如果数据库配置不正确或无法连接，程序会：
- 显示错误信息
- 继续执行其他功能
- 不影响主要的报告生成功能

## 注意事项

1. 确保MySQL服务正常运行
2. 确保数据库用户有创建数据库和表的权限
3. 如果数据库不存在，程序会尝试自动创建
4. 股票价格信息通过文本解析提取，格式可能因AI输出格式变化而需要调整
5. 如果不配置数据库或配置错误，程序仍会正常生成报告，只是不会保存到数据库