# 数据库配置说明

## MySQL数据库配置

### 1. 创建MySQL用户和数据库

如果您使用的是本地MySQL，请执行以下步骤：

```sql
-- 登录MySQL
mysql -u root -p

-- 创建数据库
CREATE DATABASE IF NOT EXISTS db_quant CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 创建用户并授权
CREATE USER 'stock_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON db_quant.* TO 'stock_user'@'localhost';
FLUSH PRIVILEGES;
```

### 2. 配置环境变量

修改 `.env` 文件：

```env
# MySQL数据库配置
DB_HOST=localhost
DB_PORT=3306
DB_USER=stock_user
DB_PASSWORD=your_password
DB_NAME=db_quant
```

### 3. 替换为您的实际配置

- `DB_USER`: MySQL用户名
- `DB_PASSWORD`: MySQL用户密码
- `DB_HOST`: MySQL服务器地址（默认localhost）
- `DB_PORT`: MySQL端口（默认3306）
- `DB_NAME`: 数据库名称（默认db_quant）

## 故障排除

### 1. 访问被拒绝错误
如果出现 `Access denied` 错误，请检查：
- 用户名和密码是否正确
- 用户是否有访问数据库的权限
- MySQL服务是否正在运行

### 2. 数据库不存在错误
如果数据库不存在，程序会尝试自动创建，但需要用户有创建数据库的权限。

### 3. 连接超时错误
检查：
- MySQL服务是否正在运行
- 主机地址和端口是否正确
- 防火墙设置

## 无数据库模式

如果不想使用数据库功能，程序仍会正常运行，只是不会将报告保存到数据库中。您会在控制台看到相应的提示信息。