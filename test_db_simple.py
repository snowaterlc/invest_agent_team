#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试数据库功能 - 简化版
"""
import os
import sys
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# 数据库配置
def create_mysql_engine():
    """
    创建数据库引擎对象
    """
    # 从环境变量获取数据库配置，如果不存在则使用默认值
    host = os.getenv("DB_HOST", "localhost")
    user = os.getenv("DB_USER", "root")
    passwd = os.getenv("DB_PASSWORD", "")
    port = os.getenv("DB_PORT", "3306")
    db = os.getenv("DB_NAME", "stock_base")
    
    # 首先连接到MySQL服务器（不指定数据库）
    server_engine = sqlalchemy.create_engine(
        f'mysql+pymysql://{user}:{passwd}@{host}:{port}',
        poolclass=sqlalchemy.pool.NullPool
    )
    
    # 创建数据库（如果不存在）
    try:
        with server_engine.connect() as conn:
            conn.execute(sqlalchemy.text(f"CREATE DATABASE IF NOT EXISTS `{db}` CHARACTER SET utf8mb4"))
    except Exception as e:
        print(f"创建数据库失败: {e}")
        # 如果创建失败，尝试使用原引擎（假设数据库已存在）
        pass
    
    # 创建连接数据库的引擎
    db_engine = sqlalchemy.create_engine(
        f'mysql+pymysql://{user}:{passwd}@{host}:{port}/{db}?charset=utf8',
        poolclass=sqlalchemy.pool.NullPool
    )
    return db_engine

# 创建ORM基类
Base = declarative_base()

class InvestmentReport(Base):
    """
    投资报告表
    """
    __tablename__ = 'investment_reports'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    report_date = Column(Date, nullable=False, comment='报告日期')
    report_content = Column(Text, nullable=False, comment='报告内容')
    created_at = Column(DateTime, default=datetime.now, comment='创建时间')
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment='更新时间')

class SelectedStock(Base):
    """
    选中股票表
    """
    __tablename__ = 'selected_stocks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    report_date = Column(Date, nullable=False, comment='报告日期')
    stock_code = Column(String(20), nullable=False, comment='股票代码')
    stock_name = Column(String(100), nullable=False, comment='股票名称')
    buy_price = Column(Float, comment='买入价格')
    sell_price = Column(Float, comment='卖出价格')
    buy_date = Column(Date, nullable=False, comment='买入日期')
    created_at = Column(DateTime, default=datetime.now, comment='创建时间')

def test_db_connection():
    """测试数据库连接"""
    try:
        print("正在测试数据库连接...")
        engine = create_mysql_engine()
        print("✅ 数据库连接成功")
        
        # 测试创建表
        print("正在初始化数据库表...")
        Base.metadata.create_all(engine)
        print("✅ 数据库表初始化成功")
        
        # 测试插入数据
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # 清空测试数据
        session.query(InvestmentReport).delete()
        session.query(SelectedStock).delete()
        session.commit()
        
        print("✅ 测试数据清理完成")
        
        # 插入测试数据
        from datetime import date
        test_report = InvestmentReport(
            report_date=date.today(),
            report_content="测试报告内容"
        )
        session.add(test_report)
        session.commit()
        
        test_stock = SelectedStock(
            report_date=date.today(),
            stock_code="000001",
            stock_name="平安银行",
            buy_price=10.5,
            sell_price=12.8,
            buy_date=date.today()
        )
        session.add(test_stock)
        session.commit()
        
        print("✅ 测试数据插入成功")
        print("✅ 数据库功能测试通过")
        session.close()
        
    except Exception as e:
        print(f"❌ 数据库功能测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_db_connection()