"""
配置文件 - 集中管理所有配置参数
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class APIConfig:
    """API配置"""
    openai_api_key: str = ""
    openai_base_url: str = "https://api.deepseek.com/v1"
    gm_api_token: str = ""
    
    def __post_init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", self.openai_base_url)
        self.gm_api_token = os.getenv("GM_API_TOKEN", self.gm_api_token)


@dataclass
class DBConfig:
    """数据库配置"""
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    name: str = "stock_base"
    
    def __post_init__(self):
        self.host = os.getenv("DB_HOST", self.host)
        self.port = int(os.getenv("DB_PORT", self.port))
        self.user = os.getenv("DB_USER", self.user)
        self.password = os.getenv("DB_PASSWORD", self.password)
        self.name = os.getenv("DB_NAME", self.name)
    
    @property
    def connection_string(self) -> str:
        return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class CacheConfig:
    """缓存配置"""
    cache_dir: str = "./cache"
    expire_seconds: int = 86400  # 24小时
    
    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)


@dataclass
class FilterConfig:
    """股票筛选配置"""
    max_market_cap: float = 100e8  # 100亿
    min_roe: float = 0.12  # 12%
    min_profit_growth: float = 0.15  # 15%
    max_debt_ratio: float = 0.60  # 60%
    min_listing_days: int = 365  # 上市满1年
    max_stocks: int = 10


@dataclass
class AppConfig:
    """应用配置"""
    api: APIConfig = None
    db: DBConfig = None
    cache: CacheConfig = None
    filter: FilterConfig = None
    log_level: str = "INFO"
    log_file: str = "./cache/invest_agent.log"
    
    def __post_init__(self):
        if self.api is None:
            self.api = APIConfig()
        if self.db is None:
            self.db = DBConfig()
        if self.cache is None:
            self.cache = CacheConfig()
        if self.filter is None:
            self.filter = FilterConfig()


# 全局配置实例
config = AppConfig()
