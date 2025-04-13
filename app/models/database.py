import sqlite3
import logging
from datetime import datetime
from app.config.config import Config

logger = logging.getLogger('image_sorting')

def init_db():
    """初始化数据库"""
    conn = sqlite3.connect(Config.DATABASE)
    cursor = conn.cursor()
    
    # 创建图片表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        upload_time TIMESTAMP NOT NULL,
        rating INTEGER DEFAULT 0,
        rating_count INTEGER DEFAULT 0,
        original_name TEXT NOT NULL
    )
    ''')
    
    # 创建评分记录表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS rating_records (
        id TEXT PRIMARY KEY,
        image_id TEXT NOT NULL,
        score INTEGER NOT NULL,
        rating_time TIMESTAMP NOT NULL,
        FOREIGN KEY (image_id) REFERENCES images(id)
    )
    ''')
    
    # 创建排序结果表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sorting_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        result_json TEXT NOT NULL,
        has_duplicates INTEGER NOT NULL,
        duplicate_count INTEGER NOT NULL,
        created_at TIMESTAMP NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("数据库初始化完成")

def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(Config.DATABASE)
    conn.row_factory = sqlite3.Row
    return conn 