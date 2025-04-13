import os

class Config:
    # 基础配置
    UPLOAD_FOLDER = 'static/uploads'
    DATABASE = 'images.db'
    DEBUG_LEVEL = 'INFO'
    
    # 确保上传目录存在
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Jinja2模板配置
    JINJA2_CONFIG = {
        'variable_start_string': '{$',
        'variable_end_string': '$}',
        'block_start_string': '{%',
        'block_end_string': '%}',
        'comment_start_string': '{#',
        'comment_end_string': '#}'
    } 