import os
import sqlite3
import uuid
import math
import random
import json
import numpy as np
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from datetime import datetime
from flask_cors import CORS  # 导入CORS
from collections import defaultdict, deque

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('image_sorting')

app = Flask(__name__)
# 修改Jinja2模板语法，避免与Vue.js冲突
app.jinja_env.variable_start_string = '{$'
app.jinja_env.variable_end_string = '$}'
app.jinja_env.block_start_string = '{%'
app.jinja_env.block_end_string = '%}'
app.jinja_env.comment_start_string = '{#'
app.jinja_env.comment_end_string = '#}'

CORS(app)  # 启用CORS，允许所有来源的跨域请求
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATABASE'] = 'images.db'
app.config['DEBUG_LEVEL'] = 'INFO'  # 可选: DEBUG, INFO, WARNING, ERROR

# 定义全局变量来存储传递闭包矩阵和排序状态
TRANSITIVE_CLOSURE = {
    'is_initialized': False,
    'matrix': None,
    'id_to_idx': {},
    'idx_to_id': []
}

SORTING_STATE = {
    'is_initialized': False,
    'image_count': 0,
    'total_comparisons': 0,
    'clear_relation_ratio': 0,
    'last_update_time': None
}

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化数据库
def init_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
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

# 在应用启动时初始化数据库
@app.before_request
def before_request():
    if not os.path.exists(app.config['DATABASE']):
        init_db()

# 首页
@app.route('/')
def index():
    return render_template('index.html')

# 上传图片
@app.route('/upload', methods=['POST'])
def upload_images():
    if 'images' not in request.files:
        return jsonify({'error': '没有找到图片文件'}), 400
    
    files = request.files.getlist('images')
    uploaded_files = []
    
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    
    for file in files:
        if file.filename == '':
            continue
        
        # 生成唯一文件名
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = str(uuid.uuid4()) + file_ext
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # 保存文件
        file.save(file_path)
        
        # 保存到数据库
        image_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO images (id, filename, upload_time, original_name) VALUES (?, ?, ?, ?)",
            (image_id, unique_filename, datetime.now(), file.filename)
        )
        
        uploaded_files.append({
            'id': image_id,
            'filename': unique_filename,
            'original_name': file.filename
        })
    
    conn.commit()
    conn.close()
    
    # 重置排序状态，因为有新图片加入
    reset_sorting_state()
    
    logger.info(f"成功上传 {len(uploaded_files)} 张图片")
    
    return jsonify({
        'success': True,
        'files': uploaded_files
    })

# 重置排序状态
def reset_sorting_state():
    global TRANSITIVE_CLOSURE, SORTING_STATE
    
    TRANSITIVE_CLOSURE = {
        'is_initialized': False,
        'matrix': None,
        'id_to_idx': {},
        'idx_to_id': []
    }
    
    SORTING_STATE = {
        'is_initialized': False,
        'image_count': 0,
        'total_comparisons': 0,
        'clear_relation_ratio': 0,
        'last_update_time': None
    }
    
    logger.debug("排序状态已重置")

# 更新传递闭包矩阵
def update_transitive_closure(graph, all_image_ids):
    """
    使用Floyd-Warshall算法高效更新传递闭包矩阵
    
    参数:
    - graph: 图片关系有向图
    - all_image_ids: 所有图片ID列表
    
    返回:
    - tc_matrix: 更新后的传递闭包矩阵
    """
    global TRANSITIVE_CLOSURE, SORTING_STATE
    
    n = len(all_image_ids)
    id_to_idx = {img_id: i for i, img_id in enumerate(all_image_ids)}
    
    # 初始化传递闭包矩阵
    tc_matrix = np.zeros((n, n), dtype=bool)
    
    # 根据当前图填充初始关系
    for node, neighbors in graph.items():
        if node in id_to_idx:  # 确保节点在映射中
            i = id_to_idx[node]
            for neighbor in neighbors:
                if neighbor in id_to_idx:  # 确保邻居在映射中
                    j = id_to_idx[neighbor]
                    tc_matrix[i, j] = True
    
    # Floyd-Warshall算法
    for k in range(n):
        for i in range(n):
            if tc_matrix[i, k]:
                for j in range(n):
                    if tc_matrix[k, j]:
                        tc_matrix[i, j] = True
    
    # 计算关系明确度
    total_pairs = n * (n - 1) // 2
    clear_pairs = 0
    
    for i in range(n):
        for j in range(i+1, n):
            if tc_matrix[i, j] or tc_matrix[j, i]:  # 如果i→j或j→i存在路径
                clear_pairs += 1
    
    # 更新排序状态
    clear_relation_ratio = clear_pairs / total_pairs if total_pairs > 0 else 0
    SORTING_STATE.update({
        'is_initialized': True,
        'image_count': n,
        'clear_relation_ratio': clear_relation_ratio,
        'last_update_time': datetime.now()
    })
    
    # 存储映射和传递闭包
    TRANSITIVE_CLOSURE.update({
        'is_initialized': True,
        'matrix': tc_matrix,
        'id_to_idx': id_to_idx,
        'idx_to_id': all_image_ids
    })
    
    logger.debug(f"传递闭包矩阵已更新: {n}个节点, 关系明确度: {clear_relation_ratio:.2f}")
    
    return tc_matrix

# 检查图中是否存在从start到end的路径 - 使用传递闭包矩阵
def has_path(graph, start, end):
    """检查图中是否存在从start到end的路径，使用缓存的传递闭包矩阵"""
    global TRANSITIVE_CLOSURE
    
    # 如果传递闭包矩阵已初始化，直接查询
    if TRANSITIVE_CLOSURE['is_initialized']:
        id_to_idx = TRANSITIVE_CLOSURE['id_to_idx']
        if start in id_to_idx and end in id_to_idx:
            i = id_to_idx[start]
            j = id_to_idx[end]
            return TRANSITIVE_CLOSURE['matrix'][i, j]
    
    # 否则使用有限深度的BFS
    if start == end:
        return True
    
    # 记录搜索深度，避免过深搜索
    max_depth = 4  # 减小搜索深度，提高效率
    
    # 使用BFS查找更短的路径
    visited = set([start])
    queue = deque([(start, 0)])  # (节点, 深度)
    
    while queue:
        current, depth = queue.popleft()
        
        # 超过最大深度则不再搜索
        if depth >= max_depth:
            continue
        
        # 检查当前节点的所有邻居
        for neighbor in graph[current]:
            if neighbor == end:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    
    return False

# 检查是否已经达到足够好的排序
def is_sorting_sufficient():
    """
    检查当前排序是否已经足够好，可以提前终止
    
    返回:
    - (bool): 是否已达到足够好的排序
    - (str): 原因描述
    """
    global SORTING_STATE
    
    if not SORTING_STATE['is_initialized']:
        return False, "排序尚未初始化"
    
    # 获取当前排序中是否存在并列项的信息
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    
    # 查询最新的排序结果
    cursor.execute("""
        SELECT has_duplicates, duplicate_count 
        FROM sorting_results 
        ORDER BY created_at DESC LIMIT 1
    """)
    
    result = cursor.fetchone()
    conn.close()
    
    # 如果存在并列项且数量超过5%的图片，则不认为排序充分
    has_duplicates = False
    duplicate_count = 0
    
    if result:
        has_duplicates = bool(result[0])
        duplicate_count = result[1]
        
        # 如果并列图片占比超过5%，则认为排序不充分
        if has_duplicates and SORTING_STATE['image_count'] > 0:
            duplicate_ratio = duplicate_count / SORTING_STATE['image_count']
            if duplicate_ratio > 0.05:
                return False, f"存在{duplicate_count}张并列图片 ({duplicate_ratio:.1%})"
    
    # 策略1: 关系明确度达到阈值
    if SORTING_STATE['clear_relation_ratio'] >= 0.9:
        return True, "90%的图片对关系已明确"
    
    # 策略2: 比较次数已经足够多
    avg_comparisons_per_image = (SORTING_STATE['total_comparisons'] / SORTING_STATE['image_count'] 
                               if SORTING_STATE['image_count'] > 0 else 0)
    if avg_comparisons_per_image >= 5 and SORTING_STATE['image_count'] > 10:
        return True, f"平均每张图片已参与{avg_comparisons_per_image:.1f}次比较"
    
    # 默认继续排序
    return False, "排序尚未完成"

# 获取需要评分的图片
@app.route('/get_images_to_rate', methods=['GET'])
def get_images_to_rate():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 获取所有图片基本信息
    cursor.execute("SELECT id, filename, rating, rating_count, original_name FROM images")
    rows = cursor.fetchall()
    
    if not rows:
        conn.close()
        return jsonify({
            'images': [],
            'task_type': 'complete'
        })
    
    # 将所有图片信息存储到字典中
    all_images = {}
    all_image_ids = []
    for row in rows:
        image_id = row['id']
        all_image_ids.append(image_id)
        all_images[image_id] = {
            'id': image_id,
            'filename': row['filename'],
            'url': f"/static/uploads/{row['filename']}",
            'rating': row['rating'],
            'rating_count': row['rating_count'],
            'original_name': row['original_name']
        }
    
    # 获取评分记录，用于构建有向图
    cursor.execute("""
        SELECT image_id, score, rating_time 
        FROM rating_records 
        ORDER BY rating_time
    """)
    rating_records = cursor.fetchall()
    
    # 构建初始有向图
    graph = defaultdict(set)  # 存储谁指向谁的关系
    
    # 处理评分记录，将同一组的评分整理在一起
    rating_groups = []
    current_group = []
    current_time = None
    
    for record in rating_records:
        image_id = record[0]
        score = record[1]
        rating_time = record[2]
        
        # 如果时间差距小于1秒，认为是同一组评分
        if current_time is None or abs((datetime.fromisoformat(rating_time) - datetime.fromisoformat(current_time)).total_seconds()) < 1:
            current_group.append((image_id, score))
            current_time = rating_time
        else:
            if current_group:
                rating_groups.append(current_group)
            current_group = [(image_id, score)]
            current_time = rating_time
    
    # 添加最后一组
    if current_group:
        rating_groups.append(current_group)
    
    # 基于评分组构建有向图
    for group in rating_groups:
        # 按评分排序：1(喜欢) > 0(一般) > -1(不喜欢)
        sorted_group = sorted(group, key=lambda x: x[1], reverse=True)
        
        # 创建有向边: 评分高的图片指向评分低的图片（表示"胜出"）
        for i in range(len(sorted_group)):
            for j in range(i+1, len(sorted_group)):
                if sorted_group[i][1] > sorted_group[j][1]:  # 只有当评分确实更高时才创建边
                    higher_node = sorted_group[i][0]
                    lower_node = sorted_group[j][0]
                    graph[higher_node].add(lower_node)
    
    # 更新传递闭包矩阵
    update_transitive_closure(graph, all_image_ids)
    
    # 更新排序状态
    global SORTING_STATE
    SORTING_STATE['total_comparisons'] = len(rating_groups)
    
    # 检查是否已经达到足够好的排序
    is_sufficient, reason = is_sorting_sufficient()
    if is_sufficient:
        logger.info(f"排序完成: {reason}")
        conn.close()
        return jsonify({
            'images': [],
            'task_type': 'complete',
            'message': f"排序已完成: {reason}"
        })
    
    # 选择下一组需要评分的图片 - 优化选择策略
    images_to_rate = select_optimal_image_group(all_images, graph, conn)
    
    # 如果没有找到需要评分的图片（可能所有关系都已建立），返回空数组
    if not images_to_rate:
        conn.close()
        return jsonify({
            'images': [],
            'task_type': 'complete'
        })
    
    conn.close()
    
    # 记录本次选择的图片
    logger.debug(f"选择了 {len(images_to_rate)} 张图片供评分: {[img['id'][:8] for img in images_to_rate]}")
    
    return jsonify({
        'images': images_to_rate,
        'task_type': 'build_graph',
        'sorting_progress': {
            'clear_relation_ratio': f"{SORTING_STATE['clear_relation_ratio']*100:.1f}%",
            'total_comparisons': SORTING_STATE['total_comparisons']
        }
    })

# 优化选择下一组需要评分的图片
def select_optimal_image_group(all_images, graph, conn):
    """
    智能选择下一组需要评分的图片，以最高效地构建完整的有向图
    
    策略:
    1. 优先选择并列的图片对
    2. 优先选择未评分的图片
    3. 优先选择边界区域的关键图片对（关系不明确的图片）
    """
    global TRANSITIVE_CLOSURE, SORTING_STATE
    
    # 获取已评分和未评分的图片ID
    rated_images = set()
    unrated_images = set()
    
    for img_id, img_info in all_images.items():
        if img_info['rating_count'] > 0:
            rated_images.add(img_id)
        else:
            unrated_images.add(img_id)
    
    selected_images = []
    
    # 检查是否有最近的排序结果
    cursor = conn.cursor()
    cursor.execute("""
        SELECT result_json FROM sorting_results 
        ORDER BY created_at DESC LIMIT 1
    """)
    row = cursor.fetchone()
    
    # 策略0: 优先选择并列的图片
    if row and row[0]:
        try:
            result_data = json.loads(row[0])
            if 'has_duplicate_ratings' in result_data and result_data['has_duplicate_ratings']:
                duplicate_images = [img for img in result_data['images'] if img.get('has_duplicate', False)]
                
                # 从并列图片中优先选择相邻的图片对
                if len(duplicate_images) >= 3:
                    # 选择前3个并列图片
                    for i in range(min(3, len(duplicate_images))):
                        selected_images.append(all_images[duplicate_images[i]['id']])
                    logger.info("选择了3张并列图片进行评分")
                    return selected_images
                elif len(duplicate_images) > 0:
                    # 将所有并列图片添加到选择列表
                    for img in duplicate_images:
                        if len(selected_images) < 3 and img['id'] in all_images:
                            selected_images.append(all_images[img['id']])
        except Exception as e:
            logger.error(f"解析排序结果时出错: {str(e)}")
    
    # 策略1: 如果有未评分的图片，优先选择这些
    if len(unrated_images) >= 3:
        # 从未评分图片中选择3张
        selected_ids = random.sample(list(unrated_images), 3)
        for img_id in selected_ids:
            selected_images.append(all_images[img_id])
        logger.info("选择了3张未评分图片")
        return selected_images
    
    # 如果未评分图片不足3张但有一些，将它们添加到选择列表
    if unrated_images:
        for img_id in unrated_images:
            selected_images.append(all_images[img_id])
        logger.info(f"添加了 {len(unrated_images)} 张未评分图片")
    
    # 如果已经初始化了传递闭包矩阵，使用它来选择图片
    if TRANSITIVE_CLOSURE['is_initialized'] and TRANSITIVE_CLOSURE['matrix'] is not None:
        tc_matrix = TRANSITIVE_CLOSURE['matrix']
        id_to_idx = TRANSITIVE_CLOSURE['id_to_idx']
        idx_to_id = TRANSITIVE_CLOSURE['idx_to_id']
        
        # 策略2: 优先选择边界图片
        # 边界图片是指那些与其他图片关系不明确且可能是相邻排名的图片
        
        # 计算近似排名
        approx_ranks = estimate_ranks_from_tc(tc_matrix, idx_to_id)
        
        # 计算边界分数
        boundary_scores = []
        n = len(idx_to_id)
        
        # 先找到每个节点的"不确定关系比例"
        uncertainty_scores = []
        for i in range(n):
            node_id = idx_to_id[i]
            
            # 跳过已选择的图片
            if node_id in [img['id'] for img in selected_images]:
                continue
                
            uncertain_relations = 0
            nearby_uncertain = 0  # 排名接近但关系不确定的节点数
            
            for j in range(n):
                if i == j:
                    continue
                    
                # 检查关系是否已确定
                is_certain = tc_matrix[i, j] or tc_matrix[j, i]
                
                if not is_certain:
                    uncertain_relations += 1
                    
                    # 检查排名是否接近
                    rank_diff = abs(approx_ranks[i] - approx_ranks[j])
                    if rank_diff < n / 10:  # 排名差距小于10%认为接近
                        nearby_uncertain += 1
            
            # 边界分数考虑两个因素：整体不确定性和与排名接近节点的不确定性
            # 优先选择排名接近但关系不确定的节点
            if uncertain_relations > 0:
                boundary_score = (0.3 * uncertain_relations / (n - 1) + 
                                0.7 * nearby_uncertain / max(1, uncertain_relations))
                uncertainty_scores.append((node_id, boundary_score))
        
        # 按边界分数降序排序
        uncertainty_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 优先选择锦标赛得分相近但关系不确定的图片对
        for i, (id1, _) in enumerate(uncertainty_scores[:10]):  # 只考虑前10个高分节点
            if len(selected_images) >= 3:
                break
                
            # 将这个节点添加到选择列表
            if id1 not in [img['id'] for img in selected_images]:
                selected_images.append(all_images[id1])
                
            # 尝试找到与该节点关系不确定的最佳配对
            for j, (id2, _) in enumerate(uncertainty_scores):
                if i == j or len(selected_images) >= 3:
                    continue
                    
                # 检查id1和id2的关系是否不确定
                idx1 = id_to_idx[id1]
                idx2 = id_to_idx[id2]
                if not (tc_matrix[idx1, idx2] or tc_matrix[idx2, idx1]):
                    # 关系不确定，添加这个节点
                    if id2 not in [img['id'] for img in selected_images]:
                        selected_images.append(all_images[id2])
                        break
        
        # 如果已选择足够图片，返回结果
        if len(selected_images) >= 3:
            logger.info(f"使用边界图片策略选择了 {len(selected_images)} 张图片")
            return selected_images
        
        # 如果仍未选择足够图片，使用不确定性分数最高的图片
        for node_id, _ in uncertainty_scores:
            if len(selected_images) >= 3:
                break
                
            if node_id not in [img['id'] for img in selected_images]:
                selected_images.append(all_images[node_id])
        
        if len(selected_images) >= 3:
            logger.info("使用不确定性分数策略选择了图片")
            return selected_images
    
    # 备用策略：如果还不足3张，随机选择图片
    if len(selected_images) < 3:
        remaining_ids = [id for id in all_images.keys() 
                       if id not in [img['id'] for img in selected_images]]
                       
        if remaining_ids:
            sample_size = min(3 - len(selected_images), len(remaining_ids))
            random_ids = random.sample(remaining_ids, sample_size)
            
            for img_id in random_ids:
                selected_images.append(all_images[img_id])
            
            logger.info(f"随机补充了 {sample_size} 张图片")
    
    return selected_images

# 根据传递闭包矩阵估计图片排名
def estimate_ranks_from_tc(tc_matrix, idx_to_id):
    """
    根据传递闭包矩阵估计每个节点的大致排名
    
    参数:
    - tc_matrix: 传递闭包矩阵
    - idx_to_id: 索引到ID的映射
    
    返回:
    - 每个节点的估计排名（越小表示越靠前）
    """
    n = len(idx_to_id)
    
    # 计算每个节点的出度和入度
    out_degrees = np.sum(tc_matrix, axis=1)  # 行和
    in_degrees = np.sum(tc_matrix, axis=0)   # 列和
    
    # 计算tournament score（出度减入度）
    tournament_scores = out_degrees - in_degrees
    
    # 将分数从高到低排序获得排名
    rank_indices = np.argsort(-tournament_scores)
    ranks = np.zeros(n)
    
    for rank, idx in enumerate(rank_indices):
        ranks[idx] = rank
    
    return ranks

# 锦标赛排序（使用传递闭包矩阵）
def tournament_sort(image_ratings, conn):
    """
    使用传递闭包矩阵进行锦标赛排序，返回排序后的图片列表和是否有并列项
    
    参数:
    - image_ratings: 图片评分列表
    - conn: 数据库连接
    
    返回:
    - sorted_images: 排序后的图片列表
    - has_duplicates: 是否有并列评分的图片
    - duplicate_count: 并列图片的数量
    """
    logger.info(f"对 {len(image_ratings)} 张图片进行锦标赛排序")
    
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 获取评分记录，用于构建有向图
    cursor.execute("""
        SELECT image_id, score, rating_time 
        FROM rating_records 
        ORDER BY rating_time
    """)
    rating_records = cursor.fetchall()
    
    # 构建初始有向图
    graph = defaultdict(set)  # 存储谁指向谁的关系
    
    # 处理评分记录，将同一组的评分整理在一起
    rating_groups = []
    current_group = []
    current_time = None
    
    for record in rating_records:
        image_id = record[0]
        score = record[1]
        rating_time = record[2]
        
        # 如果时间差距小于1秒，认为是同一组评分
        if current_time is None or abs((datetime.fromisoformat(rating_time) - datetime.fromisoformat(current_time)).total_seconds()) < 1:
            current_group.append((image_id, score))
            current_time = rating_time
        else:
            if current_group:
                rating_groups.append(current_group)
            current_group = [(image_id, score)]
            current_time = rating_time
    
    # 添加最后一组
    if current_group:
        rating_groups.append(current_group)
    
    # 基于评分组构建有向图
    for group in rating_groups:
        # 按评分排序：1(喜欢) > 0(一般) > -1(不喜欢)
        sorted_group = sorted(group, key=lambda x: x[1], reverse=True)
        
        # 创建有向边: 评分高的图片指向评分低的图片（表示"胜出"）
        for i in range(len(sorted_group)):
            for j in range(i+1, len(sorted_group)):
                if sorted_group[i][1] > sorted_group[j][1]:  # 只有当评分确实更高时才创建边
                    higher_node = sorted_group[i][0]
                    lower_node = sorted_group[j][0]
                    graph[higher_node].add(lower_node)
    
    # 获取所有图片ID
    all_image_ids = [img['id'] for img in image_ratings]
    
    # 更新传递闭包矩阵
    update_transitive_closure(graph, all_image_ids)
    
    # 使用拓扑排序确定最终顺序
    sorted_images = topological_sort_with_tc(image_ratings)
    
    # 检测并列项
    sorted_images, has_duplicates, duplicate_count = detect_duplicates_with_tc(sorted_images)
    
    logger.info(f"锦标赛排序完成: {len(sorted_images)}张图片, {duplicate_count}张并列图片")
    
    return sorted_images, has_duplicates, duplicate_count

# 拓扑排序（使用传递闭包矩阵）
def topological_sort_with_tc(images):
    """
    使用传递闭包矩阵进行拓扑排序
    
    参数:
    - images: 图片信息列表
    
    返回:
    - 排序后的图片列表
    """
    global TRANSITIVE_CLOSURE
    
    if not TRANSITIVE_CLOSURE['is_initialized'] or TRANSITIVE_CLOSURE['matrix'] is None:
        # 如果传递闭包未初始化，进行简单的评分排序
        logger.warning("传递闭包未初始化，使用普通评分排序")
        return sorted(images, key=lambda x: (x['avg_rating'] if 'avg_rating' in x else 0), reverse=True)
    
    tc_matrix = TRANSITIVE_CLOSURE['matrix']
    id_to_idx = TRANSITIVE_CLOSURE['id_to_idx']
    idx_to_id = TRANSITIVE_CLOSURE['idx_to_id']
    
    n = len(idx_to_id)
    
    # 计算每个节点的胜/负场数、出入度和锦标赛分数
    img_stats = {}
    for img in images:
        img_id = img['id']
        if img_id not in id_to_idx:
            continue
            
        i = id_to_idx[img_id]
        
        # 计算胜场数(出度)和负场数(入度)
        wins = np.sum(tc_matrix[i, :])
        losses = np.sum(tc_matrix[:, i])
        
        # 计算锦标赛分数 (Tournament Score)
        tournament_score = wins - losses
        
        # 计算胜率
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.5
        
        # 保存统计信息
        img_stats[img_id] = {
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': win_rate,
            'out_degree': int(wins),
            'in_degree': int(losses),
            'tournament_score': float(tournament_score),
            'relative_strength': float(tournament_score) / n if n > 0 else 0  # 相对强度
        }
    
    # 计算"状态分数" - 考虑胜负场数的一个评分
    for img in images:
        img_id = img['id']
        if img_id in img_stats:
            stats = img_stats[img_id]
            status_score = stats['tournament_score']
            img['tournament_score'] = stats['tournament_score']
            img['status_score'] = status_score
            img['win_rate'] = stats['win_rate']
            img['wins'] = stats['wins']
            img['losses'] = stats['losses']
            img['relative_strength'] = stats['relative_strength']
        else:
            # 没有评分记录的图片
            img['tournament_score'] = 0
            img['status_score'] = 0
            img['win_rate'] = 0.5
            img['wins'] = 0
            img['losses'] = 0
            img['relative_strength'] = 0
    
    # 按状态分数降序排序
    sorted_images = sorted(images, key=lambda x: (
        x['status_score'] if 'status_score' in x else 0, 
        x['avg_rating'] if 'avg_rating' in x else 0
    ), reverse=True)
    
    # 检查是否有环（循环依赖）
    has_cycle = False
    for i in range(len(sorted_images) - 1):
        img1_id = sorted_images[i]['id']
        img2_id = sorted_images[i + 1]['id']
        
        if img1_id in id_to_idx and img2_id in id_to_idx:
            idx1 = id_to_idx[img1_id]
            idx2 = id_to_idx[img2_id]
            
            # 如果较低排名的图片可以到达较高排名的图片，说明有环
            if tc_matrix[idx2, idx1]:
                has_cycle = True
                break
    
    # 如果有环，使用更严格的排序算法
    if has_cycle:
        logger.warning("检测到排序中存在环，使用更严格的排序算法")
        # 使用锦标赛得分和平均评分的加权组合进行排序
        for img in images:
            tournament_score = img.get('tournament_score', 0)
            avg_rating = img.get('avg_rating', 0)
            img['combined_score'] = 0.7 * tournament_score + 0.3 * avg_rating * 10
        
        sorted_images = sorted(images, key=lambda x: x.get('combined_score', 0), reverse=True)
    
    return sorted_images

# 检测并列项（使用传递闭包矩阵）
def detect_duplicates_with_tc(sorted_images):
    """
    检测排序结果中的并列项
    
    参数:
    - sorted_images: 排序后的图片列表
    
    返回:
    - 标记了并列项的图片列表
    - 是否存在并列项
    - 并列项的数量
    """
    global TRANSITIVE_CLOSURE
    
    has_duplicates = False
    duplicate_count = 0
    
    if not TRANSITIVE_CLOSURE['is_initialized'] or len(sorted_images) <= 1:
        return sorted_images, has_duplicates, duplicate_count
    
    tc_matrix = TRANSITIVE_CLOSURE['matrix']
    id_to_idx = TRANSITIVE_CLOSURE['id_to_idx']
    
    # 检测相邻图片是否关系不明确
    for i in range(len(sorted_images) - 1):
        img1 = sorted_images[i]
        img2 = sorted_images[i + 1]
        
        img1_id = img1['id']
        img2_id = img2['id']
        
        # 检查两个图片是否都在传递闭包矩阵中
        if img1_id not in id_to_idx or img2_id not in id_to_idx:
            continue
            
        idx1 = id_to_idx[img1_id]
        idx2 = id_to_idx[img2_id]
        
        # 如果两个相邻图片之间没有明确的关系（即既没有idx1->idx2也没有idx2->idx1），标记为并列
        is_ambiguous = not (tc_matrix[idx1, idx2] or tc_matrix[idx2, idx1])
        
        # 另外，检查评分是否接近
        score_threshold = 0.1  # 相对强度差异阈值
        relative_strength1 = img1.get('relative_strength', 0)
        relative_strength2 = img2.get('relative_strength', 0)
        is_score_close = abs(relative_strength1 - relative_strength2) < score_threshold
        
        # 同时满足关系不明确和评分接近，则认为是并列
        if is_ambiguous and is_score_close:
            img1['has_duplicate'] = True
            img2['has_duplicate'] = True
            has_duplicates = True
            duplicate_count += 1
    
    # 再次计算实际的并列项数量
    duplicate_count = sum(1 for img in sorted_images if img.get('has_duplicate', False))
    
    return sorted_images, has_duplicates, duplicate_count

# 页面 - 查看排序结果
@app.route('/results')
def results():
    return render_template('results.html')

# 评分图片
@app.route('/rate', methods=['POST'])
def rate_images():
    data = request.json
    if not data or 'ratings' not in data:
        return jsonify({'error': '无效的评分数据'}), 400
    
    # 验证评分数据
    ratings = data['ratings']
    if not isinstance(ratings, list) or len(ratings) == 0:
        return jsonify({'error': '评分数据格式不正确'}), 400
    
    # 记录评分数据
    logger.info(f"收到 {len(ratings)} 条评分数据")
    
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    
    # 为同一组评分使用相同的时间戳
    rating_time = datetime.now()
    
    valid_ratings = 0
    for rating in ratings:
        image_id = rating.get('id')
        score = rating.get('score')  # 1: 喜欢, 0: 一般, -1: 不喜欢
        
        # 验证评分数据
        if not image_id or score is None or score not in [-1, 0, 1]:
            logger.warning(f"跳过无效评分: {rating}")
            continue
        
        # 检查图片是否存在
        cursor.execute("SELECT id FROM images WHERE id = ?", (image_id,))
        if not cursor.fetchone():
            logger.warning(f"评分的图片不存在: {image_id}")
            continue
        
        # 更新图片评分
        cursor.execute("""
            UPDATE images 
            SET rating = rating + ?,
                rating_count = rating_count + 1
            WHERE id = ?
        """, (score, image_id))
        
        # 添加评分记录
        record_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO rating_records (id, image_id, score, rating_time)
            VALUES (?, ?, ?, ?)
        """, (record_id, image_id, score, rating_time))
        
        valid_ratings += 1
    
    conn.commit()
    
    # 更新排序状态
    global SORTING_STATE
    SORTING_STATE['total_comparisons'] += 1
    
    # 计算并保存新的排序结果
    update_sorting_results(conn)
    
    # 检查是否已达到足够好的排序
    is_sufficient, reason = is_sorting_sufficient()
    
    conn.close()
    
    logger.info(f"成功处理 {valid_ratings} 条有效评分")
    
    return jsonify({
        'success': True,
        'processed_ratings': valid_ratings,
        'sorting_progress': {
            'clear_relation_ratio': f"{SORTING_STATE['clear_relation_ratio']*100:.1f}%",
            'total_comparisons': SORTING_STATE['total_comparisons'],
            'is_sorting_finished': is_sufficient,
            'finish_reason': reason if is_sufficient else "排序进行中"
        }
    })

# 计算并更新排序结果
def update_sorting_results(conn):
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 查询所有图片及其最后评分时间
    cursor.execute("""
        SELECT i.id, i.filename, i.rating, i.rating_count, i.upload_time,
               MAX(r.rating_time) as last_rating_time
        FROM images i
        LEFT JOIN rating_records r ON i.id = r.image_id
        GROUP BY i.id
    """)
    
    images = []
    rows = cursor.fetchall()
    
    for row in rows:
        image_id = row['id']
        rating = row['rating']
        rating_count = row['rating_count']
        
        # 计算平均分
        avg_rating = 0 if rating_count == 0 else rating / rating_count
        
        images.append({
            'id': image_id,
            'filename': row['filename'],
            'url': f"/static/uploads/{row['filename']}",
            'rating': rating,
            'rating_count': rating_count,
            'avg_rating': avg_rating,
            'last_rating_time': row['last_rating_time'] if row['last_rating_time'] else '',
            'has_duplicate': False  # 初始值，将在排序函数中更新
        })
    
    # 使用锦标赛排序算法排序
    sorted_images, has_duplicates, duplicate_count = tournament_sort(images, conn)
    
    # 查询原始文件名
    cursor.execute("SELECT id, original_name FROM images")
    all_images_dict = {row['id']: row['original_name'] for row in cursor.fetchall()}
    
    # 添加原始文件名
    for img in sorted_images:
        if img['id'] in all_images_dict:
            img['original_name'] = all_images_dict[img['id']]
    
    # 将排序结果转换为JSON并保存到数据库
    result_json = json.dumps({
        'images': sorted_images,
        'has_duplicate_ratings': has_duplicates,
        'duplicate_count': duplicate_count,
        'sorting_progress': {
            'clear_relation_ratio': f"{SORTING_STATE['clear_relation_ratio']*100:.1f}%",
            'total_comparisons': SORTING_STATE['total_comparisons'],
            'is_sorting_finished': is_sorting_sufficient()[0]
        },
        'timestamp': datetime.now().isoformat()
    })
    
    cursor.execute("""
        INSERT INTO sorting_results (result_json, has_duplicates, duplicate_count, created_at)
        VALUES (?, ?, ?, ?)
    """, (result_json, 1 if has_duplicates else 0, duplicate_count, datetime.now()))
    
    conn.commit()
    logger.info(f"更新了排序结果: {len(sorted_images)}张图片, {duplicate_count}张并列图片")

# 获取评分状态信息（包括并列数、剩余评分次数等）
@app.route('/rating_status', methods=['GET'])
def rating_status():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 查询所有已评分图片的数量
    cursor.execute("SELECT COUNT(*) as total FROM images WHERE rating_count > 0")
    total_rated = cursor.fetchone()['total']
    
    # 查询所有图片的数量
    cursor.execute("SELECT COUNT(*) as total FROM images")
    total_images = cursor.fetchone()['total']
    
    # 查询最新的排序结果
    cursor.execute("""
        SELECT has_duplicates, duplicate_count, created_at
        FROM sorting_results 
        ORDER BY created_at DESC LIMIT 1
    """)
    
    sort_result = cursor.fetchone()
    
    # 获取全局排序状态
    global SORTING_STATE
    is_sorting_finished, finish_reason = is_sorting_sufficient()
    
    if sort_result:
        has_duplicates = bool(sort_result['has_duplicates'])
        duplicate_count = sort_result['duplicate_count']
        last_update = sort_result['created_at']
    else:
        # 没有排序结果时的默认值
        has_duplicates = False
        duplicate_count = 0
        last_update = None
    
    # 计算最少还需要的评分次数
    if is_sorting_finished:
        min_clicks_needed = 0
    else:
        # 计算关系明确度
        if total_images <= 1:
            clear_relation_ratio = 1.0
        else:
            clear_relation_ratio = SORTING_STATE.get('clear_relation_ratio', 0)
            
        # 设置目标关系明确度
        min_required_ratio = 0.9  # 要求90%的关系明确
        
        # 计算总关系数和已明确的关系数
        total_relations = total_images * (total_images - 1) // 2
        clear_relations = int(clear_relation_ratio * total_relations)
        
        # 计算还需要明确的关系数
        remaining_relations = max(0, int(total_relations * min_required_ratio) - clear_relations)
        
        # 估计每次评分能明确的关系数（理想情况下是3个，但实际可能低一些）
        relations_per_comparison = 2.5
        
        # 计算还需要的评分次数
        min_clicks_needed = math.ceil(remaining_relations / relations_per_comparison)
    
    # 计算已完成的百分比
    if total_images <= 1:
        completion_percentage = 100
    else:
        completion_percentage = min(100, SORTING_STATE.get('clear_relation_ratio', 0) * 100)
    
    conn.close()
    
    return jsonify({
        'total_images': total_images,
        'rated_images': total_rated,
        'duplicate_count': duplicate_count,
        'min_clicks_needed': min_clicks_needed,
        'sorting_progress': {
            'clear_relation_ratio': f"{completion_percentage:.1f}%",
            'total_comparisons': SORTING_STATE.get('total_comparisons', 0),
            'is_sorting_finished': is_sorting_finished,
            'finish_reason': finish_reason if is_sorting_finished else "排序进行中",
            'last_update': last_update
        }
    })

# 获取评分记录
@app.route('/rating_records', methods=['GET'])
def get_rating_records():
    image_id = request.args.get('image_id')
    
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if image_id:
        # 获取指定图片的评分记录
        cursor.execute("""
            SELECT r.id, r.image_id, r.score, r.rating_time, i.filename
            FROM rating_records r
            JOIN images i ON r.image_id = i.id
            WHERE r.image_id = ?
            ORDER BY r.rating_time DESC
        """, (image_id,))
    else:
        # 获取所有评分记录
        cursor.execute("""
            SELECT r.id, r.image_id, r.score, r.rating_time, i.filename
            FROM rating_records r
            JOIN images i ON r.image_id = i.id
            ORDER BY r.rating_time DESC
            LIMIT 100
        """)
    
    records = []
    for row in cursor.fetchall():
        records.append({
            'id': row['id'],
            'image_id': row['image_id'],
            'filename': row['filename'],
            'score': row['score'],
            'score_text': '喜欢' if row['score'] == 1 else ('不喜欢' if row['score'] == -1 else '一般'),
            'rating_time': row['rating_time']
        })
    
    conn.close()
    
    return jsonify({'records': records})

# 页面 - 查看评分记录
@app.route('/rating-history')
def rating_history():
    return render_template('rating_history.html')

# 获取已上传的所有图片
@app.route('/uploaded_images', methods=['GET'])
def get_uploaded_images():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, filename, upload_time, rating_count, original_name
        FROM images
        ORDER BY upload_time DESC
    """)
    
    images = []
    for row in cursor.fetchall():
        images.append({
            'id': row['id'],
            'filename': row['filename'],
            'url': f"http://127.0.0.1:5000/static/uploads/{row['filename']}",
            'upload_time': row['upload_time'],
            'rating_count': row['rating_count'],
            'original_name': row['original_name']
        })
    
    conn.close()
    
    return jsonify({'images': images})

# 删除图片
@app.route('/delete_image/<image_id>', methods=['DELETE'])
def delete_image(image_id):
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    
    # 获取图片文件名
    cursor.execute("SELECT filename FROM images WHERE id = ?", (image_id,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        return jsonify({'error': '图片不存在'}), 404
    
    filename = row[0]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # 首先删除相关的评分记录
    cursor.execute("DELETE FROM rating_records WHERE image_id = ?", (image_id,))
    
    # 然后删除图片记录
    cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
    
    conn.commit()
    conn.close()
    
    # 删除文件
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        return jsonify({'error': f'删除文件失败: {str(e)}'}), 500
    
    return jsonify({'success': True})

# 删除所有图片
@app.route('/delete_all_images', methods=['DELETE'])
def delete_all_images():
    conn = sqlite3.connect(app.config['DATABASE'])
    cursor = conn.cursor()
    
    # 获取所有图片的文件名
    cursor.execute("SELECT filename FROM images")
    rows = cursor.fetchall()
    
    # 删除所有评分记录
    cursor.execute("DELETE FROM rating_records")
    
    # 删除所有图片记录
    cursor.execute("DELETE FROM images")
    
    # 删除所有图片记录
    cursor.execute("DELETE FROM sorting_results")
    
    conn.commit()
    conn.close()
    
    # 删除所有文件
    error_files = []
    for row in rows:
        filename = row[0]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            error_files.append(f"{filename}: {str(e)}")
    
    if error_files:
        return jsonify({
            'success': True, 
            'warning': '部分文件删除失败',
            'error_files': error_files
        })
    
    return jsonify({'success': True})

# 获取排序后的图片列表
@app.route('/ranked_images', methods=['GET'])
def ranked_images():
    # 添加refresh参数，用于强制刷新排序结果
    refresh = request.args.get('refresh', '0') == '1'
    
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 检查是否有存储的排序结果，且不需要刷新
    if not refresh:
        cursor.execute("""
            SELECT result_json, created_at FROM sorting_results 
            ORDER BY created_at DESC LIMIT 1
        """)
        
        row = cursor.fetchone()
        
        if row and row['result_json']:
            # 计算结果的新鲜度
            created_at = row['created_at']
            now = datetime.now()
            
            try:
                created_time = datetime.fromisoformat(created_at)
                freshness = (now - created_time).total_seconds()
                
                # 如果结果是最近5分钟内生成的，直接返回
                if freshness < 300:  # 5分钟 = 300秒
                    logger.info(f"使用缓存的排序结果 (生成于 {freshness:.1f} 秒前)")
                    conn.close()
                    return jsonify(json.loads(row['result_json']))
            except:
                pass  # 如果日期解析失败，继续执行刷新逻辑
    
    logger.info("生成新的排序结果")
    
    # 查询所有图片及其最后评分时间
    cursor.execute("""
        SELECT i.id, i.filename, i.rating, i.rating_count, i.upload_time,
               MAX(r.rating_time) as last_rating_time
        FROM images i
        LEFT JOIN rating_records r ON i.id = r.image_id
        GROUP BY i.id
    """)
    
    images = []
    rows = cursor.fetchall()
    
    for row in rows:
        image_id = row['id']
        rating = row['rating']
        rating_count = row['rating_count']
        
        # 计算平均分
        avg_rating = 0 if rating_count == 0 else rating / rating_count
        
        images.append({
            'id': image_id,
            'filename': row['filename'],
            'url': f"/static/uploads/{row['filename']}",
            'rating': rating,
            'rating_count': rating_count,
            'avg_rating': avg_rating,
            'last_rating_time': row['last_rating_time'] if row['last_rating_time'] else '',
            'has_duplicate': False  # 初始值，将在排序函数中更新
        })
    
    # 使用锦标赛排序算法排序
    sorted_images, has_duplicates, duplicate_count = tournament_sort(images, conn)
    
    # 查询原始文件名
    cursor.execute("SELECT id, original_name FROM images")
    all_images_dict = {row['id']: row['original_name'] for row in cursor.fetchall()}
    
    # 添加原始文件名
    for img in sorted_images:
        if img['id'] in all_images_dict:
            img['original_name'] = all_images_dict[img['id']]
    
    # 检查排序是否完成
    is_sufficient, reason = is_sorting_sufficient()
        
    # 同时保存排序结果
    result_json = json.dumps({
        'images': sorted_images,
        'has_duplicate_ratings': has_duplicates,
        'duplicate_count': duplicate_count,
        'sorting_progress': {
            'clear_relation_ratio': f"{SORTING_STATE['clear_relation_ratio']*100:.1f}%",
            'total_comparisons': SORTING_STATE['total_comparisons'],
            'is_sorting_finished': is_sufficient,
            'finish_reason': reason if is_sufficient else "排序进行中",
            "clear_relation_value": SORTING_STATE['clear_relation_ratio']
        },
        'timestamp': datetime.now().isoformat()
    })
    
    cursor.execute("""
        INSERT INTO sorting_results (result_json, has_duplicates, duplicate_count, created_at)
        VALUES (?, ?, ?, ?)
    """, (result_json, 1 if has_duplicates else 0, duplicate_count, datetime.now()))
    
    conn.commit()
    conn.close()
    
    return jsonify({
        'images': sorted_images,
        'has_duplicate_ratings': has_duplicates,
        'duplicate_count': duplicate_count,
        'sorting_progress': {
            'clear_relation_ratio': f"{SORTING_STATE['clear_relation_ratio']*100:.1f}%",
            'total_comparisons': SORTING_STATE['total_comparisons'],
            'is_sorting_finished': is_sufficient,
            'finish_reason': reason if is_sufficient else "排序进行中",
            "clear_relation_value": SORTING_STATE['clear_relation_ratio']
        }
    })

# 启动应用
if __name__ == '__main__':
    # 确保数据库初始化
    init_db()
    app.run(debug=True, host='0.0.0.0', port=10000) 