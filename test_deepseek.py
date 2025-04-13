import random
from collections import defaultdict
import heapq

def main():
    print("优化版自动数字排序程序")
    print("系统包含数字1-21，智能选择数字组合，快速确定完整排序\n")
    
    numbers = list(range(1, 22))
    known_relations = defaultdict(set)  # known_relations[x] = {y | x < y}
    iteration = 0
    
    while True:
        iteration += 1
        # 检查是否已确定完整排序
        sorted_list = try_get_full_sort(numbers, known_relations)
        if sorted_list:
            print(f"\n经过 {iteration} 轮排序后，已确定完整排序:")
            print(" ".join(map(str, sorted_list)))
            break
        
        # 智能选择最有信息量的3个数字
        a, b, c = select_most_informative_triplet(numbers, known_relations)
        if not a:
            print("\n无法进一步确定关系")
            break
        
        # 自动按照从小到大排序
        sorted_triplet = sorted([a, b, c])
        print(f"第{iteration}轮: 排序数字 {a} {b} {c} -> {' '.join(map(str, sorted_triplet))}")
        
        # 记录新的关系 (x < y < z)
        x, y, z = sorted_triplet
        update_relations(x, y, z, known_relations)

def select_most_informative_triplet(numbers, known_relations):
    """选择能提供最多新信息的3个数字"""
    # 优先选择关系最不明确的数字
    uncertainty_scores = []
    for num in numbers:
        known_less = {n for n in numbers if num in known_relations.get(n, set())}
        known_greater = known_relations.get(num, set())
        unknown = set(numbers) - {num} - known_less - known_greater
        score = len(unknown)
        heapq.heappush(uncertainty_scores, (-score, num))  # 使用最大堆
    
    candidates = []
    while uncertainty_scores and len(candidates) < 3:
        _, num = heapq.heappop(uncertainty_scores)
        if num not in candidates:
            candidates.append(num)
    
    if len(candidates) < 3:
        return (None, None, None)
    
    # 从候选中选择能提供最多新关系的组合
    best_combo = None
    max_new_relations = -1
    
    # 检查几个随机组合以平衡效率和效果
    for _ in range(min(20, len(candidates)**3)):
        combo = random.sample(candidates, 3)
        new_relations = count_potential_new_relations(combo, known_relations)
        if new_relations > max_new_relations:
            max_new_relations = new_relations
            best_combo = combo
    
    return best_combo

def count_potential_new_relations(combo, known_relations):
    """计算这3个数字排序后可能新增的关系数量"""
    a, b, c = combo
    count = 0
    
    # 检查所有可能的两两关系
    for x, y in [(a,b), (a,c), (b,c)]:
        if y not in known_relations.get(x, set()):
            count += 1
    
    return count

def update_relations(x, y, z, known_relations):
    """更新已知关系"""
    # x < y < z
    known_relations[x].add(y)
    known_relations[x].add(z)
    known_relations[y].add(z)
    
    # 传递闭包：如果 a < b 且 b < c，则 a < c
    for a in list(known_relations):
        if x in known_relations[a]:
            known_relations[a].add(y)
            known_relations[a].add(z)
        if y in known_relations[a]:
            known_relations[a].add(z)

def try_get_full_sort(numbers, known_relations):
    """尝试获取完整排序"""
    in_degree = {num: 0 for num in numbers}
    graph = defaultdict(set)
    
    # 构建图
    for u in known_relations:
        for v in known_relations[u]:
            graph[u].add(v)
            in_degree[v] += 1
    
    # 初始化队列
    queue = [num for num in numbers if in_degree[num] == 0]
    sorted_list = []
    
    while queue:
        if len(queue) > 1:
            return None  # 排序不唯一
        
        u = queue.pop(0)
        sorted_list.append(u)
        
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    if len(sorted_list) != len(numbers):
        return None  # 存在环或信息不全
    
    return sorted_list

if __name__ == "__main__":
    main()