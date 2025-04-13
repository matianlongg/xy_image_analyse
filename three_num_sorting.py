import numpy as np
import random
from collections import defaultdict, deque
import time

class ThreeNumSorting:
    def __init__(self, n=21):
        self.n = n
        self.numbers = list(range(1, n+1))
        # 用邻接表表示有向图
        self.graph = defaultdict(set)
        # 用矩阵记录传递闭包，加速查询
        self.tc_matrix = np.zeros((n+1, n+1), dtype=bool)
        self.comparison_count = 0
        
    def reset(self):
        """重置状态"""
        self.graph = defaultdict(set)
        self.tc_matrix = np.zeros((self.n+1, self.n+1), dtype=bool)
        self.comparison_count = 0
    
    def compare_three(self, a, b, c):
        """模拟用户比较三个数字，返回它们的顺序"""
        self.comparison_count += 1
        sorted_nums = sorted([a, b, c])
        return sorted_nums
    
    def update_graph(self, a, b, c):
        """更新有向图，添加 a < b < c 的关系"""
        # 更新邻接表
        self.graph[a].add(b)
        self.graph[a].add(c)
        self.graph[b].add(c)
        
        # 更新传递闭包矩阵
        self.tc_matrix[a, b] = True
        self.tc_matrix[a, c] = True
        self.tc_matrix[b, c] = True
        
        # 快速更新传递闭包矩阵
        self.fast_transitive_closure([(a, b), (a, c), (b, c)])
        
    def fast_transitive_closure(self, new_edges):
        """更高效的传递闭包更新算法，只更新受新边影响的部分"""
        for i in range(1, self.n+1):
            for j in range(1, self.n+1):
                if self.tc_matrix[i, j]:
                    continue  # 已经存在的关系不需要更新
                
                # 检查是否可以通过新边构建i->j的关系
                for u, v in new_edges:
                    # 如果i可以到达u，并且v可以到达j，那么i也可以到达j
                    if (i == u or self.tc_matrix[i, u]) and (v == j or self.tc_matrix[v, j]):
                        self.tc_matrix[i, j] = True
                        # 将这个新发现的关系也作为新边处理
                        new_edges.append((i, j))
                        break
    
    def floyd_warshall(self):
        """使用Floyd-Warshall算法计算传递闭包，更适合大规模数据"""
        # 初始化矩阵，直接使用当前tc_matrix
        
        # Floyd-Warshall核心算法
        for k in range(1, self.n+1):
            for i in range(1, self.n+1):
                if self.tc_matrix[i, k]:
                    for j in range(1, self.n+1):
                        if self.tc_matrix[k, j]:
                            self.tc_matrix[i, j] = True
    
    def is_unique_order(self):
        """检查是否能确定唯一的全局顺序"""
        # 计算每个节点的入度(使用tc_matrix而不是graph)
        in_degree = {i: 0 for i in range(1, self.n+1)}
        for i in range(1, self.n+1):
            for j in range(1, self.n+1):
                if self.tc_matrix[i, j]:
                    in_degree[j] += 1
                    
        # 找到所有入度为0的节点
        zero_in_degree = [node for node, degree in in_degree.items() if degree == 0]
        
        # 如果只有一个入度为0的节点，并且所有节点都参与了关系
        # 那么我们可以确定唯一的全局顺序
        if len(zero_in_degree) != 1:
            return False
            
        # 检查是否图中每对节点间都有关系
        # 对于每一对节点i,j，要么i<j，要么j<i
        for i in range(1, self.n+1):
            for j in range(i+1, self.n+1):
                if not (self.tc_matrix[i, j] or self.tc_matrix[j, i]):
                    return False
                    
        return True
    
    def get_final_order(self):
        """获取最终的全局顺序"""
        # 拓扑排序
        in_degree = {i: 0 for i in range(1, self.n+1)}
        for i in range(1, self.n+1):
            for j in range(1, self.n+1):
                if self.tc_matrix[i, j]:
                    in_degree[j] += 1
        
        queue = deque([node for node in range(1, self.n+1) if in_degree[node] == 0])
        topo_order = []
        
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            
            for neighbor in range(1, self.n+1):
                if self.tc_matrix[node, neighbor]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        return topo_order
    
    def select_best_triplet(self):
        """启发式选择最有信息量的三元组 - 优化版"""
        # 计算已知关系的数量
        relation_matrix = np.logical_or(self.tc_matrix, self.tc_matrix.T)
        relation_count = np.sum(relation_matrix, axis=1)
        
        # 获取关系最少的前15个节点
        least_related = np.argsort(relation_count[1:])[:15] + 1
        
        # 从这些节点中选择三个关系最不明确的构成三元组
        best_score = -1
        best_triplet = None
        
        # 限制尝试组合的数量
        max_combinations = 100
        combinations_tried = 0
        
        # 随机采样一些组合
        for _ in range(max_combinations):
            if len(least_related) < 3:
                a, b, c = random.sample(range(1, self.n+1), 3)
            else:
                a, b, c = random.sample(list(least_related), 3)
            
            # 计算未知关系数
            unknown_relations = 0
            if not (relation_matrix[a, b] or relation_matrix[b, a]):
                unknown_relations += 1
            if not (relation_matrix[a, c] or relation_matrix[c, a]):
                unknown_relations += 1
            if not (relation_matrix[b, c] or relation_matrix[c, b]):
                unknown_relations += 1
            
            if unknown_relations > best_score:
                best_score = unknown_relations
                best_triplet = (a, b, c)
                
            combinations_tried += 1
            if best_score == 3:  # 如果找到完全未知的三元组，立即返回
                break
        
        if best_triplet:
            return best_triplet
            
        # 如果上面方法没找到好的三元组，随机选择
        return random.sample(range(1, self.n+1), 3)
    
    def run_algorithm(self):
        """运行主算法"""
        self.reset()
        start_time = time.time()
        iteration = 0
        
        while not self.is_unique_order():
            iteration += 1
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(f"迭代 {iteration}，已用时 {elapsed:.2f}秒，已进行 {self.comparison_count} 次比较")
            
            # 选择三个数字
            a, b, c = self.select_best_triplet()
            
            # 比较并获取它们的顺序
            sorted_nums = self.compare_three(a, b, c)
            
            # 更新图
            self.update_graph(sorted_nums[0], sorted_nums[1], sorted_nums[2])
            
            # 每10次迭代进行一次全面的传递闭包计算
            if iteration % 10 == 0:
                self.floyd_warshall()
        
        elapsed = time.time() - start_time
        print(f"完成排序，用时 {elapsed:.2f}秒")
        # 返回最终顺序和比较次数
        return self.get_final_order(), self.comparison_count

# 运行多次实验以获取平均性能
def run_experiments(n_trials=10, n=21):
    total_comparisons = 0
    min_comparisons = float('inf')
    max_comparisons = 0
    total_time = 0
    
    for i in range(n_trials):
        start_time = time.time()
        sorter = ThreeNumSorting(n)
        _, comparisons = sorter.run_algorithm()
        elapsed = time.time() - start_time
        
        total_comparisons += comparisons
        total_time += elapsed
        min_comparisons = min(min_comparisons, comparisons)
        max_comparisons = max(max_comparisons, comparisons)
        
        print(f"试验 {i+1}: 需要 {comparisons} 次比较，耗时 {elapsed:.2f}秒")
    
    avg_comparisons = total_comparisons / n_trials
    avg_time = total_time / n_trials
    print(f"\n统计结果 ({n_trials} 次试验):")
    print(f"平均比较次数: {avg_comparisons:.2f}")
    print(f"最小比较次数: {min_comparisons}")
    print(f"最大比较次数: {max_comparisons}")
    print(f"平均耗时: {avg_time:.2f}秒")

if __name__ == "__main__":
    print(f"排序范围: 1-99")
    # 运行单次排序
    sorter = ThreeNumSorting(n=99)
    final_order, comparisons = sorter.run_algorithm()
    print(f"最终顺序: {final_order}")
    print(f"比较次数: {comparisons}")
    
    # 运行多次实验，测试性能
    # print("\n运行多次实验以测试性能...")
    # run_experiments(n_trials=3, n=99) 