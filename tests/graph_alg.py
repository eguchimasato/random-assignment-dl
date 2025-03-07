import torch
import numpy as np
import multiprocessing as mp


class Graph:
    def __init__(self, Q, preferences):
        self.Q = Q
        self.preferences = preferences
        self.num_agents, self.num_objects = Q.shape
        
        
    def build_graph(self):
        """
        Q: 現在の n x n 二重確率行列 (torch.Tensor)
        オブジェクトを頂点とするグラフを構築する。
        エッジ (a -> b) は、あるエージェント i が a を b より好む（preferences[i, a] > preferences[i, b]）
        かつ Q[i, b] > 0 である場合に追加する。
        エッジには (b, i, Q[i, b]) の情報を記録する。
        """
        graph = {a: [] for a in range(self.num_objects)}
        for a in range(self.num_objects):
            for b in range(self.num_objects):
                if a == b:
                    continue
                
                for i in range(self.num_agents):
                    if (self.preferences[i, a] > self.preferences[i, b]) and (self.Q[i, b] > 0):
                        graph[a].append((b, i, self.Q[i, b].item()))
                        break # 同じ (a, b) ペアについて、最初に条件を満たしたエージェントを witness とする
        
        return graph

    
    def find_cycle(self, graph):
        """
        DFS を用いてグラフ中のサイクル（閉路）を探索する。
        サイクルが見つかった場合は、サイクルを構成するエッジのリストを返す。
        各エッジは (from_object, to_object, witness_agent, available_probability) のタプル。
        サイクルがなければ None を返す。
        """
        visited = set()
        rec_stack = []  # 各要素は (vertex, edge_info)。最初の頂点は edge_info=None

        def dfs(v):
            visited.add(v)
            rec_stack.append((v, None))
            for (nbr, agent, avail) in graph[v]:
                for idx, (node, _) in enumerate(rec_stack):
                    if node == nbr:
                        cycle_edges = []
                        # rec_stack[idx+1:] に記録されているエッジ情報がサイクル内のエッジ
                        for j in range(idx + 1, len(rec_stack)):
                            edge = rec_stack[j][1]
                            if edge is not None:
                                cycle_edges.append(edge)
                        
                        cycle_edges.append((v, nbr, agent, avail))
                        return cycle_edges
                
                if nbr not in visited:
                    rec_stack.append((v, (v, nbr, agent, avail)))
                    result = dfs(nbr)
                    if result is not None:
                        return result
                    
                    rec_stack.pop()
            
            rec_stack.pop()
            return None

        for vertex in range(self.num_objects):
            if vertex not in visited:
                cycle = dfs(vertex)
                if cycle is not None:
                    return cycle

        return None
    
    
    def execute_all_cycles(self):
        """
        idx: バッチ内のインデックス
        対応する n x n 行列に対してサイクル交換を実施し、違反量 (violation) を計算する。
        """
        # 抽出: バッチ内の idx 番目の n x n 行列
        cycles_exchanges = []
        violation = 0.0
        while True:
            graph = self.build_graph()
            cycle = self.find_cycle(graph)
            if cycle is None:
                break
            
            # サイクル内の各エッジの利用可能な確率の最小値を epsilon とする
            epsilons = [edge[3] for edge in cycle]
            epsilon = min(epsilons)
            violation += epsilon
            # サイクル内の各エッジについて交換を実施
            for (a, b, agent, avail) in cycle:
                self.Q[agent, b] -= epsilon
                self.Q[agent, a] += epsilon
            
            cycles_exchanges.append((cycle, epsilon))
        
        return violation


class compute_ev:
    def __init__(self, P, preferences, num_processes=None):
        """
        P: batch_size x n x n の二重確率行列 (torch.Tensor)
        preferences: batch_size x n x n の選好行列 (torch.Tensor)
                     各行 i はエージェント i の選好を表し、値が大きいほど好む
        """
        self.P = P
        self.preferences = preferences
        self.batch_size = P.shape[0]
        self.num_agents = P.shape[1]
        self.num_objects = P.shape[2]

        if num_processes is None:
            self.num_processes = mp.cpu_count()
        else:
            self.num_processes = num_processes
        
    
    @staticmethod
    def func(Q, preferences):
        g = Graph(Q, preferences)
        return g.execute_all_cycles()
        
    
    def divide_P_matrices(self, idx):
        Q = self.P[idx].cpu().detach().numpy()
        return Q
    
    
    def divide_preferences_matrices(self, idx):
        preferences = self.preferences[idx].cpu().detach().numpy()
        return preferences

    
    def execute_all_cycles_batch(self):
        """
        バッチ内の各 n x n 行列に対して execute_all_cycles を計算し、違反量 (violation) をまとめる。
        結果はバッチサイズ x 1 のテンソルとなる。
        """
        # P行列とpreferences行列を分割
        P_list = [self.divide_P_matrices(i) for i in range(self.batch_size)]
        preferences_list = [self.divide_preferences_matrices(i) for i in range(self.batch_size)]

        with mp.Pool(self.num_processes) as pool:
            # 各プロセスに対してバッチ内の異なる idx を渡して並列実行
            violations = pool.starmap(self.func, zip(P_list, preferences_list))

        # 結果を torch.Tensor に変換し、形状を (batch_size, 1) にする
        violations = torch.tensor(violations, dtype=torch.float32).unsqueeze(1)

        return violations, self.num_processes
