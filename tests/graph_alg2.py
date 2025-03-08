import networkx as nx
import torch
import numpy as np
import multiprocessing as mp

class Graph:
    def __init__(self, Q, preferences):
        """
        Q: エージェントごとの n x n 二重確率行列 (numpy.array または torch.Tensor)
        preferences: エージェントごとの n x n 選好行列 (numpy.array または torch.Tensor)
        """
        self.Q = Q  # compute_ev で numpy.array に変換済み
        self.preferences = preferences
        self.num_agents, self.num_objects = self.Q.shape

    def build_graph(self):
        """
        オブジェクトを頂点とするグラフを構築する処理を、全エージェント分の 3 次元配列を作成せずに
        各エージェントごとに 2 次元の条件行列を作成することで高速化する。
        エッジ (a -> b) は、エージェント i について
          - preferences[i, a] > preferences[i, b]
          - Q[i, b] > 0
        の条件を満たす場合に追加し、最初に条件を満たしたエージェント i を witness とする。
        """
        P = self.preferences  # shape: (num_agents, num_objects)
        Q = self.Q            # shape: (num_agents, num_objects)
        num_agents, num_objects = P.shape

        # 結果のグラフ: 各頂点 a に対して、(b, witness_agent, Q 値) のリスト
        graph = {a: [] for a in range(num_objects)}
        # 各 (a, b) ペアについて、すでにエッジが追加されたかを記録（最初の witness だけを採用）
        used = np.zeros((num_objects, num_objects), dtype=bool)
        
        # エージェント i ごとに条件行列を計算し、条件を満たす (a, b) ペアを抽出
        for i in range(num_agents):
            # 各エージェントについて、a, b の比較結果を 2 次元配列として計算
            # ※ Q[i, b] > 0 は、各行に対して b 軸でブロードキャストされる
            cond_i = (P[i][:, None] > P[i][None, :]) & (Q[i][None, :] > 0)
            # すでに他のエージェントでエッジが追加されていない (a, b) ペアのみ対象
            valid = cond_i & (~used)
            # 有効な (a, b) ペアのインデックスを一括取得
            a_indices, b_indices = np.where(valid)
            # a == b（自己ループ）は除外
            mask = a_indices != b_indices
            a_indices = a_indices[mask]
            b_indices = b_indices[mask]
            
            # 取得した全 (a, b) ペアに対して、エージェント i を witness としてエッジを追加
            for a, b in zip(a_indices, b_indices):
                graph[a].append((b, i, float(Q[i, b])))
                used[a, b] = True
        return graph

    def find_cycle(self, graph):
        """
        NetworkX を用いて、グラフ中の閉路（サイクル）を探索する。
        サイクルが存在する場合、各エッジ (from, to, witness, available_probability) のリストを返す。
        サイクルがなければ None を返す。
        """
        G = nx.DiGraph()
        for u, edges in graph.items():
            for (v, agent, avail) in edges:
                G.add_edge(u, v, agent=agent, avail=avail)
        try:
            cycle_edges = nx.find_cycle(G)
            cycle = []
            for u, v in cycle_edges:
                data = G[u][v]
                cycle.append((u, v, data['agent'], data['avail']))
            return cycle
        except nx.NetworkXNoCycle:
            return None

    def execute_all_cycles(self):
        """
        サイクルがなくなるまで、グラフ構築 → サイクル検出 → サイクル交換を実施する。
        各サイクルでは、サイクル内のエッジのうち最小の Q 値を epsilon として交換を行う。
        """
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

            # サイクル内の各エッジについて、Q の交換を実施
            for (a, b, agent, avail) in cycle:
                self.Q[agent, b] -= epsilon
                self.Q[agent, a] += epsilon

            cycles_exchanges.append((cycle, epsilon))

        return violation, cycles_exchanges

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
        P_list = [self.divide_P_matrices(i) for i in range(self.batch_size)]
        preferences_list = [self.divide_preferences_matrices(i) for i in range(self.batch_size)]

        with mp.Pool(self.num_processes) as pool:
            results = pool.starmap(self.func, zip(P_list, preferences_list))
            violations, cycles = zip(*results)

        violations = torch.tensor(violations, dtype=torch.float32).unsqueeze(1)
        return violations, cycles