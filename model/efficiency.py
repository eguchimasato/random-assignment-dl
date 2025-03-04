class compute_ev:
    def __init__(self, P, preferences):
        """
        P: n×n の二重確率行列 (torch.Tensor)
        preferences: n×n の選好行列 (torch.Tensor)
                     各行 i はエージェント i の選好を表し、値が大きいほど好む
        """
        self.P = P.clone()
        self.preferences = preferences
        self.n = P.shape[0]

    def build_graph(self, Q):
        """
        Q: 現在の二重確率行列 (torch.Tensor)
        オブジェクトを頂点とするグラフを構築する。
        エッジ (a -> b) は、あるエージェント i が a を b より好む（preferences[i, a] > preferences[i, b]）
        かつ Q[i, b] > 0 である場合に追加する。
        エッジには (b, i, Q[i, b]) の情報を記録する。
        """
        graph = {a: [] for a in range(self.n)}
        for a in range(self.n):
            for b in range(self.n):
                if a == b:
                    continue
                for i in range(self.n):
                    if self.preferences[i, a] > self.preferences[i, b] and Q[i, b] > 0:
                        graph[a].append((b, i, Q[i, b].item()))
                        # 同じ (a, b) ペアについて、最初に条件を満たしたエージェントを witness とする
                        break
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

        for vertex in range(self.n):
            if vertex not in visited:
                cycle = dfs(vertex)
                if cycle is not None:
                    return cycle
        return None

    def execute_all_cycles(self):
        """
        現在の二重確率行列 self.P に対して、グラフを構築し、サイクルを探索しては
        そのサイクル内で交換可能な最小の確率 epsilon だけ交換を実施する。
        複数のサイクルが存在する場合、すべてのサイクルで交換が行われるまで反復する。
        交換が行われたサイクルと各サイクルでの epsilon を記録し、最終的な更新後行列 Q と
        サイクル情報のリストを返す。
        """
        Q = self.P.clone()
        cycles_exchanges = []
        ev = 0.0
        while True:
            graph = self.build_graph(Q)
            cycle = self.find_cycle(graph)
            if cycle is None:
                break
            # サイクル内の各エッジの利用可能な確率の最小値を epsilon とする
            epsilons = [edge[3] for edge in cycle]
            epsilon = min(epsilons)
            ev += epsilon
            # サイクル内の各エッジについて交換を実施
            for (a, b, agent, avail) in cycle:
                Q[agent, b] -= epsilon
                Q[agent, a] += epsilon
            cycles_exchanges.append((cycle, epsilon))
        return ev