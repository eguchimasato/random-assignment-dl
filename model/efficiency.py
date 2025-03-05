import torch
import numpy as np

class compute_ev:
    def __init__(self, cfg, P, preferences):
        """
        P: n*n の二重確率行列 (torch.Tensor)
        preferences: n*n の選好行列 (torch.Tensor)
                     各行 i はエージェント i の選好を表し、値が大きいほど好む
        """
        self.cfg = cfg
        self.P = P.clone()
        if isinstance(preferences, np.ndarray):
            preferences = torch.tensor(preferences, dtype=torch.float32)
        self.preferences = preferences.clone().detach().float()
        self.n = cfg.num_goods

    def build_graph(self, Q):
        graph = {a: [] for a in range(self.n)}
        for i in range(self.n):
            sorted_preferences = torch.argsort(self.preferences[i], descending=True)
            for a in sorted_preferences:
                for b in sorted_preferences:
                    if a == b:
                        continue
                    if Q[i, b] > 0:
                        if a not in graph:
                            graph[a] = []
                        graph[a].append((b, i, Q[i, b].item()))
                        break
        return graph
    
    def find_cycle(self, graph):
        visited = set()
        rec_stack = []

        def dfs(v):
            visited.add(v)
            rec_stack.append((v, None))
            for (nbr, agent, avail) in graph[v]:
                if nbr in visited:
                    cycle_edges = [(v, nbr, agent, avail)]
                    for node, edge in reversed(rec_stack):
                        if node == nbr:
                            break
                        cycle_edges.append(edge)
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
        Q = self.P.clone()
        cycles_exchanges = []
        violation = 0.0
        while True:
            graph = self.build_graph(Q)
            cycle = self.find_cycle(graph)
            if cycle is None:
                break
            epsilons = [edge[3] for edge in cycle]
            epsilon = min(epsilons)
            violation += epsilon
            for (a, b, agent, avail) in cycle:
                Q[agent, b] -= epsilon
                Q[agent, a] += epsilon
            cycles_exchanges.append((cycle, epsilon))
        return violation

    def execute_all_cycles_batch(self):
        """
        P と preferences の各バッチに対して execute_all_cycles を計算し、
        それらの結果を合わせて n*1 の行列にする。
        """
        batch_size = self.P.shape[0]
        results = torch.zeros((batch_size, 1), dtype=torch.float32)

        for b in range(batch_size):
            P_batch = self.P[b].view(self.n, self.n)
            preferences_batch = self.preferences[b].view(self.n, self.n)
            ev_instance = compute_ev(self.cfg, P_batch, preferences_batch)
            results[b, 0] = ev_instance.execute_all_cycles()

        return results