import yaml
import torch
from collections import deque
import os

class ComorbidityDAG:
    def __init__(self, config_path):
        if not os.path.exists(config_path):
            # Try finding it in the project root if called from src/
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            config_path = os.path.join(project_root, 'config.yaml')
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.disease_names = config['diseases']['labels']
        self.edges = []
        raw_edges = config['graph']['edges']
        
        self.predecessors = {d: [] for d in self.disease_names}
        self.successors = {d: [] for d in self.disease_names}
        self.in_degree = {d: 0 for d in self.disease_names}
        
        for src, dst, weight in raw_edges:
            self.edges.append((src, dst, weight))
            self.predecessors[dst].append((src, weight))
            self.successors[src].append((dst, weight))
            self.in_degree[dst] += 1
            
        self.topological_sort_order = self._compute_topological_sort()

    def _compute_topological_sort(self):
        # Kahn's algorithm
        in_degree = self.in_degree.copy()
        
        # Initial roots from the queue
        roots_in_queue = [d for d in self.disease_names if in_degree[d] == 0]
        
        # Priority: obesity, osteoporosis
        # We want root nodes to come first. 
        # But we also need to maintain the topological order property.
        
        queue = deque()
        # Add preferred roots first if they are in the initial roots
        for root in ['obesity', 'osteoporosis']:
            if root in roots_in_queue:
                queue.append(root)
                roots_in_queue.remove(root)
        
        # Add remaining roots
        for r in sorted(roots_in_queue):
            queue.append(r)
        
        topo_order = []
        while queue:
            # Sort current queue to favor obesity/osteoporosis if they just became available?
            # Actually, in a DAG, if they are roots they are available at the start.
            u = queue.popleft()
            topo_order.append(u)
            
            # Sort successors to keep some determinism if multiple become 0 in-degree
            successors_nodes = sorted(self.successors[u], key=lambda x: x[0])
            for v, weight in successors_nodes:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        return topo_order

    def get_predecessors(self, disease_name):
        return self.predecessors.get(disease_name, [])

    def topological_order(self):
        return self.topological_sort_order

def build_augmented_inputs(z, initial_preds, dag, disease_names):
    """
    Args:
        z: (batch, hidden_dim) tensor
        initial_preds: dict mapping disease_name -> (batch, 1) tensor
        dag: ComorbidityDAG instance
        disease_names: list of all disease names
    Returns:
        dict mapping disease_name -> z_tilde tensor
    """
    z_tilde_dict = {}
    topo_order = dag.topological_order()
    
    for d in topo_order:
        preds = dag.get_predecessors(d)
        if not preds:
            z_tilde_d = z
        else:
            # upstream: (batch, num_preds)
            upstream_list = []
            for (p, w) in preds:
                # initial_preds[p] is (batch, 1)
                # w is a scalar weight
                # torch.stack([initial_preds[p] * w for (p,w) in preds], dim=1) as per prompt
                upstream_list.append(initial_preds[p] * w)
            
            # Use stack then cat as per Prompt 12 PART B
            # upstream = torch.stack([initial_preds[p] * w for (p,w) in preds], dim=1)
            # wait, initial_preds[p] is (batch, 1). 
            # initial_preds[p] * w is (batch, 1).
            # stack(..., dim=1) would give (batch, num_preds, 1).
            # The prompt says: upstream = torch.stack([initial_preds[p] * w for (p,w) in preds], dim=1)
            # z_tilde_d = torch.cat([z, upstream], dim=1)
            # If z is (batch, hidden_dim), then upstream must be (batch, something).
            # If stack results in (batch, num_preds, 1), cat with (batch, hidden_dim) fails on dim 1.
            # Maybe it meant cat? Or maybe upstream should be squeezed?
            # Actually, if initial_preds[p] is (batch, 1), and we want to cat it, we need (batch, num_preds).
            # Let's follow the prompt's `torch.stack` if possible, but be careful.
            
            # Prompt says: upstream = torch.stack([initial_preds[p] * w for (p,w) in preds], dim=1)
            # This would be (batch, num_preds, 1) if initial_preds is (batch, 1).
            # If we cat [z, upstream], they must match in all dims except dim 1.
            # z is (batch, hidden_dim). upstream is (batch, num_preds, 1).
            # This doesn't look right unless z is reshaped or upstream is squeezed.
            # Let's look at the prompt again: "z_tilde_d = torch.cat([z, upstream], dim=1)"
            # Most likely upstream should be (batch, num_preds).
            # If we stack (batch, 1) tensors along dim 1, we get (batch, num_preds, 1).
            # If we squeeze the last dim, we get (batch, num_preds).
            
            upstream = torch.stack([initial_preds[p] * w for (p, w) in preds], dim=1).squeeze(-1)
            z_tilde_d = torch.cat([z, upstream], dim=1)
        
        z_tilde_dict[d] = z_tilde_d
        
    return z_tilde_dict

# Print topological order on import
try:
    # Assuming config.yaml is in the project root
    # We use a default path but the constructor handles relative paths
    _config_path = 'config.yaml'
    _dag = ComorbidityDAG(_config_path)
    print(f"[GRAPH REGISTRY] Topological Order: {_dag.topological_order()}")
except Exception as e:
    # Silently handle if config.yaml not found during import (e.g. when running tests from different CWD)
    pass
