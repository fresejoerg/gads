---
id: causal_discovery
description: "causal-learn PC/FCI/GES structure learning from observational data: CI tests, graph rendering via networkx+matplotlib, edge interpretation."
triggers: ["causal discovery", "structure learning", "DAG learning", "PC algorithm", "FCI", "GES", "causal-learn", "conditional independence", "skeleton", "CPDAG", "PAG", "causal structure", "discover causal"]
---
# Causal Discovery with causal-learn

## PC Algorithm (constraint-based, assumes causal sufficiency)

```python
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

# data_matrix: numpy array shape (n_samples, n_variables), NO missing values
cg = pc(
    data=data_matrix,
    alpha=0.05,           # significance threshold for CI tests
    indep_test=fisherz,   # Fisher Z for continuous data
    stable=True,
    uc_rule=0,
    uc_priority=-1,
    verbose=False
)
# cg.G is the GeneralGraph — access edges via cg.G.graph
discovered_graph = cg
```

## FCI Algorithm (allows latent confounders → produces PAG)

```python
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz

g, edges = fci(
    dataset=data_matrix,
    independence_test_method=fisherz,
    alpha=0.05,
    verbose=False
)
discovered_graph = g
```

## GES (score-based, Gaussian data)

```python
from causallearn.search.ScoreBased.GES import ges

Record = ges(data_matrix)
discovered_graph = Record['G']
```

## Visualize with networkx + matplotlib (no graphviz binary needed)

```python
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

adj = cg.G.graph   # adjacency matrix: adj[i,j]=1 and adj[j,i]=-1 → i→j
n = len(var_names)
G_nx = nx.DiGraph()
G_nx.add_nodes_from(var_names)

directed_edges, undirected_edges, bidir_edges = [], [], []
for i in range(n):
    for j in range(i+1, n):
        if adj[i, j] == 1 and adj[j, i] == -1:
            directed_edges.append((var_names[i], var_names[j]))
        elif adj[j, i] == 1 and adj[i, j] == -1:
            directed_edges.append((var_names[j], var_names[i]))
        elif adj[i, j] == -1 and adj[j, i] == -1:
            undirected_edges.append((var_names[i], var_names[j]))
        elif adj[i, j] == 1 and adj[j, i] == 1:
            bidir_edges.append((var_names[i], var_names[j]))

fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(G_nx, seed=42)
nx.draw_networkx_nodes(G_nx, pos, ax=ax, node_color="lightblue", node_size=800)
nx.draw_networkx_labels(G_nx, pos, ax=ax, font_size=9)
nx.draw_networkx_edges(G_nx, pos, edgelist=directed_edges, ax=ax,
                       arrows=True, arrowsize=20, edge_color="black")
# Undirected: draw as grey lines without arrows
if undirected_edges:
    G_un = nx.Graph()
    G_un.add_edges_from(undirected_edges)
    nx.draw_networkx_edges(G_un, pos, ax=ax, edge_color="grey", style="dashed")
# Bidirected (latent confounder): draw in red
if bidir_edges:
    G_bi = nx.Graph()
    G_bi.add_edges_from(bidir_edges)
    nx.draw_networkx_edges(G_bi, pos, ax=ax, edge_color="red", style="dotted")
ax.set_title("Discovered Causal Graph")
ax.axis("off")
plt.tight_layout()
plt.savefig("causal_graph.png", bbox_inches="tight", dpi=150)
plt.close()
```

## Edge Interpretation
| adj[i,j] / adj[j,i] | Meaning |
|---|---|
| `1 / -1` | Directed edge i → j |
| `-1 / -1` | Undirected (Markov-equivalent, direction not identified) |
| `1 / 1` | Bidirected ↔ (latent common cause, FCI only) |
| `0 / 0` | No edge |

## Sandbox Rules
- `causallearn` (import as `from causallearn.search...`) is available.
- `networkx`, `matplotlib`, `numpy` are available — use these for rendering.
- Do NOT use `graphviz` binary calls or `pydot` — use networkx+matplotlib.
- Data must be numeric with no missing values before calling discovery algorithms.
