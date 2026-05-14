import networkx as nx
import matplotlib.pyplot as plt


def draw_topology(cluster):
    pos = nx.kamada_kawai_layout(cluster.topology)
    edge_labels = {(u, v): "{:.1f}".format(data['usage']) for u, v, data in cluster.topology.edges(data=True)}

    node_colors = []
    node_text_colors = {}
    labels = {}

    for node in cluster.topology.nodes():
        node_data = cluster.topology.nodes[node]
        if node_data['type'] == 'node':  # Compute node
            util = cluster.compute_node_utilization(node_data['compute_node'])
            node_colors.append(util)
            labels[node] = f"{node}\n{util:.1%}"
            node_text_colors[node] = 'white'  # readable on colormap background
        else:  # Switch node
            node_colors.append(0.95)  # almost white/light gray
            labels[node] = node
            node_text_colors[node] = 'black'

    fig, ax = plt.subplots(figsize=(10, 6))
    nodes = nx.draw(
        cluster.topology, pos=pos, ax=ax,
        with_labels=False,
        node_size=400,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        alpha=0.9,
        edge_color='gray'
    )

    # Manually draw labels with specified font color
    for node, (x, y) in pos.items():
        ax.text(x, y, labels[node], fontsize=7, ha='center', va='center', color=node_text_colors[node])

    nx.draw_networkx_edge_labels(cluster.topology, pos=pos, edge_labels=edge_labels, font_size=7, ax=ax)

    # Create colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Node Resource Utilization')

    plt.tight_layout()
    plt.show()


def draw_tree_topology(cluster):
    highest_level_switch = max(cluster.switches, key=lambda x: int(x['Level']))
    pos = hierarchy_pos(cluster.topology, highest_level_switch['id'])
    edge_labels = {}
    for (u, v, data) in cluster.topology.edges(data=True):
        edge_labels[(u, v)] = "{:.1f}".format(data['usage'])
    nx.draw(cluster.topology, pos=pos, with_labels=True, arrows=False, font_size=8, node_size=200, node_color='lightblue', alpha=0.8)
    nx.draw_networkx_edge_labels(cluster.topology, pos, edge_labels=edge_labels, alpha=0.8)
    plt.show()


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos


def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                 vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos,
                                 parent=root, parsed=parsed)
    return pos


def communication(cluster, node_a, node_b, data, weights=None):
    path = nx.shortest_path(cluster.topology, source=node_a, target=node_b, weight=weights)
    for i in range(len(path) - 1):
        cluster.topology[path[i]][path[i + 1]]['usage'] += data


def running_inspection(env):
    env.cluster.reset_communication()
    draw_topology(env.cluster)
