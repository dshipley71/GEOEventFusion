"""Actor network visualization for GEOEventFusion.

Renders the co-occurrence graph with Hub/Broker/Peripheral role coloring
and community detection coloring. Rendering only — no data transformation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def render_actor_network_chart(
    actor_graph: Any,
    title: str = "Actor Co-occurrence Network",
    output_path: Optional[str | Path] = None,
    figsize: tuple = (12, 10),
    max_nodes: int = 80,
) -> Optional[str | Path]:
    """Render the actor co-occurrence network graph.

    Node size is proportional to PageRank. Node color encodes role:
    - Hub: Red
    - Broker: Amber
    - Peripheral: Gray

    Edge width is proportional to co-occurrence count.

    Args:
        actor_graph: ActorGraph dataclass object.
        title: Chart title string.
        output_path: If provided, save the chart to this path.
        figsize: Figure size tuple (width, height) in inches.
        max_nodes: Maximum nodes to display (selects top PageRank nodes).

    Returns:
        Output path if saved, None otherwise.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx

        from geoeventfusion.visualization.theme import (
            ROLE_BROKER_COLOR,
            ROLE_HUB_COLOR,
            ROLE_PERIPHERAL_COLOR,
            THEME_BACKGROUND,
            THEME_TEXT,
            THEME_ACCENT,
            get_dark_rcparams,
        )

        plt.rcParams.update(get_dark_rcparams())

        if not actor_graph or not actor_graph.nodes:
            logger.warning("Actor network chart: empty actor graph — skipping")
            return None

        # Build NetworkX graph
        G = nx.Graph()
        for edge in actor_graph.edges:
            G.add_edge(edge.actor_a, edge.actor_b, weight=edge.weight)

        if G.number_of_nodes() == 0:
            return None

        # Limit to top max_nodes by PageRank
        if G.number_of_nodes() > max_nodes:
            top_nodes = {
                n.name for n in sorted(
                    actor_graph.nodes, key=lambda x: x.pagerank, reverse=True
                )[:max_nodes]
            }
            nodes_to_remove = [n for n in G.nodes() if n not in top_nodes]
            G.remove_nodes_from(nodes_to_remove)

        if G.number_of_nodes() == 0:
            return None

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(THEME_BACKGROUND)
        fig.patch.set_facecolor(THEME_BACKGROUND)

        # Layout
        try:
            pos = nx.spring_layout(G, seed=42, k=2.0 / (G.number_of_nodes() ** 0.5))
        except Exception:
            pos = nx.kamada_kawai_layout(G)

        # Node properties from actor_graph
        node_by_name = actor_graph.node_by_name
        node_colors = []
        node_sizes = []
        for node_name in G.nodes():
            actor = node_by_name.get(node_name)
            if actor:
                if actor.role == "Hub":
                    node_colors.append(ROLE_HUB_COLOR)
                elif actor.role == "Broker":
                    node_colors.append(ROLE_BROKER_COLOR)
                else:
                    node_colors.append(ROLE_PERIPHERAL_COLOR)
                node_sizes.append(200 + actor.pagerank * 5000)
            else:
                node_colors.append(ROLE_PERIPHERAL_COLOR)
                node_sizes.append(200)

        # Edge widths
        edge_widths = []
        for u, v, data in G.edges(data=True):
            edge_widths.append(0.5 + min(data.get("weight", 1), 10) * 0.2)

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=edge_widths,
            edge_color=THEME_ACCENT,
            alpha=0.3,
        )
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.85,
        )

        # Label top-20 nodes by PageRank
        top_label_nodes = {
            n.name for n in sorted(
                actor_graph.nodes, key=lambda x: x.pagerank, reverse=True
            )[:20]
            if n.name in G.nodes()
        }
        labels = {n: n for n in G.nodes() if n in top_label_nodes}
        nx.draw_networkx_labels(
            G, pos, labels=labels, ax=ax,
            font_size=7,
            font_color=THEME_TEXT,
        )

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=ROLE_HUB_COLOR, label="Hub"),
            Patch(facecolor=ROLE_BROKER_COLOR, label="Broker"),
            Patch(facecolor=ROLE_PERIPHERAL_COLOR, label="Peripheral"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            fontsize=9,
        )

        ax.set_title(title, color=THEME_TEXT, pad=12, fontsize=12)
        ax.axis("off")

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
            logger.info("Saved actor network chart: %s", output_path)
            plt.close(fig)
            return output_path

        plt.close(fig)
        return None

    except ImportError:
        logger.warning("matplotlib and networkx are required for actor network chart")
        return None
    except Exception as exc:
        logger.error("Actor network chart rendering failed: %s", exc)
        return None
