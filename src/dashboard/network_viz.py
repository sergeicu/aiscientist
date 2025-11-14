"""Create interactive network visualizations."""

import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Optional
from loguru import logger


class NetworkVisualizer:
    """
    Create interactive network visualizations.

    Uses NetworkX for graph operations and Plotly for rendering.

    Example:
        >>> viz = NetworkVisualizer()
        >>> fig = viz.create_network_plot(graph, layout='spring')
    """

    def __init__(self):
        self.layout_functions = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'random': nx.random_layout
        }

    def create_network_plot(
        self,
        graph: nx.Graph,
        layout: str = 'spring',
        node_size_by: Optional[str] = None,
        edge_width_by: Optional[str] = None,
        title: str = 'Collaboration Network'
    ) -> go.Figure:
        """
        Create interactive network visualization.

        Args:
            graph: NetworkX graph
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
            node_size_by: Node attribute for sizing ('degree', 'pagerank', etc.)
            edge_width_by: Edge attribute for width ('weight', etc.)
            title: Plot title

        Returns:
            Plotly Figure
        """
        logger.info(
            f"Creating network plot: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )

        # Calculate layout
        layout_func = self.layout_functions.get(layout, nx.spring_layout)
        pos = layout_func(graph)

        # Create edge traces
        edge_trace = self._create_edge_trace(graph, pos, edge_width_by)

        # Create node traces
        node_trace = self._create_node_trace(graph, pos, node_size_by)

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=1000,
                height=800
            )
        )

        return fig

    def _create_edge_trace(
        self,
        graph: nx.Graph,
        pos: Dict,
        width_by: Optional[str]
    ) -> go.Scatter:
        """Create edge trace for plot."""
        edge_x = []
        edge_y = []
        edge_widths = []

        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            # Edge width
            if width_by and width_by in graph.edges[edge]:
                width = graph.edges[edge][width_by]
            else:
                width = 1

            edge_widths.extend([width, width, width])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        return edge_trace

    def _create_node_trace(
        self,
        graph: nx.Graph,
        pos: Dict,
        size_by: Optional[str]
    ) -> go.Scatter:
        """Create node trace for plot."""
        node_x = []
        node_y = []
        node_sizes = []
        node_text = []

        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Node size
            if size_by == 'degree':
                size = graph.degree(node) * 5 + 10
            elif size_by and size_by in graph.nodes[node]:
                size = graph.nodes[node][size_by] * 20
            else:
                size = 20

            node_sizes.append(size)

            # Hover text
            node_name = graph.nodes[node].get('name', str(node))
            node_text.append(
                f"{node_name}<br>"
                f"Connections: {graph.degree(node)}"
            )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_sizes,
                color='#1f77b4',
                line_width=2,
                line_color='white'
            )
        )

        return node_trace

    def filter_network(
        self,
        graph: nx.Graph,
        min_degree: Optional[int] = None,
        max_nodes: Optional[int] = None
    ) -> nx.Graph:
        """
        Filter network by criteria.

        Args:
            graph: Input graph
            min_degree: Minimum node degree
            max_nodes: Maximum nodes to keep (keeps highest degree)

        Returns:
            Filtered graph
        """
        filtered = graph.copy()

        # Filter by degree
        if min_degree:
            nodes_to_remove = [
                node for node, degree in dict(graph.degree()).items()
                if degree < min_degree
            ]
            filtered.remove_nodes_from(nodes_to_remove)

        # Limit to top nodes by degree
        if max_nodes and filtered.number_of_nodes() > max_nodes:
            degrees = dict(filtered.degree())
            top_nodes = sorted(
                degrees,
                key=degrees.get,
                reverse=True
            )[:max_nodes]

            nodes_to_remove = set(filtered.nodes()) - set(top_nodes)
            filtered.remove_nodes_from(nodes_to_remove)

        logger.info(
            f"Filtered network: {filtered.number_of_nodes()} nodes, "
            f"{filtered.number_of_edges()} edges"
        )

        return filtered

    def export_network(self, graph: nx.Graph, path: str):
        """Export network visualization to HTML."""
        fig = self.create_network_plot(graph)
        fig.write_html(path)
        logger.info(f"✓ Exported network to {path}")
