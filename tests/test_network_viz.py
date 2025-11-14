"""Tests for network visualization module."""

import pytest
import networkx as nx
from src.dashboard.network_viz import NetworkVisualizer


@pytest.fixture
def sample_network():
    """Create sample collaboration network."""
    G = nx.Graph()

    # Add nodes (authors)
    G.add_node('author_1', name='John Smith')
    G.add_node('author_2', name='Jane Doe')
    G.add_node('author_3', name='Alice Brown')

    # Add edges (collaborations)
    G.add_edge('author_1', 'author_2', weight=5)
    G.add_edge('author_2', 'author_3', weight=3)
    G.add_edge('author_1', 'author_3', weight=2)

    return G


def test_create_visualizer():
    """Should initialize network visualizer."""
    viz = NetworkVisualizer()
    assert viz is not None


def test_create_network_plot(sample_network):
    """Should create interactive network plot."""
    viz = NetworkVisualizer()

    fig = viz.create_network_plot(sample_network)

    assert fig is not None
    assert len(fig.data) > 0  # Has traces


def test_layout_options(sample_network):
    """Should support different layout algorithms."""
    viz = NetworkVisualizer()

    layouts = ['spring', 'circular', 'kamada_kawai']

    for layout in layouts:
        fig = viz.create_network_plot(sample_network, layout=layout)
        assert fig is not None


def test_node_sizing_by_degree(sample_network):
    """Should size nodes by degree centrality."""
    viz = NetworkVisualizer()

    fig = viz.create_network_plot(
        sample_network,
        node_size_by='degree'
    )

    assert fig is not None


def test_edge_width_by_weight(sample_network):
    """Should vary edge width by weight."""
    viz = NetworkVisualizer()

    fig = viz.create_network_plot(
        sample_network,
        edge_width_by='weight'
    )

    assert fig is not None


def test_filter_network(sample_network):
    """Should filter network by criteria."""
    viz = NetworkVisualizer()

    # Filter to nodes with degree >= 2
    filtered = viz.filter_network(
        sample_network,
        min_degree=2
    )

    assert filtered.number_of_nodes() <= sample_network.number_of_nodes()


def test_export_network(sample_network, tmp_path):
    """Should export network to file."""
    viz = NetworkVisualizer()

    output_path = tmp_path / "network.html"

    viz.export_network(sample_network, str(output_path))

    assert output_path.exists()
