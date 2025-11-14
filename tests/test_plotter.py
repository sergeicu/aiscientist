import pytest
import numpy as np
import pandas as pd
from src.visualization.plotter import InteractivePlotter

@pytest.fixture
def sample_data():
    """Create sample data for plotting."""
    np.random.seed(42)

    return pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'cluster': np.random.randint(0, 5, 100),
        'pmid': [f'{i:08d}' for i in range(100)],
        'title': [f'Paper {i}' for i in range(100)],
        'year': np.random.choice(['2020', '2021', '2022', '2023'], 100)
    })

def test_create_plotter():
    """Should initialize plotter."""
    plotter = InteractivePlotter()
    assert plotter is not None

def test_create_2d_scatter(sample_data):
    """Should create 2D scatter plot."""
    plotter = InteractivePlotter()

    fig = plotter.create_scatter_2d(
        data=sample_data,
        x='x',
        y='y',
        color='cluster',
        hover_data=['pmid', 'title']
    )

    assert fig is not None
    assert len(fig.data) > 0  # Has traces

def test_create_3d_scatter(sample_data):
    """Should create 3D scatter plot."""
    sample_data['z'] = np.random.randn(100)

    plotter = InteractivePlotter()

    fig = plotter.create_scatter_3d(
        data=sample_data,
        x='x',
        y='y',
        z='z',
        color='cluster'
    )

    assert fig is not None

def test_add_cluster_labels(sample_data):
    """Should add cluster labels to plot."""
    plotter = InteractivePlotter()

    cluster_labels = {
        0: 'Immunotherapy',
        1: 'Gene Therapy',
        2: 'Drug Discovery'
    }

    fig = plotter.create_scatter_2d(
        data=sample_data,
        x='x',
        y='y',
        color='cluster',
        cluster_labels=cluster_labels
    )

    # Check if labels appear in legend
    assert fig is not None

def test_save_plot(sample_data, tmp_path):
    """Should save plot to HTML."""
    plotter = InteractivePlotter()

    fig = plotter.create_scatter_2d(
        data=sample_data,
        x='x',
        y='y',
        color='cluster'
    )

    output_path = tmp_path / "plot.html"
    plotter.save_html(fig, str(output_path))

    assert output_path.exists()
