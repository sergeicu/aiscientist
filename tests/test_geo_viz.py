"""Tests for geographic visualization module."""

import pytest
import pandas as pd
from src.dashboard.geo_viz import GeoVisualizer


@pytest.fixture
def sample_geo_data():
    """Create sample geographic collaboration data."""
    return pd.DataFrame({
        'institution': ['MIT', 'Stanford', 'Harvard'],
        'latitude': [42.3601, 37.4275, 42.3770],
        'longitude': [-71.0942, -122.1697, -71.1167],
        'publications': [100, 150, 120]
    })


def test_create_visualizer():
    """Should initialize geo visualizer."""
    viz = GeoVisualizer()
    assert viz is not None


def test_create_institution_map(sample_geo_data):
    """Should create map with institution markers."""
    viz = GeoVisualizer()

    map_obj = viz.create_institution_map(sample_geo_data)

    assert map_obj is not None


def test_add_collaboration_lines(sample_geo_data):
    """Should add lines between collaborating institutions."""
    viz = GeoVisualizer()

    collaborations = pd.DataFrame({
        'inst1_lat': [42.3601],
        'inst1_lon': [-71.0942],
        'inst2_lat': [37.4275],
        'inst2_lon': [-122.1697],
        'strength': [10]
    })

    map_obj = viz.create_collaboration_map(
        institutions=sample_geo_data,
        collaborations=collaborations
    )

    assert map_obj is not None


def test_create_heatmap(sample_geo_data):
    """Should create heat map of publication density."""
    viz = GeoVisualizer()

    map_obj = viz.create_heatmap(
        sample_geo_data,
        weight_column='publications'
    )

    assert map_obj is not None


def test_save_map(sample_geo_data, tmp_path):
    """Should save map to HTML."""
    viz = GeoVisualizer()

    map_obj = viz.create_institution_map(sample_geo_data)

    output_path = tmp_path / "map.html"
    viz.save_map(map_obj, str(output_path))

    assert output_path.exists()
