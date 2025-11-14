# Task 7: Network & Geographic Visualization Dashboard - TDD Implementation

## Executive Summary

Implement an interactive Streamlit dashboard for visualizing research collaboration networks and geographic distribution of institutions. This combines graph visualizations (author/institution networks) with geographic maps showing research collaborations across locations.

**Key Requirements:**
- Fetch network data from Neo4j graph database
- Create interactive network visualizations using networkx + plotly
- Build geographic collaboration maps using Folium/Plotly
- Geocode institution locations
- Implement Streamlit dashboard with filters and controls
- Support exporting visualizations
- Follow Test-Driven Development (TDD) principles

**Dependencies**: Requires Task 4 (Neo4j Graph Setup) and Task 5 (Graph Analytics) to be completed first.

## Background & Context

### Why Network + Geographic Visualization?

**Research Intelligence Benefits:**
- **Collaboration Discovery**: See who works with whom
- **Geographic Patterns**: Identify regional research hubs
- **Partnership Opportunities**: Find institutions in specific locations
- **Impact Assessment**: Visualize reach of research networks
- **Strategic Planning**: Understand collaboration landscapes

### Visualization Types

**1. Network Graphs:**
- Author collaboration networks
- Institution partnership networks
- Citation networks

**2. Geographic Maps:**
- Institution locations
- Collaboration flows between cities/countries
- Heat maps of publication density

## Technical Architecture

### Module Structure

```
src/dashboard/
├── __init__.py
├── network_viz.py       # Network graph visualizations
├── geo_viz.py           # Geographic visualizations
├── geocoder.py          # Geocoding service
├── data_fetcher.py      # Fetch data from Neo4j
└── app.py               # Streamlit dashboard

tests/
├── __init__.py
├── test_network_viz.py
├── test_geo_viz.py
├── test_geocoder.py
├── test_data_fetcher.py
└── fixtures/
    └── sample_network_data.py
```

### Dependencies

```python
# Required packages
streamlit>=1.28           # Dashboard framework
networkx>=3.2             # Network analysis
plotly>=5.17              # Interactive plots
folium>=0.15              # Map visualizations
geopy>=2.4                # Geocoding
neo4j>=5.14               # Database connection
pandas>=2.0               # Data manipulation
pydantic>=2.0             # Data validation
pytest>=7.4               # Testing
pytest-mock>=3.12         # Mocking
```

## TDD Implementation Plan

### Phase 1: Geocoder (TDD)

#### Test Cases

```python
# tests/test_geocoder.py

import pytest
from src.dashboard.geocoder import Geocoder

def test_geocode_institution():
    """Should geocode institution to coordinates."""
    geocoder = Geocoder()

    location = geocoder.geocode_institution("Boston Children's Hospital, Boston, MA")

    assert location is not None
    assert 'latitude' in location
    assert 'longitude' in location
    assert 42.0 < location['latitude'] < 43.0  # Boston latitude range
    assert -72.0 < location['longitude'] < -70.0  # Boston longitude range

def test_geocode_with_caching():
    """Should cache geocoding results."""
    geocoder = Geocoder()

    # First call
    loc1 = geocoder.geocode_institution("MIT, Cambridge, MA")

    # Second call (should use cache)
    loc2 = geocoder.geocode_institution("MIT, Cambridge, MA")

    assert loc1 == loc2

def test_handle_geocoding_failure():
    """Should handle failed geocoding gracefully."""
    geocoder = Geocoder()

    location = geocoder.geocode_institution("Invalid Location XYZ123")

    # Should return None or default location
    assert location is None or 'latitude' in location

def test_batch_geocode():
    """Should geocode multiple institutions."""
    geocoder = Geocoder()

    institutions = [
        "Harvard Medical School, Boston, MA",
        "Stanford University, Stanford, CA",
        "MIT, Cambridge, MA"
    ]

    locations = geocoder.batch_geocode(institutions)

    assert len(locations) == 3
    assert all('latitude' in loc for loc in locations)

def test_reverse_geocode():
    """Should get location name from coordinates."""
    geocoder = Geocoder()

    name = geocoder.reverse_geocode(latitude=42.3601, longitude=-71.0589)

    assert name is not None
    assert isinstance(name, str)
```

#### Implementation

```python
# src/dashboard/geocoder.py

from typing import Dict, List, Optional
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from loguru import logger
import time

class Geocoder:
    """
    Geocode institution addresses to coordinates.

    Uses OpenStreetMap Nominatim (free, no API key required).
    Implements caching to avoid repeated lookups.

    Example:
        >>> geocoder = Geocoder()
        >>> location = geocoder.geocode_institution("MIT, Cambridge, MA")
        >>> print(location['latitude'], location['longitude'])
    """

    def __init__(self, user_agent: str = "research_network_viz"):
        self.geolocator = Nominatim(user_agent=user_agent)
        self.cache = {}  # Simple in-memory cache

    def geocode_institution(
        self,
        address: str,
        timeout: int = 10
    ) -> Optional[Dict]:
        """
        Geocode institution address to coordinates.

        Args:
            address: Institution address string
            timeout: Request timeout in seconds

        Returns:
            Dictionary with latitude, longitude, or None
        """
        # Check cache
        if address in self.cache:
            logger.debug(f"Cache hit for: {address}")
            return self.cache[address]

        try:
            logger.debug(f"Geocoding: {address}")

            location = self.geolocator.geocode(address, timeout=timeout)

            if location:
                result = {
                    'latitude': location.latitude,
                    'longitude': location.longitude,
                    'address': location.address
                }

                # Cache result
                self.cache[address] = result

                # Rate limit (Nominatim requires 1 request/second)
                time.sleep(1)

                return result
            else:
                logger.warning(f"Could not geocode: {address}")
                return None

        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.error(f"Geocoding error for {address}: {e}")
            return None

    def batch_geocode(
        self,
        addresses: List[str],
        delay: float = 1.0
    ) -> List[Optional[Dict]]:
        """
        Geocode multiple addresses.

        Args:
            addresses: List of address strings
            delay: Delay between requests (seconds)

        Returns:
            List of location dictionaries
        """
        logger.info(f"Geocoding {len(addresses)} addresses...")

        locations = []

        for i, address in enumerate(addresses):
            if i > 0:
                time.sleep(delay)  # Rate limiting

            location = self.geocode_institution(address)
            locations.append(location)

            if (i + 1) % 10 == 0:
                logger.info(f"Geocoded {i + 1}/{len(addresses)}")

        return locations

    def reverse_geocode(
        self,
        latitude: float,
        longitude: float
    ) -> Optional[str]:
        """
        Get location name from coordinates.

        Args:
            latitude: Latitude
            longitude: Longitude

        Returns:
            Location name string or None
        """
        try:
            location = self.geolocator.reverse(
                (latitude, longitude),
                timeout=10
            )

            if location:
                return location.address

            return None

        except Exception as e:
            logger.error(f"Reverse geocoding error: {e}")
            return None
```

### Phase 2: Network Visualization (TDD)

#### Test Cases

```python
# tests/test_network_viz.py

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
```

#### Implementation

```python
# src/dashboard/network_viz.py

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
```

### Phase 3: Geographic Visualization (TDD)

#### Test Cases

```python
# tests/test_geo_viz.py

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
```

#### Implementation

```python
# src/dashboard/geo_viz.py

import folium
from folium.plugins import HeatMap
import pandas as pd
from typing import Optional
from loguru import logger

class GeoVisualizer:
    """
    Create geographic visualizations of research collaborations.

    Uses Folium for interactive maps.

    Example:
        >>> viz = GeoVisualizer()
        >>> map = viz.create_institution_map(institutions_df)
    """

    def __init__(self):
        self.default_zoom = 4

    def create_institution_map(
        self,
        data: pd.DataFrame,
        lat_col: str = 'latitude',
        lon_col: str = 'longitude',
        name_col: str = 'institution',
        size_col: Optional[str] = None,
        center: Optional[tuple] = None
    ) -> folium.Map:
        """
        Create map with institution markers.

        Args:
            data: DataFrame with institution data
            lat_col: Latitude column name
            lon_col: Longitude column name
            name_col: Institution name column
            size_col: Column for marker sizing
            center: Map center (lat, lon)

        Returns:
            Folium Map object
        """
        logger.info(f"Creating institution map ({len(data)} institutions)...")

        # Calculate center if not provided
        if center is None:
            center = (
                data[lat_col].mean(),
                data[lon_col].mean()
            )

        # Create map
        m = folium.Map(
            location=center,
            zoom_start=self.default_zoom,
            tiles='OpenStreetMap'
        )

        # Add markers
        for _, row in data.iterrows():
            # Marker size
            if size_col and size_col in row:
                radius = min(max(row[size_col] / 10, 5), 20)
            else:
                radius = 8

            # Create marker
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=radius,
                popup=row[name_col],
                color='blue',
                fill=True,
                fillColor='blue',
                fillOpacity=0.6
            ).add_to(m)

        return m

    def create_collaboration_map(
        self,
        institutions: pd.DataFrame,
        collaborations: pd.DataFrame,
        center: Optional[tuple] = None
    ) -> folium.Map:
        """
        Create map with collaboration lines.

        Args:
            institutions: Institution data
            collaborations: Collaboration data with coordinates
            center: Map center

        Returns:
            Folium Map object
        """
        logger.info(
            f"Creating collaboration map ({len(collaborations)} connections)..."
        )

        # Create base map with institutions
        m = self.create_institution_map(institutions, center=center)

        # Add collaboration lines
        for _, row in collaborations.iterrows():
            # Draw line between institutions
            folium.PolyLine(
                locations=[
                    [row['inst1_lat'], row['inst1_lon']],
                    [row['inst2_lat'], row['inst2_lon']]
                ],
                color='red',
                weight=row.get('strength', 1),
                opacity=0.5
            ).add_to(m)

        return m

    def create_heatmap(
        self,
        data: pd.DataFrame,
        lat_col: str = 'latitude',
        lon_col: str = 'longitude',
        weight_column: Optional[str] = None,
        center: Optional[tuple] = None
    ) -> folium.Map:
        """
        Create heat map of publication density.

        Args:
            data: DataFrame with location data
            lat_col: Latitude column
            lon_col: Longitude column
            weight_column: Column for heat intensity
            center: Map center

        Returns:
            Folium Map with heatmap layer
        """
        logger.info(f"Creating heat map ({len(data)} points)...")

        if center is None:
            center = (data[lat_col].mean(), data[lon_col].mean())

        m = folium.Map(location=center, zoom_start=self.default_zoom)

        # Prepare heat data
        if weight_column:
            heat_data = [
                [row[lat_col], row[lon_col], row[weight_column]]
                for _, row in data.iterrows()
            ]
        else:
            heat_data = [
                [row[lat_col], row[lon_col]]
                for _, row in data.iterrows()
            ]

        # Add heatmap layer
        HeatMap(heat_data).add_to(m)

        return m

    def save_map(self, map_obj: folium.Map, path: str):
        """Save map to HTML file."""
        map_obj.save(path)
        logger.info(f"✓ Saved map to {path}")
```

## Streamlit Dashboard

```python
# src/dashboard/app.py

import streamlit as st
from src.dashboard.network_viz import NetworkVisualizer
from src.dashboard.geo_viz import GeoVisualizer
from src.dashboard.data_fetcher import DataFetcher
import plotly.graph_objects as go

def main():
    st.set_page_config(
        page_title="Research Network Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🔬 Research Network Dashboard")

    # Sidebar
    st.sidebar.header("Filters")

    viz_type = st.sidebar.selectbox(
        "Visualization Type",
        ["Network Graph", "Geographic Map", "Both"]
    )

    # Fetch data
    fetcher = DataFetcher()

    # Network visualization
    if viz_type in ["Network Graph", "Both"]:
        st.header("Collaboration Network")

        layout = st.sidebar.selectbox(
            "Layout",
            ["spring", "circular", "kamada_kawai"]
        )

        # Fetch network data
        graph = fetcher.get_collaboration_network(limit=100)

        # Create visualization
        viz = NetworkVisualizer()
        fig = viz.create_network_plot(graph, layout=layout)

        st.plotly_chart(fig, use_container_width=True)

    # Geographic visualization
    if viz_type in ["Geographic Map", "Both"]:
        st.header("Geographic Distribution")

        # Fetch institution data
        institutions = fetcher.get_institutions_with_locations()

        # Create map
        geo_viz = GeoVisualizer()
        map_obj = geo_viz.create_institution_map(institutions)

        # Display map
        st.components.v1.html(
            map_obj._repr_html_(),
            height=600
        )

if __name__ == "__main__":
    main()
```

## Running Tests

```bash
# Install dependencies
pip install streamlit networkx plotly folium geopy neo4j pandas pydantic pytest

# Run tests
pytest tests/ -v --cov=src/dashboard

# Run Streamlit app
streamlit run src/dashboard/app.py
```

## Usage Examples

### Network Visualization

```python
from src.dashboard.network_viz import NetworkVisualizer
import networkx as nx

# Create sample network
G = nx.karate_club_graph()

# Visualize
viz = NetworkVisualizer()
fig = viz.create_network_plot(G, layout='spring', node_size_by='degree')
fig.show()
```

### Geographic Map

```python
from src.dashboard.geo_viz import GeoVisualizer
import pandas as pd

# Sample data
data = pd.DataFrame({
    'institution': ['MIT', 'Stanford'],
    'latitude': [42.3601, 37.4275],
    'longitude': [-71.0942, -122.1697],
    'publications': [100, 150]
})

# Create map
viz = GeoVisualizer()
map_obj = viz.create_institution_map(data, size_col='publications')
map_obj.save('institutions.html')
```

## Success Criteria

✅ **Must Have:**
1. All unit tests passing (>90% coverage)
2. Network graph visualization working
3. Geographic map visualization working
4. Geocoding service functional
5. Streamlit dashboard runs
6. Data fetches from Neo4j

✅ **Should Have:**
7. Multiple layout algorithms
8. Interactive filtering
9. Export capabilities
10. Collaboration flow visualization

✅ **Nice to Have:**
11. Real-time updates
12. Custom styling
13. Performance optimization

## Deliverables

1. **Source code** (>90% coverage)
2. **Tests**
3. **Streamlit dashboard**
4. **Sample visualizations**
5. **Documentation**

## Environment Setup

```bash
# Install dependencies
pip install streamlit networkx plotly folium geopy neo4j pandas pytest

# Run tests
pytest tests/ -v --cov=src/dashboard

# Run app
streamlit run src/dashboard/app.py
```

---

**Task completion**: When all tests pass, network graphs display correctly, geographic maps show institutions, and Streamlit dashboard runs without errors.
