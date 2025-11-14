import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Optional
from loguru import logger

class InteractivePlotter:
    """
    Create interactive Plotly visualizations.

    Generates scatter plots with hover information,
    filtering, and customization options.

    Example:
        >>> plotter = InteractivePlotter()
        >>> fig = plotter.create_scatter_2d(df, x='x', y='y', color='cluster')
    """

    def __init__(self):
        self.default_colors = px.colors.qualitative.Set3

    def create_scatter_2d(
        self,
        data: pd.DataFrame,
        x: str = 'x',
        y: str = 'y',
        color: Optional[str] = None,
        hover_data: Optional[List[str]] = None,
        cluster_labels: Optional[Dict[int, str]] = None,
        title: str = 'UMAP Projection',
        width: int = 1000,
        height: int = 800
    ) -> go.Figure:
        """
        Create 2D scatter plot.

        Args:
            data: DataFrame with coordinates and metadata
            x: X coordinate column
            y: Y coordinate column
            color: Column to color by
            hover_data: Additional columns to show on hover
            cluster_labels: Mapping of cluster IDs to names
            title: Plot title
            width: Plot width
            height: Plot height

        Returns:
            Plotly Figure
        """
        logger.info(f"Creating 2D scatter plot ({len(data)} points)...")

        # Map cluster labels if provided
        if cluster_labels and color:
            data = data.copy()
            data[f'{color}_label'] = data[color].map(cluster_labels)
            color = f'{color}_label'

        # Create scatter plot
        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=color,
            hover_data=hover_data,
            title=title,
            width=width,
            height=height
        )

        # Customize
        fig.update_traces(marker=dict(size=5, opacity=0.7))

        fig.update_layout(
            hovermode='closest',
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            legend_title=color if color else 'Cluster'
        )

        return fig

    def create_scatter_3d(
        self,
        data: pd.DataFrame,
        x: str = 'x',
        y: str = 'y',
        z: str = 'z',
        color: Optional[str] = None,
        hover_data: Optional[List[str]] = None,
        title: str = 'UMAP 3D Projection'
    ) -> go.Figure:
        """Create 3D scatter plot."""
        logger.info(f"Creating 3D scatter plot ({len(data)} points)...")

        fig = px.scatter_3d(
            data,
            x=x,
            y=y,
            z=z,
            color=color,
            hover_data=hover_data,
            title=title
        )

        fig.update_traces(marker=dict(size=3, opacity=0.6))

        return fig

    def save_html(self, fig: go.Figure, path: str):
        """Save plot as HTML file."""
        fig.write_html(path)
        logger.info(f"✓ Saved plot to {path}")
