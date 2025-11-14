"""Create geographic visualizations of research collaborations."""

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
