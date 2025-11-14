"""Streamlit dashboard for research network and geographic visualizations."""

import streamlit as st
from src.dashboard.network_viz import NetworkVisualizer
from src.dashboard.geo_viz import GeoVisualizer
from src.dashboard.data_fetcher import DataFetcher
import plotly.graph_objects as go


def main():
    """Main Streamlit dashboard application."""
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

    # Initialize data fetcher
    try:
        fetcher = DataFetcher()
    except Exception as e:
        st.error(f"Could not connect to Neo4j: {e}")
        st.info("Make sure Neo4j is running and credentials are configured in environment variables.")
        return

    # Network visualization
    if viz_type in ["Network Graph", "Both"]:
        st.header("Collaboration Network")

        # Layout options
        layout = st.sidebar.selectbox(
            "Layout Algorithm",
            ["spring", "circular", "kamada_kawai", "random"]
        )

        # Node limit
        node_limit = st.sidebar.slider(
            "Maximum Nodes",
            min_value=10,
            max_value=500,
            value=100,
            step=10
        )

        # Node sizing option
        node_size_by = st.sidebar.selectbox(
            "Size Nodes By",
            ["None", "degree"],
            index=1
        )
        node_size_by = None if node_size_by == "None" else node_size_by

        # Edge width option
        edge_width_by = st.sidebar.selectbox(
            "Edge Width By",
            ["None", "weight"],
            index=1
        )
        edge_width_by = None if edge_width_by == "None" else edge_width_by

        # Fetch network data
        with st.spinner("Fetching collaboration network..."):
            graph = fetcher.get_collaboration_network(limit=node_limit)

        if graph.number_of_nodes() > 0:
            # Network filtering
            min_degree = st.sidebar.number_input(
                "Minimum Degree",
                min_value=0,
                max_value=10,
                value=0
            )

            # Create visualization
            viz = NetworkVisualizer()

            if min_degree > 0:
                graph = viz.filter_network(graph, min_degree=min_degree)

            st.info(
                f"Network: {graph.number_of_nodes()} nodes, "
                f"{graph.number_of_edges()} edges"
            )

            fig = viz.create_network_plot(
                graph,
                layout=layout,
                node_size_by=node_size_by,
                edge_width_by=edge_width_by
            )

            st.plotly_chart(fig, use_container_width=True)

            # Export option
            if st.button("Export Network to HTML"):
                viz.export_network(graph, "network_export.html")
                st.success("Network exported to network_export.html")
        else:
            st.warning("No network data available. Make sure Neo4j database has collaboration data.")

    # Geographic visualization
    if viz_type in ["Geographic Map", "Both"]:
        st.header("Geographic Distribution")

        # Map type selection
        map_type = st.sidebar.selectbox(
            "Map Type",
            ["Institution Markers", "Collaboration Lines", "Heat Map"]
        )

        # Fetch institution data
        with st.spinner("Fetching institution locations..."):
            institutions = fetcher.get_institutions_with_locations()

        if len(institutions) > 0:
            # Create map
            geo_viz = GeoVisualizer()

            if map_type == "Institution Markers":
                map_obj = geo_viz.create_institution_map(
                    institutions,
                    size_col='publications'
                )
            elif map_type == "Collaboration Lines":
                collaborations = fetcher.get_institution_collaborations()
                if len(collaborations) > 0:
                    map_obj = geo_viz.create_collaboration_map(
                        institutions=institutions,
                        collaborations=collaborations
                    )
                else:
                    st.warning("No collaboration data available.")
                    map_obj = geo_viz.create_institution_map(institutions)
            else:  # Heat Map
                map_obj = geo_viz.create_heatmap(
                    institutions,
                    weight_column='publications'
                )

            st.info(f"Showing {len(institutions)} institutions")

            # Display map
            st.components.v1.html(
                map_obj._repr_html_(),
                height=600
            )

            # Export option
            if st.button("Export Map to HTML"):
                geo_viz.save_map(map_obj, "map_export.html")
                st.success("Map exported to map_export.html")
        else:
            st.warning("No institution data available. Make sure Neo4j database has geocoded institutions.")

    # Close connection
    fetcher.close()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard visualizes research collaboration networks "
        "and geographic distribution of institutions using data from Neo4j."
    )


if __name__ == "__main__":
    main()
