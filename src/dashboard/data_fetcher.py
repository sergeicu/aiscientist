"""Fetch data from Neo4j graph database for visualizations."""

from typing import Optional, Dict, List
import pandas as pd
import networkx as nx
from neo4j import GraphDatabase
from loguru import logger
import os


class DataFetcher:
    """
    Fetch network and geographic data from Neo4j.

    Connects to Neo4j database and retrieves collaboration networks
    and institution data for visualizations.

    Example:
        >>> fetcher = DataFetcher()
        >>> graph = fetcher.get_collaboration_network(limit=100)
        >>> institutions = fetcher.get_institutions_with_locations()
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize data fetcher.

        Args:
            uri: Neo4j connection URI (default from NEO4J_URI env var)
            user: Neo4j username (default from NEO4J_USER env var)
            password: Neo4j password (default from NEO4J_PASSWORD env var)
        """
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = user or os.getenv('NEO4J_USER', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'password')

        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}")
            self.driver = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            self.driver.close()

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")

    def get_collaboration_network(
        self,
        limit: Optional[int] = None
    ) -> nx.Graph:
        """
        Get author collaboration network from Neo4j.

        Args:
            limit: Maximum number of nodes to fetch

        Returns:
            NetworkX graph with author collaboration network
        """
        if not self.driver:
            logger.warning("No Neo4j connection, returning empty graph")
            return nx.Graph()

        query = """
        MATCH (a1:Author)-[r:COLLABORATED_WITH]->(a2:Author)
        RETURN a1.id as author1_id, a1.name as author1_name,
               a2.id as author2_id, a2.name as author2_name,
               r.publications as weight
        LIMIT $limit
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, limit=limit or 1000)

                # Build NetworkX graph
                G = nx.Graph()

                for record in result:
                    # Add nodes
                    G.add_node(
                        record['author1_id'],
                        name=record['author1_name']
                    )
                    G.add_node(
                        record['author2_id'],
                        name=record['author2_name']
                    )

                    # Add edge
                    G.add_edge(
                        record['author1_id'],
                        record['author2_id'],
                        weight=record['weight']
                    )

                logger.info(
                    f"Fetched collaboration network: {G.number_of_nodes()} nodes, "
                    f"{G.number_of_edges()} edges"
                )

                return G

        except Exception as e:
            logger.error(f"Error fetching collaboration network: {e}")
            return nx.Graph()

    def get_institutions_with_locations(self) -> pd.DataFrame:
        """
        Get institutions with geographic coordinates.

        Returns:
            DataFrame with columns: institution, latitude, longitude, publications
        """
        if not self.driver:
            logger.warning("No Neo4j connection, returning empty dataframe")
            return pd.DataFrame(columns=['institution', 'latitude', 'longitude', 'publications'])

        query = """
        MATCH (i:Institution)
        WHERE i.latitude IS NOT NULL AND i.longitude IS NOT NULL
        RETURN i.name as institution,
               i.latitude as latitude,
               i.longitude as longitude,
               i.publication_count as publications
        """

        try:
            with self.driver.session() as session:
                result = session.run(query)

                data = [
                    {
                        'institution': record['institution'],
                        'latitude': record['latitude'],
                        'longitude': record['longitude'],
                        'publications': record['publications'] or 0
                    }
                    for record in result
                ]

                df = pd.DataFrame(data)
                logger.info(f"Fetched {len(df)} institutions with locations")

                return df

        except Exception as e:
            logger.error(f"Error fetching institutions: {e}")
            return pd.DataFrame(columns=['institution', 'latitude', 'longitude', 'publications'])

    def get_institution_collaborations(self) -> pd.DataFrame:
        """
        Get collaboration connections between institutions.

        Returns:
            DataFrame with columns: inst1_lat, inst1_lon, inst2_lat, inst2_lon, strength
        """
        if not self.driver:
            logger.warning("No Neo4j connection, returning empty dataframe")
            return pd.DataFrame(columns=['inst1_lat', 'inst1_lon', 'inst2_lat', 'inst2_lon', 'strength'])

        query = """
        MATCH (i1:Institution)-[r:COLLABORATED]->(i2:Institution)
        WHERE i1.latitude IS NOT NULL AND i1.longitude IS NOT NULL
          AND i2.latitude IS NOT NULL AND i2.longitude IS NOT NULL
        RETURN i1.latitude as inst1_lat, i1.longitude as inst1_lon,
               i2.latitude as inst2_lat, i2.longitude as inst2_lon,
               r.strength as strength
        LIMIT 100
        """

        try:
            with self.driver.session() as session:
                result = session.run(query)

                data = [
                    {
                        'inst1_lat': record['inst1_lat'],
                        'inst1_lon': record['inst1_lon'],
                        'inst2_lat': record['inst2_lat'],
                        'inst2_lon': record['inst2_lon'],
                        'strength': record['strength'] or 1
                    }
                    for record in result
                ]

                df = pd.DataFrame(data)
                logger.info(f"Fetched {len(df)} institution collaborations")

                return df

        except Exception as e:
            logger.error(f"Error fetching collaborations: {e}")
            return pd.DataFrame(columns=['inst1_lat', 'inst1_lon', 'inst2_lat', 'inst2_lon', 'strength'])
