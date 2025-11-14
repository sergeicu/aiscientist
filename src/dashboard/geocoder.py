"""Geocode institution addresses to coordinates."""

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
