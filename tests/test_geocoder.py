"""Tests for geocoder module."""

import pytest
from unittest.mock import Mock, patch
from src.dashboard.geocoder import Geocoder


@pytest.fixture
def mock_location():
    """Mock location object."""
    location = Mock()
    location.latitude = 42.3601
    location.longitude = -71.0942
    location.address = "Boston Children's Hospital, Boston, MA"
    return location


def test_geocode_institution(mock_location):
    """Should geocode institution to coordinates."""
    geocoder = Geocoder()

    with patch.object(geocoder.geolocator, 'geocode', return_value=mock_location):
        location = geocoder.geocode_institution("Boston Children's Hospital, Boston, MA")

    assert location is not None
    assert 'latitude' in location
    assert 'longitude' in location
    assert 42.0 < location['latitude'] < 43.0  # Boston latitude range
    assert -72.0 < location['longitude'] < -70.0  # Boston longitude range


def test_geocode_with_caching(mock_location):
    """Should cache geocoding results."""
    geocoder = Geocoder()

    with patch.object(geocoder.geolocator, 'geocode', return_value=mock_location) as mock_geocode:
        # First call
        loc1 = geocoder.geocode_institution("MIT, Cambridge, MA")

        # Second call (should use cache)
        loc2 = geocoder.geocode_institution("MIT, Cambridge, MA")

        # Geocode should only be called once due to caching
        assert mock_geocode.call_count == 1
        assert loc1 == loc2


def test_handle_geocoding_failure():
    """Should handle failed geocoding gracefully."""
    geocoder = Geocoder()

    with patch.object(geocoder.geolocator, 'geocode', return_value=None):
        location = geocoder.geocode_institution("Invalid Location XYZ123")

    # Should return None
    assert location is None


def test_batch_geocode(mock_location):
    """Should geocode multiple institutions."""
    geocoder = Geocoder()

    institutions = [
        "Harvard Medical School, Boston, MA",
        "Stanford University, Stanford, CA",
        "MIT, Cambridge, MA"
    ]

    with patch.object(geocoder.geolocator, 'geocode', return_value=mock_location):
        locations = geocoder.batch_geocode(institutions, delay=0)  # No delay for tests

    assert len(locations) == 3
    assert all('latitude' in loc for loc in locations if loc is not None)


def test_reverse_geocode(mock_location):
    """Should get location name from coordinates."""
    geocoder = Geocoder()

    with patch.object(geocoder.geolocator, 'reverse', return_value=mock_location):
        name = geocoder.reverse_geocode(latitude=42.3601, longitude=-71.0589)

    assert name is not None
    assert isinstance(name, str)
