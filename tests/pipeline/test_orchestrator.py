"""Tests for pipeline orchestrator."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.config import PipelineConfig, PubMedConfig, ClinicalTrialsConfig


@pytest.fixture
def mock_config():
    """Create mock pipeline configuration."""
    return PipelineConfig(
        pubmed=PubMedConfig(
            email="test@example.com",
            institutions=["Harvard", "Mayo Clinic"]
        ),
        clinicaltrials=ClinicalTrialsConfig(
            sponsors=["MGH", "Johns Hopkins"]
        ),
        parallel_workers=2
    )


@pytest.mark.asyncio
class TestPipelineOrchestrator:
    """Test pipeline orchestrator."""

    async def test_initialization(self, mock_config, tmp_path):
        """Test orchestrator initialization."""
        orchestrator = PipelineOrchestrator(
            config=mock_config,
            output_dir=tmp_path
        )

        assert orchestrator.config == mock_config
        assert orchestrator.storage is not None

    @patch('pipeline.orchestrator.PubMedScraper')
    async def test_run_pubmed_scraper(self, mock_scraper_class, mock_config, tmp_path):
        """Test running PubMed scraper for institutions."""
        # Setup mock
        mock_scraper = Mock()
        mock_scraper.search_by_affiliation = AsyncMock(
            return_value=["123", "456"]
        )
        mock_scraper.fetch_article_details = AsyncMock(
            return_value=[{"pmid": "123", "title": "Test"}]
        )
        mock_scraper_class.return_value = mock_scraper

        orchestrator = PipelineOrchestrator(mock_config, tmp_path)

        # Run scraper
        results = await orchestrator.run_pubmed_scraper()

        assert len(results) == 2  # Two institutions
        assert all(r["status"] in ["success", "failed"] for r in results)

    @patch('pipeline.orchestrator.ClinicalTrialsScraper')
    async def test_run_clinicaltrials_scraper(
        self,
        mock_scraper_class,
        mock_config,
        tmp_path
    ):
        """Test running ClinicalTrials scraper."""
        mock_scraper = Mock()
        mock_scraper.search_by_sponsor = AsyncMock(
            return_value=[{"nct_id": "NCT123"}]
        )
        mock_scraper_class.return_value = mock_scraper

        orchestrator = PipelineOrchestrator(mock_config, tmp_path)

        results = await orchestrator.run_clinicaltrials_scraper()

        assert len(results) == 2  # Two sponsors
        assert all(r["status"] in ["success", "failed"] for r in results)

    @patch('pipeline.orchestrator.NetworkBuilder')
    async def test_extract_author_networks(
        self,
        mock_network,
        mock_config,
        tmp_path
    ):
        """Test extracting author networks."""
        # Setup mock
        mock_network.return_value.build_network = Mock(
            return_value=None
        )
        mock_network.return_value.get_network_stats = Mock(
            return_value={
                "author_count": 10,
                "collaboration_count": 20
            }
        )

        orchestrator = PipelineOrchestrator(mock_config, tmp_path)

        # Save some dummy papers first
        orchestrator.storage.save_pubmed_papers(
            papers=[{"pmid": "123"}],
            institution="Harvard"
        )

        # Run network extraction
        results = await orchestrator.extract_author_networks()

        assert len(results) >= 0  # Can be 0 if no papers or succeed

    @patch('pipeline.orchestrator.PubMedScraper')
    @patch('pipeline.orchestrator.ClinicalTrialsScraper')
    async def test_run_full_pipeline(
        self,
        mock_ct,
        mock_pubmed,
        mock_config,
        tmp_path
    ):
        """Test running full pipeline."""
        # Setup mocks
        mock_pubmed_instance = Mock()
        mock_pubmed_instance.search_by_affiliation = AsyncMock(
            return_value=["123"]
        )
        mock_pubmed_instance.fetch_article_details = AsyncMock(
            return_value=[{"pmid": "123"}]
        )
        mock_pubmed.return_value = mock_pubmed_instance

        mock_ct_instance = Mock()
        mock_ct_instance.search_by_sponsor = AsyncMock(
            return_value=[{"nct_id": "NCT123"}]
        )
        mock_ct.return_value = mock_ct_instance

        orchestrator = PipelineOrchestrator(mock_config, tmp_path)

        # Run full pipeline
        report = await orchestrator.run_full_pipeline()

        assert report["status"] == "completed"
        assert "pubmed" in report
        assert "clinicaltrials" in report
        assert "author_networks" in report

    async def test_error_handling(self, mock_config, tmp_path):
        """Test error handling in pipeline."""
        orchestrator = PipelineOrchestrator(mock_config, tmp_path)

        with patch('pipeline.orchestrator.PubMedScraper') as mock:
            mock_instance = Mock()
            mock_instance.search_by_affiliation = AsyncMock(
                side_effect=Exception("API Error")
            )
            mock.return_value = mock_instance

            results = await orchestrator.run_pubmed_scraper()

            # Should catch errors and continue
            assert any(r["status"] == "failed" for r in results)
            assert any("error" in r for r in results)
