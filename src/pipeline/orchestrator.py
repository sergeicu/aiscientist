"""Pipeline orchestrator for data collection."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.console import Console

from .config import PipelineConfig
from .storage import StorageManager

# Import scrapers from completed tasks
from src.scrapers.pubmed_scraper import PubMedScraper
from src.scrapers.clinicaltrials_scraper import ClinicalTrialsScraper
from src.network.network_builder import NetworkBuilder

logger = logging.getLogger(__name__)
console = Console()


class PipelineOrchestrator:
    """Orchestrates the full data collection pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        output_dir: str = "./data",
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None
    ):
        """Initialize orchestrator."""
        self.config = config
        self.storage = StorageManager(base_dir=output_dir)
        self.pubmed_scraper = None
        self.ct_scraper = None
        self.network_builder = None
        self.neo4j_uri = neo4j_uri or "bolt://localhost:7687"
        self.neo4j_user = neo4j_user or "neo4j"
        self.neo4j_password = neo4j_password or "password"

    def _init_scrapers(self):
        """Initialize all scrapers."""
        if not self.pubmed_scraper:
            self.pubmed_scraper = PubMedScraper(
                email=self.config.pubmed.email,
                api_key=self.config.pubmed.api_key
            )

        if not self.ct_scraper:
            self.ct_scraper = ClinicalTrialsScraper()

        logger.info("Initialized all scrapers")

    def _init_network_builder(self):
        """Initialize network builder (optional, requires Neo4j)."""
        if not self.network_builder:
            try:
                self.network_builder = NetworkBuilder(
                    uri=self.neo4j_uri,
                    user=self.neo4j_user,
                    password=self.neo4j_password
                )
                logger.info("Initialized network builder")
            except Exception as e:
                logger.warning(f"Could not initialize network builder: {e}")
                self.network_builder = None

    async def run_pubmed_scraper(self) -> List[Dict[str, Any]]:
        """Run PubMed scraper for all institutions."""
        if not self.pubmed_scraper:
            self._init_scrapers()

        results = []

        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                "[cyan]Scraping PubMed...",
                total=len(self.config.pubmed.institutions)
            )

            for institution in self.config.pubmed.institutions:
                try:
                    logger.info(f"Fetching papers for {institution}")

                    # Search for articles
                    query = f"{institution}[Affiliation]"
                    pmids = await self.pubmed_scraper.search_by_affiliation(
                        query=query,
                        max_results=self.config.pubmed.max_results_per_institution
                    )

                    # Fetch article details
                    papers = await self.pubmed_scraper.fetch_article_details(pmids)

                    # Save to storage
                    filepath = self.storage.save_pubmed_papers(
                        papers=papers,
                        institution=institution
                    )

                    results.append({
                        "institution": institution,
                        "status": "success",
                        "papers_count": len(papers),
                        "filepath": str(filepath)
                    })

                    console.print(
                        f"✓ {institution}: {len(papers)} papers",
                        style="green"
                    )

                except Exception as e:
                    logger.error(f"Failed to scrape {institution}: {e}")
                    results.append({
                        "institution": institution,
                        "status": "failed",
                        "error": str(e)
                    })
                    console.print(f"✗ {institution}: {str(e)}", style="red")

                finally:
                    progress.update(task, advance=1)

        return results

    async def run_clinicaltrials_scraper(self) -> List[Dict[str, Any]]:
        """Run ClinicalTrials scraper for all sponsors."""
        if not self.ct_scraper:
            self._init_scrapers()

        results = []

        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                "[cyan]Scraping ClinicalTrials.gov...",
                total=len(self.config.clinicaltrials.sponsors)
            )

            for sponsor in self.config.clinicaltrials.sponsors:
                try:
                    logger.info(f"Fetching trials for {sponsor}")

                    trials = await self.ct_scraper.search_by_sponsor(
                        sponsor=sponsor,
                        max_results=self.config.clinicaltrials.max_results_per_sponsor
                    )

                    filepath = self.storage.save_clinicaltrials(
                        trials=trials,
                        sponsor=sponsor
                    )

                    results.append({
                        "sponsor": sponsor,
                        "status": "success",
                        "trials_count": len(trials),
                        "filepath": str(filepath)
                    })

                    console.print(
                        f"✓ {sponsor}: {len(trials)} trials",
                        style="green"
                    )

                except Exception as e:
                    logger.error(f"Failed to scrape {sponsor}: {e}")
                    results.append({
                        "sponsor": sponsor,
                        "status": "failed",
                        "error": str(e)
                    })
                    console.print(f"✗ {sponsor}: {str(e)}", style="red")

                finally:
                    progress.update(task, advance=1)

        return results

    async def extract_author_networks(self) -> List[Dict[str, Any]]:
        """Extract author networks from all papers."""
        results = []

        console.print("[cyan]Extracting author networks...[/cyan]")

        # Load all PubMed papers
        all_papers = self.storage.load_all_pubmed_papers()

        if not all_papers:
            console.print("[yellow]No papers found to extract networks[/yellow]")
            return results

        try:
            # Try to build network in Neo4j if available
            self._init_network_builder()

            if self.network_builder:
                self.network_builder.build_network(all_papers)
                stats = self.network_builder.get_network_stats()

                # Save network stats
                network_data = {
                    "authors": stats.get("author_count", 0),
                    "collaborations": stats.get("collaboration_count", 0),
                    "papers": len(all_papers),
                    "timestamp": datetime.now().isoformat()
                }

                filepath = self.storage.save_author_network(
                    network=network_data,
                    source="all_institutions"
                )

                results.append({
                    "source": "all_institutions",
                    "status": "success",
                    "authors_count": stats.get("author_count", 0),
                    "collaborations_count": stats.get("collaboration_count", 0),
                    "filepath": str(filepath)
                })

                console.print(
                    f"✓ Extracted {stats.get('author_count', 0)} authors, "
                    f"{stats.get('collaboration_count', 0)} collaborations",
                    style="green"
                )
            else:
                # Save papers data without network building
                network_data = {
                    "papers": len(all_papers),
                    "note": "Network builder not available (Neo4j not configured)",
                    "timestamp": datetime.now().isoformat()
                }

                filepath = self.storage.save_author_network(
                    network=network_data,
                    source="all_institutions"
                )

                results.append({
                    "source": "all_institutions",
                    "status": "skipped",
                    "note": "Neo4j not configured",
                    "filepath": str(filepath)
                })

                console.print(
                    "[yellow]Network builder skipped (Neo4j not configured)[/yellow]"
                )

        except Exception as e:
            logger.error(f"Failed to extract networks: {e}")
            results.append({
                "source": "all_institutions",
                "status": "failed",
                "error": str(e)
            })
            console.print(f"✗ Network extraction failed: {str(e)}", style="red")

        return results

    async def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline."""
        console.print("\n[bold cyan]Starting Data Collection Pipeline[/bold cyan]\n")

        start_time = datetime.now()

        # Run scrapers in parallel
        pubmed_task = asyncio.create_task(self.run_pubmed_scraper())
        ct_task = asyncio.create_task(self.run_clinicaltrials_scraper())

        pubmed_results, ct_results = await asyncio.gather(
            pubmed_task,
            ct_task,
            return_exceptions=True
        )

        # Handle exceptions
        if isinstance(pubmed_results, Exception):
            logger.error(f"PubMed scraper failed: {pubmed_results}")
            pubmed_results = []
        if isinstance(ct_results, Exception):
            logger.error(f"ClinicalTrials scraper failed: {ct_results}")
            ct_results = []

        # Extract author networks
        network_results = await self.extract_author_networks()

        # Create unified dataset
        console.print("\n[cyan]Creating unified dataset...[/cyan]")
        unified_path = self.storage.create_unified_dataset()
        console.print(f"✓ Unified dataset: {unified_path}", style="green")

        # Generate report
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Count totals
        total_papers = sum(
            r.get("papers_count", 0) for r in pubmed_results
            if isinstance(r, dict) and r.get("status") == "success"
        )
        total_trials = sum(
            r.get("trials_count", 0) for r in ct_results
            if isinstance(r, dict) and r.get("status") == "success"
        )

        report = {
            "status": "completed",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "pubmed": pubmed_results,
            "clinicaltrials": ct_results,
            "author_networks": network_results,
            "total_papers": total_papers,
            "total_trials": total_trials,
            "unified_dataset": str(unified_path)
        }

        # Save report
        report_path = self.storage.base_dir / "logs" / f"pipeline_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"

        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Print summary
        console.print("\n[bold green]Pipeline Completed![/bold green]")
        console.print(f"Duration: {duration:.1f}s")
        console.print(f"Total Papers: {total_papers}")
        console.print(f"Total Trials: {total_trials}")
        console.print(f"Report: {report_path}\n")

        return report


# CLI entry point
async def main():
    """Main entry point for CLI."""
    import argparse
    from pipeline.config import load_config

    parser = argparse.ArgumentParser(
        description="Data Collection Pipeline Orchestrator"
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to pipeline configuration YAML file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for data storage"
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j connection URI"
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default="neo4j",
        help="Neo4j username"
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        help="Neo4j password"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config = load_config(args.config)

    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        config=config,
        output_dir=args.output_dir,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password
    )

    # Run pipeline
    await orchestrator.run_full_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
