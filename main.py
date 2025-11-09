#!/usr/bin/env python3
"""
PubMed Clinical Trial Classifier - Main CLI

Command-line interface for classifying PubMed articles as clinical trials.
"""

import sys
from pathlib import Path
import click
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src import Config, ArticleProcessor, get_config, set_config

console = Console()


def setup_logging(log_level: str, log_file: Path) -> None:
    """Configure logging."""
    logger.remove()  # Remove default handler

    # Console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level,
        colorize=True
    )

    # File handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="10 MB"
    )


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """PubMed Clinical Trial Classifier

    Classify PubMed articles as clinical trials using local Ollama models.
    """
    pass


@cli.command()
@click.option(
    '--input-csv',
    type=click.Path(exists=True, path_type=Path),
    help='Path to input CSV file (overrides config)'
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    help='Output directory (overrides config)'
)
@click.option(
    '--model',
    type=str,
    help='Ollama model to use (e.g., llama3.1:8b)'
)
@click.option(
    '--batch-size',
    type=int,
    help='Number of articles to process before saving checkpoint'
)
@click.option(
    '--resume/--no-resume',
    default=True,
    help='Resume from checkpoint if available'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Enable verbose output'
)
def process(
    input_csv: Path,
    output_dir: Path,
    model: str,
    batch_size: int,
    resume: bool,
    verbose: bool
):
    """Process all articles in the CSV file."""

    # Load config
    config = get_config()

    # Override with CLI options
    if input_csv:
        config.input_csv = input_csv
    if output_dir:
        config.output_dir = output_dir
    if model:
        config.ollama_model = model
    if batch_size:
        config.batch_size = batch_size
    config.resume_from_checkpoint = resume
    config.verbose = verbose

    if verbose:
        config.log_level = "DEBUG"

    # Ensure directories exist
    config.ensure_directories()

    # Setup logging
    log_file = config.log_dir / f"classifier_{config.output_dir.name}.log"
    setup_logging(config.log_level, log_file)

    # Display configuration
    console.print(Panel.fit(
        f"[bold]PubMed Clinical Trial Classifier[/bold]\n\n"
        f"Input CSV: {config.input_csv}\n"
        f"Output Dir: {config.output_dir}\n"
        f"Model: {config.ollama_model}\n"
        f"Batch Size: {config.batch_size}\n"
        f"Resume: {resume}",
        title="Configuration",
        border_style="blue"
    ))

    # Create processor
    processor = ArticleProcessor(config)

    # Initialize
    console.print("\n[yellow]Initializing...[/yellow]")
    if not processor.initialize():
        console.print("[red]✗ Initialization failed[/red]")
        sys.exit(1)

    console.print("[green]✓ Initialization successful[/green]\n")

    # Process
    try:
        console.print("[yellow]Processing articles...[/yellow]\n")
        stats = processor.process_all()

        # Display results
        _display_results(stats)

        console.print("\n[green]✓ Processing complete![/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        console.print("Progress has been saved. Use --resume to continue.")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]✗ Error: {e}[/red]")
        logger.exception("Processing failed")
        sys.exit(1)


@cli.command()
@click.option(
    '--input-csv',
    type=click.Path(exists=True, path_type=Path),
    help='Path to input CSV file'
)
@click.option(
    '--model',
    type=str,
    help='Ollama model to use'
)
def test(input_csv: Path, model: str):
    """Test the classifier on a few articles."""

    config = get_config()

    if input_csv:
        config.input_csv = input_csv
    if model:
        config.ollama_model = model

    config.ensure_directories()

    # Setup logging
    setup_logging("INFO", config.log_dir / "test.log")

    console.print(Panel.fit(
        "[bold]Test Mode[/bold]\n\n"
        f"Will process first 5 articles from: {config.input_csv}",
        border_style="yellow"
    ))

    # Create processor
    processor = ArticleProcessor(config)

    # Initialize
    if not processor.initialize():
        console.print("[red]✗ Initialization failed[/red]")
        sys.exit(1)

    # Process first 5 articles
    try:
        results = processor.process_batch(start_index=0, batch_size=5)

        console.print(f"\n[green]Processed {len(results)} articles[/green]\n")

        # Display results in table
        table = Table(title="Test Results")
        table.add_column("PMID", style="cyan")
        table.add_column("Is Trial?", style="magenta")
        table.add_column("Confidence", style="green")
        table.add_column("Method", style="yellow")

        for result in results:
            if result.success and result.classification:
                c = result.classification
                table.add_row(
                    result.article.pmid,
                    "✓ Yes" if c.is_clinical_trial else "✗ No",
                    f"{c.confidence:.2f}",
                    result.parsing_method or "N/A"
                )
            else:
                table.add_row(
                    result.article.pmid,
                    "ERROR",
                    "-",
                    result.error or "Unknown"
                )

        console.print(table)

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        logger.exception("Test failed")
        sys.exit(1)


@cli.command()
@click.option(
    '--model',
    type=str,
    help='Ollama model to use'
)
def check(model: str):
    """Check Ollama connection and model availability."""

    config = get_config()

    if model:
        config.ollama_model = model

    setup_logging("INFO", config.log_dir / "check.log")

    console.print(Panel.fit(
        "[bold]System Check[/bold]\n\n"
        f"Ollama URL: {config.ollama_base_url}\n"
        f"Model: {config.ollama_model}",
        border_style="cyan"
    ))

    from src import OllamaClient

    client = OllamaClient(config)

    # Check connection
    console.print("\n[yellow]Checking Ollama connection...[/yellow]")
    if client.check_connection():
        console.print("[green]✓ Ollama is running[/green]")
    else:
        console.print("[red]✗ Cannot connect to Ollama[/red]")
        console.print(f"Make sure Ollama is running at {config.ollama_base_url}")
        sys.exit(1)

    # Check model
    console.print(f"\n[yellow]Checking model '{config.ollama_model}'...[/yellow]")
    if client.check_model_availability():
        console.print(f"[green]✓ Model '{config.ollama_model}' is available[/green]")
    else:
        console.print(f"[red]✗ Model '{config.ollama_model}' not found[/red]")
        console.print(f"\nTo pull the model, run:")
        console.print(f"  ollama pull {config.ollama_model}")
        sys.exit(1)

    # Test generation
    console.print("\n[yellow]Testing generation...[/yellow]")
    if client.test_generation():
        console.print("[green]✓ Generation test successful[/green]")
    else:
        console.print("[red]✗ Generation test failed[/red]")
        sys.exit(1)

    console.print("\n[green]✓ All checks passed![/green]")


def _display_results(stats) -> None:
    """Display final statistics."""

    table = Table(title="Processing Statistics", show_header=True)
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", style="magenta", justify="right")

    table.add_row("Total Articles", str(stats.total_articles))
    table.add_row("Processed", str(stats.processed))
    table.add_row("Successful", str(stats.successful))
    table.add_row("Failed", str(stats.failed))
    table.add_section()
    table.add_row("Clinical Trials", str(stats.clinical_trials))
    table.add_row("Non-Trials", str(stats.non_trials))
    table.add_section()
    table.add_row("Total Time", f"{stats.total_time_seconds:.1f}s")
    table.add_row("Avg Time/Article", f"{stats.average_time_per_article:.2f}s")

    console.print(table)

    # Parsing methods
    if stats.parsing_methods:
        console.print("\n[bold]Parsing Methods Used:[/bold]")
        for method, count in stats.parsing_methods.items():
            console.print(f"  {method}: {count}")


if __name__ == '__main__':
    cli()
