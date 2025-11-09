#!/usr/bin/env python3
"""
Basic usage examples for the PubMed Clinical Trial Classifier.

This script demonstrates how to use the classifier programmatically
without the CLI.
"""

from pathlib import Path
from src import (
    Config,
    ArticleProcessor,
    Article,
    set_config
)


def example_1_basic_processing():
    """Example 1: Basic processing with default configuration."""

    print("=" * 60)
    print("Example 1: Basic Processing")
    print("=" * 60)

    # Create config
    config = Config()
    config.input_csv = Path("data/pubmed_data_2000.csv")
    config.output_dir = Path("output/example1")
    config.batch_size = 5
    set_config(config)

    # Create processor
    processor = ArticleProcessor(config)

    # Initialize
    if not processor.initialize():
        print("Initialization failed!")
        return

    # Process first 10 articles
    results = processor.process_batch(start_index=0, batch_size=10)

    # Print results
    print(f"\nProcessed {len(results)} articles:\n")

    for result in results:
        if result.success and result.classification:
            c = result.classification
            print(f"PMID: {result.article.pmid}")
            print(f"  Is Clinical Trial: {c.is_clinical_trial}")
            print(f"  Confidence: {c.confidence:.2f}")
            print(f"  Reasoning: {c.reasoning[:100]}...")
            print()
        else:
            print(f"PMID: {result.article.pmid} - FAILED: {result.error}")
            print()


def example_2_custom_article():
    """Example 2: Classify a custom article."""

    print("=" * 60)
    print("Example 2: Custom Article")
    print("=" * 60)

    # Create config
    config = Config()
    config.ollama_model = "llama3.1:8b"
    set_config(config)

    from src import OllamaClient, PromptLoader, StructuredOutputHandler

    # Initialize components
    client = OllamaClient(config)
    prompt_loader = PromptLoader(Path("prompts/clinical_trial_classifier.yaml"))
    output_handler = StructuredOutputHandler(config)

    # Check connection
    if not client.check_connection():
        print("Ollama not available!")
        return

    # Create a custom article
    article = Article(
        pmid="CUSTOM001",
        title="Randomized Trial of Vitamin D in Pediatric Asthma",
        abstract="""
        Background: Vitamin D deficiency is common in children with asthma.
        We conducted a randomized, double-blind, placebo-controlled trial to
        evaluate vitamin D supplementation in pediatric asthma.

        Methods: We enrolled 408 children aged 6-16 years with mild to moderate
        asthma. Participants were randomized to receive vitamin D3 (2000 IU daily)
        or placebo for 12 months. The primary outcome was time to first asthma
        exacerbation.

        Results: Vitamin D supplementation reduced the risk of exacerbation by
        26% (hazard ratio 0.74, 95% CI 0.58-0.94, p=0.01).

        Conclusion: Vitamin D supplementation reduces asthma exacerbations in
        children.
        """
    )

    # Format prompt
    system_prompt = prompt_loader.get_system_prompt()
    user_prompt = prompt_loader.format_prompt(article, include_examples=True)

    print("Classifying article...")
    print(f"Title: {article.title}\n")

    # Generate
    response = client.generate_with_retry(
        prompt=user_prompt,
        system_prompt=system_prompt,
        format='json'
    )

    if not response:
        print("Generation failed!")
        return

    # Parse output
    raw_output = response.get('message', {}).get('content', '')
    classification, method = output_handler.parse_json_output(raw_output)

    if classification:
        print("Classification Results:")
        print(f"  Is Clinical Trial: {classification.is_clinical_trial}")
        print(f"  Confidence: {classification.confidence:.2f}")
        print(f"  Trial Phase: {classification.trial_phase}")
        print(f"  Study Type: {classification.study_type}")
        print(f"  Intervention: {classification.intervention_type}")
        print(f"  Sample Size: {classification.sample_size}")
        print(f"  Randomized: {classification.randomized}")
        print(f"  Blinded: {classification.blinded}")
        print(f"  Reasoning: {classification.reasoning}")
        print(f"\n  Parsing Method: {method}")
    else:
        print("Failed to parse output!")
        print(f"Raw output:\n{raw_output}")


def example_3_statistics():
    """Example 3: Access processing statistics."""

    print("=" * 60)
    print("Example 3: Processing Statistics")
    print("=" * 60)

    from src import ProcessingStats

    # Create stats object
    stats = ProcessingStats(
        total_articles=2000,
        processed=2000,
        successful=1950,
        failed=50
    )

    # Simulate some classifications
    stats.clinical_trials = 450
    stats.non_trials = 1500
    stats.total_time_seconds = 3600.0
    stats.average_time_per_article = 1.8

    stats.trial_phases = {
        "Phase I": 50,
        "Phase II": 120,
        "Phase III": 200,
        "Phase IV": 80
    }

    stats.parsing_methods = {
        "direct": 1800,
        "json_repair": 120,
        "regex": 30
    }

    print(f"Total Articles: {stats.total_articles}")
    print(f"Processed: {stats.processed}")
    print(f"Success Rate: {stats.successful/stats.processed*100:.1f}%")
    print(f"\nClinical Trials: {stats.clinical_trials} ({stats.clinical_trials/stats.processed*100:.1f}%)")
    print(f"Non-Trials: {stats.non_trials} ({stats.non_trials/stats.processed*100:.1f}%)")
    print(f"\nTotal Time: {stats.total_time_seconds/3600:.2f} hours")
    print(f"Avg Time/Article: {stats.average_time_per_article:.2f}s")

    print("\nTrial Phases:")
    for phase, count in stats.trial_phases.items():
        print(f"  {phase}: {count}")

    print("\nParsing Methods:")
    for method, count in stats.parsing_methods.items():
        print(f"  {method}: {count}")


if __name__ == '__main__':
    import sys

    examples = {
        '1': example_1_basic_processing,
        '2': example_2_custom_article,
        '3': example_3_statistics,
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("PubMed Clinical Trial Classifier - Usage Examples")
        print("\nAvailable examples:")
        print("  1 - Basic processing")
        print("  2 - Custom article classification")
        print("  3 - Processing statistics")
        print("\nRun: python examples/basic_usage.py <number>")
        print("Example: python examples/basic_usage.py 2")
