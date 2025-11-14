#!/usr/bin/env python3
"""
Multi-Agent System Demo

Demonstrates the investment evaluation workflow with the multi-agent system.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents import CoordinatorAgent, MemoryManager
from src import Config, OllamaClient
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


# Sample clinical trial
SAMPLE_TRIAL = {
    "nct_id": "NCT04567890",
    "title": "Phase 2 Study of Novel CD19/CD22 Dual-Targeting CAR-T in Relapsed/Refractory B-Cell Lymphoma",
    "phase": "Phase II",
    "indication": "Relapsed/refractory B-cell lymphoma",
    "intervention": "CD19/CD22 dual-targeting CAR-T cell therapy",
    "sample_size": 60,
    "primary_outcome": "Overall response rate (ORR) at 3 months",
    "secondary_outcomes": "Duration of response, progression-free survival, safety and tolerability, CAR-T expansion kinetics",
    "sponsor": "Boston Children's Hospital",
    "status": "Recruiting",
    "description": """
    This is a Phase 2, single-arm, open-label study evaluating a novel dual-targeting
    CAR-T therapy in patients with relapsed/refractory B-cell lymphoma who have failed
    at least 2 prior lines of therapy. The CAR-T product targets both CD19 and CD22
    antigens to prevent antigen escape, a major limitation of current single-target
    CAR-T therapies.
    """
}


async def main():
    """Run investment evaluation demo."""

    console.print(Panel.fit(
        "[bold cyan]Multi-Agent Investment Evaluation Demo[/bold cyan]\n\n"
        "This demo evaluates a clinical trial using the multi-agent system.",
        border_style="cyan"
    ))

    # 1. Setup
    console.print("\n[yellow]Step 1: Initializing system...[/yellow]")

    config = Config()
    ollama_client = OllamaClient(config)

    # Check Ollama connection
    if not ollama_client.check_connection():
        console.print("[red]✗ Ollama is not running. Please start Ollama first.[/red]")
        console.print("\nRun: ollama serve")
        return

    if not ollama_client.check_model_availability():
        console.print(f"[red]✗ Model '{config.ollama_model}' not available[/red]")
        console.print(f"\nRun: ollama pull {config.ollama_model}")
        return

    console.print(f"[green]✓ Ollama connected (model: {config.ollama_model})[/green]")

    # Initialize memory
    memory = MemoryManager(chroma_path="./data/chroma", persist=False)
    console.print("[green]✓ Memory manager initialized[/green]")

    # Initialize coordinator
    coordinator = CoordinatorAgent(
        name="coordinator",
        llm_client=ollama_client,
        memory_manager=memory,
        prompt_dir=Path("prompts/agents")
    )
    console.print("[green]✓ Coordinator agent initialized[/green]")

    # 2. Display trial info
    console.print("\n[yellow]Step 2: Trial Information[/yellow]")
    console.print(Panel(
        f"[bold]{SAMPLE_TRIAL['title']}[/bold]\n\n"
        f"NCT ID: {SAMPLE_TRIAL['nct_id']}\n"
        f"Phase: {SAMPLE_TRIAL['phase']}\n"
        f"Indication: {SAMPLE_TRIAL['indication']}\n"
        f"Intervention: {SAMPLE_TRIAL['intervention']}\n"
        f"Sample Size: {SAMPLE_TRIAL['sample_size']}\n"
        f"Primary Outcome: {SAMPLE_TRIAL['primary_outcome']}",
        title="Clinical Trial",
        border_style="blue"
    ))

    # 3. Run evaluation
    console.print("\n[yellow]Step 3: Running investment evaluation...[/yellow]")
    console.print("This may take 1-2 minutes...\n")

    try:
        result = await coordinator.process({
            "type": "investment_evaluation",
            "input": SAMPLE_TRIAL
        })

        # 4. Display results
        console.print("\n[green]✓ Evaluation complete![/green]\n")

        # Show workflow status
        console.print(f"[bold]Workflow Status:[/bold] {result['status']}")
        console.print(f"[bold]Steps Completed:[/bold] {len(result['steps'])}")

        # Show step results
        console.print("\n[bold cyan]Step Results:[/bold cyan]")
        for i, step in enumerate(result["steps"], 1):
            status_icon = "✓" if step["success"] else "✗"
            status_color = "green" if step["success"] else "red"
            console.print(
                f"  [{status_color}]{status_icon}[/{status_color}] "
                f"Step {step['step']}: {step['agent']} - {step['task']}"
            )

        # Show final recommendation if available
        if len(result["steps"]) >= 3 and result["steps"][2]["success"]:
            final = result["steps"][2]["output"]

            console.print("\n" + "="*60)
            console.print("[bold cyan]INVESTMENT RECOMMENDATION[/bold cyan]")
            console.print("="*60)

            console.print(f"\n[bold]Overall Score:[/bold] {final.get('overall_score', 'N/A')}/100")
            console.print(f"[bold]Recommendation:[/bold] {final.get('recommendation', 'N/A')}")
            console.print(f"[bold]Confidence:[/bold] {final.get('confidence', 'N/A')}")

            # Show score breakdown
            if "scores" in final:
                scores = final["scores"]
                console.print(f"\n[bold]Score Breakdown:[/bold]")
                console.print(f"  • Market Analysis: {scores.get('market_analysis', 0)}/30")
                console.print(f"  • Competitive Position: {scores.get('competitive_position', 0)}/25")
                console.print(f"  • Scientific Assessment: {scores.get('scientific_assessment', 0)}/20")
                console.print(f"  • Regulatory/IP: {scores.get('regulatory_ip', 0)}/15")
                console.print(f"  • Commercial Viability: {scores.get('commercial_viability', 0)}/10")

            # Show market metrics
            console.print(f"\n[bold]Market Metrics:[/bold]")
            console.print(f"  • Market Size: ${final.get('market_size_usd_billions', 'N/A')}B")
            console.print(f"  • Patient Population: {final.get('patient_population', 'N/A'):,}")

            # Show key risks
            if "key_risks" in final and len(final["key_risks"]) > 0:
                console.print(f"\n[bold]Key Risks:[/bold]")
                for risk in final["key_risks"][:3]:
                    if isinstance(risk, dict):
                        console.print(f"  • {risk.get('risk', '')} (Impact: {risk.get('impact', 'N/A')})")
                    else:
                        console.print(f"  • {risk}")

            # Show key opportunities
            if "key_opportunities" in final and len(final["key_opportunities"]) > 0:
                console.print(f"\n[bold]Key Opportunities:[/bold]")
                for opp in final["key_opportunities"][:3]:
                    if isinstance(opp, dict):
                        console.print(f"  • {opp.get('opportunity', '')} (Value: {opp.get('value_potential', 'N/A')})")
                    else:
                        console.print(f"  • {opp}")

            # Show next steps
            if "next_steps" in final and len(final["next_steps"]) > 0:
                console.print(f"\n[bold]Next Steps:[/bold]")
                for step in final["next_steps"][:3]:
                    console.print(f"  {step}")

        # Show memo if generated
        if "memo" in result and result.get("status") == "success":
            console.print("\n" + "="*60)
            console.print("[bold cyan]INVESTMENT MEMO[/bold cyan]")
            console.print("="*60 + "\n")

            # Render as markdown
            md = Markdown(result["memo"])
            console.print(md)

            # Save to file
            memo_file = Path("output/investment_memo.md")
            memo_file.parent.mkdir(parents=True, exist_ok=True)
            with open(memo_file, "w") as f:
                f.write(result["memo"])

            console.print(f"\n[green]✓ Memo saved to: {memo_file}[/green]")

    except Exception as e:
        console.print(f"\n[red]✗ Error during evaluation: {e}[/red]")
        import traceback
        traceback.print_exc()
        return

    console.print("\n[green]Demo complete![/green]")


if __name__ == "__main__":
    asyncio.run(main())
