#!/usr/bin/env python3
"""
Interactive Log Visualization Tool for Polymarket Training

Features:
- View training progress and loss curves
- Browse evaluation samples with prompts and answers
- Filter by correct/incorrect predictions
- Export analysis reports
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    data = []
    if file_path.exists():
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    return data


def print_header(text, char="=", width=100):
    """Print formatted header."""
    print(f"\n{char*width}")
    print(f"{text.center(width)}")
    print(f"{char*width}\n")


def print_training_summary(training_logs: List[Dict]):
    """Print training summary."""
    print_header("TRAINING SUMMARY")

    # Get epoch summaries
    epoch_summaries = [log for log in training_logs if log['type'] == 'epoch_summary']

    if not epoch_summaries:
        print("No training data found.")
        return

    print(f"Total epochs: {len(epoch_summaries)}")
    print(f"\n{'Epoch':<10} {'Train Loss':<15} {'Val Loss':<15}")
    print("─" * 40)

    for summary in epoch_summaries:
        epoch = summary['epoch']
        train_loss = summary['train_loss']
        val_loss = summary.get('val_loss', 'N/A')
        val_str = f"{val_loss:.4f}" if val_loss != 'N/A' else 'N/A'
        print(f"{epoch:<10} {train_loss:<15.4f} {val_str:<15}")

    # Plot ASCII chart if we have data
    if len(epoch_summaries) > 1:
        print("\n📈 Training Loss Progression:")
        plot_ascii_chart([s['train_loss'] for s in epoch_summaries])


def plot_ascii_chart(values: List[float], height=10, width=60):
    """Plot ASCII chart."""
    if not values:
        return

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val > min_val else 1

    # Normalize values to chart height
    normalized = [(v - min_val) / range_val * (height - 1) for v in values]

    # Create chart
    for row in range(height - 1, -1, -1):
        # Y-axis label
        y_val = min_val + (row / (height - 1)) * range_val
        print(f"{y_val:6.4f} │ ", end="")

        # Plot points
        for norm_val in normalized:
            if abs(norm_val - row) < 0.5:
                print("●", end="")
            elif row == 0:
                print("─", end="")
            else:
                print(" ", end="")
        print()

    # X-axis
    print(" " * 8 + "└" + "─" * len(values))
    print(" " * 9 + "".join([str(i % 10) for i in range(len(values))]))


def print_evaluation_summary(eval_logs: List[Dict]):
    """Print evaluation summary."""
    print_header("EVALUATION SUMMARY")

    if not eval_logs:
        print("No evaluation data found.")
        return

    # Calculate statistics
    total = len(eval_logs)
    correct = sum(1 for log in eval_logs if log['is_correct'])
    accuracy = correct / total if total > 0 else 0

    # Group by question type
    by_question_type = defaultdict(lambda: {'correct': 0, 'total': 0})
    for log in eval_logs:
        q_type = log['question_type']
        by_question_type[q_type]['total'] += 1
        if log['is_correct']:
            by_question_type[q_type]['correct'] += 1

    print(f"Total samples: {total}")
    print(f"Correct: {correct} ({accuracy:.1%})")
    print(f"Incorrect: {total - correct} ({(1-accuracy):.1%})")

    print(f"\n{'Question Type':<25} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("─" * 55)
    for q_type, stats in sorted(by_question_type.items()):
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{q_type:<25} {stats['correct']:<10} {stats['total']:<10} {acc:<10.1%}")


def browse_samples(eval_logs: List[Dict], filter_type: str = 'all'):
    """Browse evaluation samples interactively."""
    if filter_type == 'correct':
        samples = [log for log in eval_logs if log['is_correct']]
        title = "CORRECT PREDICTIONS"
    elif filter_type == 'incorrect':
        samples = [log for log in eval_logs if not log['is_correct']]
        title = "INCORRECT PREDICTIONS"
    else:
        samples = eval_logs
        title = "ALL PREDICTIONS"

    if not samples:
        print(f"\nNo {filter_type} samples found.")
        return

    print_header(f"{title} ({len(samples)} samples)")

    current_idx = 0

    while True:
        sample = samples[current_idx]

        print(f"\n{'='*100}")
        print(f"Sample {current_idx + 1}/{len(samples)}")
        print(f"{'='*100}")

        print(f"\n📊 Market Information:")
        print(f"   Market ID: {sample['market_id']}")
        print(f"   Question Type: {sample['question_type']}")

        print(f"\n📝 PROMPT STRUCTURE:")
        print(f"{'─'*100}")

        # Pre-prompt
        print(f"\n[PRE-PROMPT]")
        print(f"{sample['pre_prompt'][:200]}..." if len(sample['pre_prompt']) > 200 else sample['pre_prompt'])

        # Time series info
        for i, ts_info in enumerate(sample['time_series_info']):
            print(f"\n[TIME SERIES {i+1}]")
            print(f"   Description: {ts_info['description'][:100]}...")
            print(f"   Length: {ts_info['length']} points")
            print(f"   Patches: {ts_info['num_patches']} embedding tokens")

        # Post-prompt
        print(f"\n[POST-PROMPT / QUESTION]")
        print(f"{sample['post_prompt']}")

        print(f"\n{'─'*100}")

        # Answers
        print(f"\n🎯 EXPECTED ANSWER:")
        print(f"   {sample['expected_answer']}")

        print(f"\n🤖 GENERATED ANSWER:")
        print(f"   {sample['generated_answer'][:200]}..." if len(sample['generated_answer']) > 200 else sample['generated_answer'])

        print(f"\n{'✅ CORRECT' if sample['is_correct'] else '❌ INCORRECT'}")

        # Navigation
        print(f"\n{'─'*100}")
        print("Navigation: [n]ext | [p]revious | [q]uit | [j]ump to # | [s]earch")
        choice = input("Your choice: ").strip().lower()

        if choice == 'n':
            current_idx = (current_idx + 1) % len(samples)
        elif choice == 'p':
            current_idx = (current_idx - 1) % len(samples)
        elif choice == 'q':
            break
        elif choice == 'j':
            try:
                jump_to = int(input(f"Jump to sample (1-{len(samples)}): ")) - 1
                if 0 <= jump_to < len(samples):
                    current_idx = jump_to
                else:
                    print("Invalid sample number.")
            except ValueError:
                print("Invalid input.")
        elif choice == 's':
            search_term = input("Search in answers: ").strip().lower()
            found = False
            for i in range(len(samples)):
                check_idx = (current_idx + i + 1) % len(samples)
                sample_to_check = samples[check_idx]
                if (search_term in sample_to_check['expected_answer'].lower() or
                    search_term in sample_to_check['generated_answer'].lower()):
                    current_idx = check_idx
                    found = True
                    break
            if not found:
                print(f"No samples found containing '{search_term}'")


def export_report(session_dir: Path):
    """Export markdown report."""
    config_file = session_dir / "config.json"
    metrics_file = session_dir / "metrics.json"
    training_log = session_dir / "training_log.jsonl"
    eval_log = session_dir / "evaluation_log.jsonl"

    if not config_file.exists():
        print("Config file not found.")
        return

    # Load data
    with open(config_file) as f:
        config = json.load(f)

    metrics = {}
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)

    training_logs = load_jsonl(training_log)
    eval_logs = load_jsonl(eval_log)

    # Generate report
    report_file = session_dir / "report.md"

    with open(report_file, 'w') as f:
        f.write(f"# Training Report - {config['session_id']}\n\n")

        f.write("## Configuration\n\n")
        for key, value in config.items():
            f.write(f"- **{key}**: {value}\n")

        f.write("\n## Final Metrics\n\n")
        for key, value in metrics.items():
            f.write(f"- **{key}**: {value}\n")

        f.write("\n## Training Progress\n\n")
        epoch_summaries = [log for log in training_logs if log['type'] == 'epoch_summary']
        f.write("| Epoch | Train Loss |\n")
        f.write("|-------|------------|\n")
        for summary in epoch_summaries:
            f.write(f"| {summary['epoch']} | {summary['train_loss']:.4f} |\n")

        f.write("\n## Evaluation Results\n\n")
        if eval_logs:
            by_question_type = defaultdict(lambda: {'correct': 0, 'total': 0})
            for log in eval_logs:
                q_type = log['question_type']
                by_question_type[q_type]['total'] += 1
                if log['is_correct']:
                    by_question_type[q_type]['correct'] += 1

            f.write("| Question Type | Correct | Total | Accuracy |\n")
            f.write("|--------------|---------|-------|----------|\n")
            for q_type, stats in sorted(by_question_type.items()):
                acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                f.write(f"| {q_type} | {stats['correct']} | {stats['total']} | {acc:.1%} |\n")

        f.write("\n## Sample Predictions\n\n")
        for i, log in enumerate(eval_logs[:5]):
            f.write(f"### Sample {i+1}\n\n")
            f.write(f"- **Market**: {log['market_id']}\n")
            f.write(f"- **Question Type**: {log['question_type']}\n")
            f.write(f"- **Expected**: {log['expected_answer']}\n")
            f.write(f"- **Generated**: {log['generated_answer'][:100]}...\n")
            f.write(f"- **Status**: {'✅ Correct' if log['is_correct'] else '❌ Incorrect'}\n\n")

    print(f"✅ Report exported to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Polymarket training logs")
    parser.add_argument("session_dir", nargs='?', help="Session directory to visualize")
    parser.add_argument("--latest", action="store_true", help="Use latest session")
    parser.add_argument("--export", action="store_true", help="Export markdown report and exit")
    args = parser.parse_args()

    log_base = Path("/local/home/wangni/results/polymarket_enhanced_logs")

    if args.latest or not args.session_dir:
        # Find latest session
        sessions = sorted([d for d in log_base.iterdir() if d.is_dir()])
        if not sessions:
            print("No training sessions found.")
            return
        session_dir = sessions[-1]
        print(f"Using latest session: {session_dir.name}")
    else:
        session_dir = Path(args.session_dir)
        if not session_dir.is_absolute():
            session_dir = log_base / session_dir

    if not session_dir.exists():
        print(f"Session directory not found: {session_dir}")
        return

    # Load logs
    training_log = session_dir / "training_log.jsonl"
    eval_log = session_dir / "evaluation_log.jsonl"

    training_logs = load_jsonl(training_log)
    eval_logs = load_jsonl(eval_log)

    if args.export:
        export_report(session_dir)
        return

    # Interactive menu
    while True:
        print_header("POLYMARKET TRAINING LOG VIEWER", "=")
        print(f"Session: {session_dir.name}")
        print(f"Location: {session_dir}")
        print()
        print("1. View training summary")
        print("2. View evaluation summary")
        print("3. Browse all samples")
        print("4. Browse correct predictions")
        print("5. Browse incorrect predictions")
        print("6. Export report")
        print("7. Quit")
        print()

        choice = input("Select option (1-7): ").strip()

        if choice == '1':
            print_training_summary(training_logs)
            input("\nPress Enter to continue...")
        elif choice == '2':
            print_evaluation_summary(eval_logs)
            input("\nPress Enter to continue...")
        elif choice == '3':
            browse_samples(eval_logs, 'all')
        elif choice == '4':
            browse_samples(eval_logs, 'correct')
        elif choice == '5':
            browse_samples(eval_logs, 'incorrect')
        elif choice == '6':
            export_report(session_dir)
            input("\nPress Enter to continue...")
        elif choice == '7':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
