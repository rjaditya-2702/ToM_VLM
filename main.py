# main.py
import argparse
from runner import run_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="VQA Model Benchmarking Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Name of the model to benchmark'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required = True,
        help='Name of the dataset to use (e.g., msed, mmtom)'
    )

    parser.add_argument(
        '--experiment',
        type=str,
        choices=['expt1', 'expt2', 'expt3', 'expt4'],
        help='Experiment type: expt1 (blank+no_q), expt2 (blank+q), expt3 (img+no_q), expt4 (img+q)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference (default: 32)'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models and exit'
    )
    
    args = parser.parse_args()
    
    # Handle listing commands
    if args.list_models:
        models = runner.list_available_models()
        print("\nAvailable Models:")
        for model in models:
            print(f"  - {model}")
        return
    
    # Validate required arguments
    if not args.model or not args.experiment:
        parser.error("--model and --experiment are required (unless using --list-models or --list-experiments)")
    
    # Run benchmark
    try:
        run_benchmark(
            datset = args.dataset,
            model_name=args.model,
            experiment_type=args.experiment,
            batch_size=args.batch_size,
            device=args.device
        )
    except Exception as e:
        print(f"\nError running benchmark: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())