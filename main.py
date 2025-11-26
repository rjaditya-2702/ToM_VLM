# main.py
import argparse
from .inference.runner import run_benchmark


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
        nargs='+',
        type=str,
        choices=['expt1', 'expt2', 'expt3', 'expt4', 'expt5'],
        help='One or more experiment types (e.g., --experiment expt1 expt3)'
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

    parser.add_argument(
        '--size',
        help='samples to infer',
        default = None,
        type = int
    )

    parser.add_argument(
        '--prompt',
        help='samples to infer',
        default = 'v1',
        type = str
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
        for exp in args.experiment:
            print("-"*40)
            print(f"\t\t\tRUNNIGN EXPERIMENT - {exp}")
            print("-"*40)
            run_benchmark(
                dataset = args.dataset,
                model_name=args.model,
                experiment_type=exp,
                batch_size=args.batch_size,
                n = args.size,
                v = args.prompt
            )
            print()
    except Exception as e:
        print(f"\nError running benchmark: {e}")
        return e
    
    return 0


if __name__ == "__main__":
    exit(main())