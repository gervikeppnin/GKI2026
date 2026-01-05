#!/usr/bin/env python3
"""
Run Inverse Optimization for Pipe Roughness Calibration.

Command-line script to execute the inverse optimization solver.
Saves models in per-pipe format for direct use in dashboard.

Usage:
    python -m training.run_optimizer --algorithm differential_evolution --max-evals 200
    python -m training.run_optimizer --algorithm cma-es --date 2024-01-15 --n-solutions 5
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.optimization import (
    InverseOptimizer,
    DEFAULT_MAX_EVALUATIONS,
)
from core.model_storage import save_model


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Inverse Optimization for Pipe Roughness Calibration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Algorithm selection
    parser.add_argument(
        '--algorithm', '-a',
        type=str,
        default='differential_evolution',
        choices=['differential_evolution', 'cma-es', 'l-bfgs-b'],
        help='Optimization algorithm'
    )
    
    # Optimization parameters
    parser.add_argument(
        '--max-evals', '-m',
        type=int,
        default=DEFAULT_MAX_EVALUATIONS,
        help='Maximum objective function evaluations'
    )
    
    parser.add_argument(
        '--n-solutions', '-n',
        type=int,
        default=5,
        help='Number of best solutions to return'
    )
    
    parser.add_argument(
        '--tolerance', '-t',
        type=float,
        default=1e-4,
        help='Convergence tolerance'
    )
    
    # Data configuration
    parser.add_argument(
        '--date', '-d',
        type=str,
        default=None,
        help='Specific date for sensor data (YYMMDD format)'
    )
    
    parser.add_argument(
        '--all-dates',
        action='store_true',
        help='Optimize across ALL available dates for robust calibration'
    )
    
    parser.add_argument(
        '--grouping', '-g',
        type=str,
        default='smart',
        choices=['decade', 'year', 'detailed', 'diameter', 'decade_diameter', '5year', '5year_diameter', '5year_diameter_length', 'smart', 'compact', 'sophisticated'],
        help='Pipe grouping strategy (smart=6 groups, compact=4 groups, sophisticated=~80 groups)'
    )
    
    # Model naming
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Custom name for the saved model'
    )
    
    # Multi-start
    parser.add_argument(
        '--multistart',
        type=int,
        default=0,
        help='Number of random restarts (0 = single run)'
    )
    
    # Misc
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Get available dates if using all-dates mode
    dates_to_use = None
    if args.all_dates:
        from core.data_utils import get_available_dates
        dates_to_use = get_available_dates()
        if not dates_to_use:
            print("ERROR: --all-dates specified but no dates found in data directory")
            return 1
    
    # Print header
    print("=" * 70)
    print("   INVERSE OPTIMIZATION FOR PIPE ROUGHNESS CALIBRATION")
    print("=" * 70)
    print(f"  Algorithm:    {args.algorithm}")
    print(f"  Max Evals:    {args.max_evals}")
    print(f"  N Solutions:  {args.n_solutions}")
    print(f"  Grouping:     {args.grouping}")
    if args.all_dates:
        print(f"  Mode:         MULTI-DATE ({len(dates_to_use)} dates)")
    else:
        print(f"  Date:         {args.date or 'Default'}")
    if args.multistart > 0:
        print(f"  Multi-start:  {args.multistart} restarts")
    print("=" * 70)
    
    try:
        # Single optimization run (possibly multi-date)
        optimizer = InverseOptimizer(
            date=args.date if not args.all_dates else None,
            dates=dates_to_use,
            grouping_strategy=args.grouping,
        )
        
        result = optimizer.optimize(
            algorithm=args.algorithm,
            max_evaluations=args.max_evals,
            seed=args.seed,
        )
        
        # Print results
        print("\n" + "=" * 70)
        print("   OPTIMIZATION RESULTS")
        print("=" * 70)
        print(f"  Converged:     {result.converged}")
        print(f"  Evaluations:   {result.n_evaluations}")
        print(f"  Elapsed Time:  {result.elapsed_time:.1f}s")
        print("-" * 70)
        
        for i, (sol, obj) in enumerate(zip(result.best_solutions, result.objective_values)):
            print(f"\n  Solution {i+1}:  J(θ) = {obj:.6f}")
            for name, value in sol.items():
                print(f"    {name:25s} = {value:.2f}")
        
        print("\n" + "=" * 70)
        
        # Save best solution using new per-pipe format
        if result.converged and result.best_solutions:
            best_group_roughness = result.best_solutions[0]
            best_mae = result.objective_values[0]
            
            # Generate model name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = args.name or f"opt_{args.algorithm}_{timestamp}"
            
            # Save with per-pipe expansion
            saved_model = save_model(
                name=model_name,
                group_roughness=best_group_roughness,
                pipe_groups=optimizer.pipe_groups,
                mae=best_mae,
                algorithm=args.algorithm,
                grouping_strategy=args.grouping,
                training_date=args.date,
                n_evaluations=result.n_evaluations,
                elapsed_time=result.elapsed_time,
            )
            
            print(f"\n  ✓ Model saved: {saved_model.path}")
            print(f"    - Per-pipe roughness for {saved_model.n_pipes} pipes")
            print(f"    - Ready for direct use in dashboard")
        else:
            print("\n  ⚠ Optimization did not converge - no model saved")
        
        return 0
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

