#!/usr/bin/env python3
"""
Entry point for IG V2 Pipe Roughness Calibration.
"""
import sys
import os
import logging
import argparse
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from Network_Traversal.core.engine import SimulationEngine
from Network_Traversal.training.ig_v2.agent import PipeRoughnessCalibrationIGV2
from Network_Traversal.training.ig_v2.data import DataPipeline
from Network_Traversal.training.ig_v2 import config

import warnings
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
# Suppress noisy WNTR logs
logging.getLogger("wntr").setLevel(logging.ERROR)
# Suppress specific engine logs if needed, but keeping agent logs mostly
logging.getLogger("Network_Traversal.core.engine").setLevel(logging.WARNING) 
logging.getLogger("Network_Traversal.training.ig_v2.data").setLevel(logging.WARNING)

# Suppress Warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("IG_V2_Runner")

def main():
    parser = argparse.ArgumentParser(description="Run IG V2 Calibration Agent")
    parser.add_argument("--budget", type=float, default=None, help="Override default budget")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to data directory")
    parser.add_argument("--init-model", type=str, default=None, help="Name of model to initialize from (in models/ dir)")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (DDMMYY) for training data")
    parser.add_argument('--end-date', type=str, help='Filter data end date (DDMMYY)')
    parser.add_argument('--max-gp-samples', type=int, default=100, help='Max GP samples to keep (pruning limit)')
    args = parser.parse_args()

    # Override config if budget provided
    if args.budget:
        config.TOTAL_BUDGET_UNITS = args.budget
        logger.info(f"Overriding budget to {args.budget}")

    logger.info("Initializing components...")
    
    # Initialize Engine (Data handling via engine or pipeline?)
    # SimulationEngine loads data internally based on DATA_DIR config
    sim_engine = SimulationEngine(data_dir=Path(args.data_dir) if args.data_dir else None)
    
    # Initialize Pipeline
    # DataPipeline also loads data. Ideally they share the same source.
    data_pipeline = DataPipeline(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Load initial model if provided
    initial_roughness = None
    if args.init_model:
        from Network_Traversal.core.model_storage import load_model, MODELS_DIR
        model_path = MODELS_DIR / f"{args.init_model}.json"
        if not model_path.exists():
            # Try as exact path
            model_path = Path(args.init_model)
            
        if model_path.exists():
            try:
                loaded_model = load_model(model_path)
                logger.info(f"Loaded initialization model: {loaded_model.name} (MAE: {loaded_model.mae:.4f})")
                initial_roughness = loaded_model.pipe_roughness
            except Exception as e:
                logger.error(f"Failed to load initialization model: {e}")
        else:
            logger.warning(f"Initialization model not found: {args.init_model}")

    # Calculate dynamic parameters based on actual selected data
    active_days = len(data_pipeline.dates)
    if active_days > 0:
        # Scale cost linearly? Config has COST_FULL_WINDOW_SIM for default window (20).
        # Let's use unit cost.
        from Network_Traversal.training.ig_v2.config import COST_1_DAY_SIM
        dynamic_full_cost = COST_1_DAY_SIM * active_days
        logger.info(f"Dynamic Configuration: Window={active_days} days, FullSimCost={dynamic_full_cost}")
    else:
        # Should catch empty before here, but fallback
        active_days = config.WINDOW_SIZE_DAYS
        dynamic_full_cost = config.COST_FULL_WINDOW_SIM

    # Initialize Agent
    agent = PipeRoughnessCalibrationIGV2(
        data_pipeline, 
        sim_engine, 
        initial_pipe_roughness=initial_roughness,
        budget=config.TOTAL_BUDGET_UNITS, # config was updated with args.budget if present
        cost_full_sim=dynamic_full_cost,
        window_size=active_days,
        max_gp_samples=args.max_gp_samples
    )
    
    logger.info("Starting Agent...")
    theta_star, g_star_mapping = agent.run()
    
    logger.info("Calibration Complete.")
    logger.info(f"Final Roughness (Theta): {theta_star}")
    
    # Invert mapping for storage: (pipe -> group) to (group -> [pipes])
    pipe_groups_inverted = {}
    for pipe, group in g_star_mapping.items():
        if group not in pipe_groups_inverted:
            pipe_groups_inverted[group] = []
        pipe_groups_inverted[group].append(pipe)
    
    # Save results using standard storage
    from Network_Traversal.core.model_storage import save_model
    
    # Create models dir if not exists (though save_model handles parent dir creation)
    # User requested: Network_Traversal/models
    models_dir = Path("Network_Traversal/models")
    models_dir.mkdir(exist_ok=True, parents=True) # Ensure parent exists if not running from root
    
    saved_path = save_model(
        name="IG_V2_Calibrated",
        group_roughness=theta_star,
        pipe_groups=pipe_groups_inverted,
        mae=agent.best_mae,
        algorithm="IG_V2",
        training_date=None, # or fetch from agent if single-date
        models_dir=models_dir
    )
        
    logger.info(f"Model saved to {saved_path}")

    # Save Surrogate (Brain) if available
    if agent.best_surrogate:
        import joblib
        surrogate_path = models_dir / f"{saved_path.path.stem}_surrogate.pkl"
        try:
            joblib.dump(agent.best_surrogate, surrogate_path)
            logger.info(f"Surrogate (Brain) saved to {surrogate_path}")
        except Exception as e:
            logger.error(f"Failed to save surrogate: {e}")

if __name__ == "__main__":
    main()
