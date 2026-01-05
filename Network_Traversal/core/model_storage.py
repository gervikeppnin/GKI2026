#!/usr/bin/env python3
"""
Model Storage for Pipe Roughness Calibration.

This module handles saving and loading calibration models with per-pipe
roughness values. Training can use pipe groups for efficiency, but saved
models store expanded per-pipe values for direct application.

Model Format (v2):
    {
        "name": "model_name",
        "version": 2,
        "mae": 0.1523,
        "pipe_roughness": {"Pipe-1": 95.2, "Pipe-2": 102.1, ...},
        "metadata": {...}
    }
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .config import DATA_DIR, ROUGHNESS_DEFAULT

logger = logging.getLogger(__name__)

# Model format version
MODEL_VERSION = 2

# Default models directory
MODELS_DIR = Path(__file__).parent.parent / "models"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CalibrationModel:
    """A calibration model with per-pipe roughness values.
    
    Attributes:
        name: Human-readable model name
        mae: Mean Absolute Error achieved during training
        pipe_roughness: Roughness value for each individual pipe
        metadata: Additional training metadata
        version: Model format version
        path: File path if loaded from disk
    """
    name: str
    mae: float
    pipe_roughness: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = MODEL_VERSION
    path: Optional[Path] = None
    
    @property
    def n_pipes(self) -> int:
        """Number of pipes in the model."""
        return len(self.pipe_roughness)
    
    @property
    def algorithm(self) -> str:
        """Training algorithm used."""
        return self.metadata.get('algorithm', 'unknown')
    
    @property
    def training_date(self) -> Optional[str]:
        """Date used for training data."""
        return self.metadata.get('training_date')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'mae': float(self.mae),
            'pipe_roughness': {k: float(v) for k, v in self.pipe_roughness.items()},
            'metadata': self.metadata,
        }
    
    def save(self, path: Optional[Path] = None) -> Path:
        """Save model to JSON file."""
        if path is None:
            path = self.path or (MODELS_DIR / f"{self.name}.json")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        self.path = path
        logger.info(f"Saved model '{self.name}' to {path}")
        return path


# =============================================================================
# Core Functions
# =============================================================================

def expand_to_pipes(
    group_roughness: Dict[str, float],
    pipe_groups: Dict[str, List[str]],
    default_roughness: float = ROUGHNESS_DEFAULT
) -> Dict[str, float]:
    """Expand group roughness values to per-pipe roughness.
    
    Args:
        group_roughness: Roughness value for each group
        pipe_groups: Mapping from group name to list of pipe IDs
        default_roughness: Default value for unmapped pipes
        
    Returns:
        Dictionary mapping each pipe ID to its roughness value
    """
    pipe_roughness: Dict[str, float] = {}
    
    for group_name, pipe_ids in pipe_groups.items():
        c_value = group_roughness.get(group_name, default_roughness)
        for pipe_id in pipe_ids:
            pipe_roughness[pipe_id] = float(c_value)
    
    return pipe_roughness


def save_model(
    name: str,
    group_roughness: Dict[str, float],
    pipe_groups: Dict[str, List[str]],
    mae: float,
    algorithm: Optional[str] = None,
    grouping_strategy: Optional[str] = None,
    training_date: Optional[str] = None,
    models_dir: Optional[Path] = None,
    **extra_metadata
) -> CalibrationModel:
    """Save a calibration model with expanded per-pipe roughness.
    
    Args:
        name: Model name
        group_roughness: Roughness values by group (from training)
        pipe_groups: Mapping from group name to pipe IDs
        mae: Mean Absolute Error achieved
        algorithm: Training algorithm used
        grouping_strategy: Group strategy used during training
        training_date: Date used for sensor data
        models_dir: Directory to save model (default: models/)
        **extra_metadata: Additional metadata to store
        
    Returns:
        CalibrationModel instance (also saved to disk)
    """
    # Expand groups to individual pipes
    pipe_roughness = expand_to_pipes(group_roughness, pipe_groups)
    
    # Build metadata
    metadata = {
        'algorithm': algorithm,
        'grouping_strategy': grouping_strategy,
        'training_date': training_date,
        'timestamp': datetime.now().isoformat(),
        'n_groups': len(group_roughness),
        'group_names': list(group_roughness.keys()),
        'groups_map': pipe_groups,
        **extra_metadata
    }
    
    # Create model
    model = CalibrationModel(
        name=name,
        mae=mae,
        pipe_roughness=pipe_roughness,
        metadata=metadata,
    )
    
    # Save
    save_dir = models_dir or MODELS_DIR
    filename = f"{name.replace(' ', '_').lower()}.json"
    model.save(save_dir / filename)
    
    return model


def load_model(path: Path) -> CalibrationModel:
    """Load a calibration model from JSON file.
    
    Handles both v1 (group-based) and v2 (per-pipe) formats.
    
    Args:
        path: Path to model JSON file
        
    Returns:
        CalibrationModel instance
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model format is invalid
    """
    with open(path) as f:
        data = json.load(f)
    
    version = data.get('version', 1)
    
    if version >= 2:
        # New format with per-pipe roughness
        return CalibrationModel(
            name=data['name'],
            mae=data['mae'],
            pipe_roughness=data['pipe_roughness'],
            metadata=data.get('metadata', {}),
            version=version,
            path=path,
        )
    else:
        # Legacy format with group roughness
        # We store the group roughness directly for now
        # It will need to be expanded when applied (requires pipe_groups)
        logger.warning(f"Loading legacy v1 model from {path}. Consider migrating.")
        
        return CalibrationModel(
            name=data.get('name', path.stem),
            mae=data.get('mae', 999.0),
            pipe_roughness=data.get('roughness', {}),  # Actually group roughness
            metadata={
                'legacy_format': True,
                'timestamp': data.get('timestamp'),
            },
            version=1,
            path=path,
        )


def list_models(models_dir: Optional[Path] = None) -> List[CalibrationModel]:
    """List all available calibration models.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        List of CalibrationModel instances, sorted by MAE (best first)
    """
    directory = models_dir or MODELS_DIR
    
    if not directory.exists():
        return []
    
    models = []
    for path in directory.glob("*.json"):
        try:
            model = load_model(path)
            models.append(model)
        except Exception as e:
            logger.warning(f"Failed to load model {path}: {e}")
            continue
    
    # Sort by MAE (best first)
    models.sort(key=lambda m: m.mae)
    
    return models


def migrate_legacy_model(
    legacy_path: Path,
    pipe_groups: Dict[str, List[str]],
    output_dir: Optional[Path] = None
) -> CalibrationModel:
    """Migrate a v1 (group-based) model to v2 (per-pipe) format.
    
    Args:
        legacy_path: Path to legacy model file
        pipe_groups: Pipe groups to use for expansion
        output_dir: Directory for migrated model
        
    Returns:
        New CalibrationModel in v2 format
    """
    # Load legacy model
    with open(legacy_path) as f:
        data = json.load(f)
    
    group_roughness = data.get('roughness', {})
    
    # Expand to per-pipe
    pipe_roughness = expand_to_pipes(group_roughness, pipe_groups)
    
    # Create new model
    model = CalibrationModel(
        name=data.get('name', legacy_path.stem) + '_migrated',
        mae=data.get('mae', 999.0),
        pipe_roughness=pipe_roughness,
        metadata={
            'migrated_from': str(legacy_path),
            'original_groups': group_roughness,
            'migration_timestamp': datetime.now().isoformat(),
        },
    )
    
    # Save
    save_dir = output_dir or MODELS_DIR
    model.save(save_dir / f"{model.name}.json")
    
    logger.info(f"Migrated {legacy_path.name} → {model.name}.json")
    return model


# =============================================================================
# Helper Functions
# =============================================================================

def get_best_model(models_dir: Optional[Path] = None) -> Optional[CalibrationModel]:
    """Get the model with lowest MAE."""
    models = list_models(models_dir)
    return models[0] if models else None


def model_summary(model: CalibrationModel) -> str:
    """Generate a text summary of a model."""
    lines = [
        f"Model: {model.name}",
        f"MAE: {model.mae:.4f} bar",
        f"Pipes: {model.n_pipes}",
        f"Algorithm: {model.algorithm}",
    ]
    if model.training_date:
        lines.append(f"Training Date: {model.training_date}")
    return "\n".join(lines)


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("=" * 60)
    print("Testing Model Storage Module")
    print("=" * 60)
    
    # List existing models
    models = list_models()
    print(f"\nFound {len(models)} models:")
    for m in models:
        print(f"  - {m.name}: MAE={m.mae:.4f}, pipes={m.n_pipes}, v{m.version}")
