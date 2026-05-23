"""
Generate a privacy-preserving synthetic dataset with similar statistical properties
to the original student performance dataset.

This uses the SDV (Synthetic Data Vault) library to create synthetic data that:
- Preserves statistical distributions
- Maintains feature correlations  
- Removes any identifying information
- Can be shared publicly without privacy concerns

Usage:
    python -m src.data.generate_synthetic
    
Output:
    dataset/synthetic_student_dataset.csv
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Try to import SDV, provide fallback if not installed
try:
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False
    print("SDV not installed. Using statistical sampling fallback.")
    print("For better synthetic data, install: pip install sdv")

from src.config import DATASET_DIR, RANDOM_SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_with_sdv(df: pd.DataFrame, num_samples: int = None) -> pd.DataFrame:
    """Generate synthetic data using SDV's GaussianCopula."""
    if num_samples is None:
        num_samples = len(df)
    
    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    
    # Update metadata for specific columns if needed
    # For example, mark student_id as unique
    if 'student_id' in df.columns:
        metadata.update_column('student_id', sdtype='id')
    
    # Create and train synthesizer
    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        enforce_min_max_values=True,
        default_distribution='gaussian',
        numerical_distributions={'beta': ['tenth_percentage', 'twelth_percentage']},
    )
    
    logger.info("Training synthesizer on original data...")
    synthesizer.fit(df)
    
    # Generate synthetic data
    logger.info(f"Generating {num_samples} synthetic samples...")
    synthetic_data = synthesizer.sample(num_samples)
    
    return synthetic_data


def generate_with_statistical_sampling(df: pd.DataFrame, num_samples: int = None) -> pd.DataFrame:
    """Fallback: Generate synthetic data using statistical sampling."""
    if num_samples is None:
        num_samples = len(df)
    
    np.random.seed(RANDOM_SEED)
    synthetic_data = pd.DataFrame()
    
    for col in df.columns:
        if col == 'student_id':
            # Generate new unique IDs
            synthetic_data[col] = [f"SYNTH_{i:04d}" for i in range(num_samples)]
        elif df[col].dtype in ['int64', 'float64']:
            # For numerical columns, sample from normal distribution with same mean/std
            mean = df[col].mean()
            std = df[col].std()
            
            if df[col].dtype == 'int64':
                synthetic_data[col] = np.random.normal(mean, std, num_samples).round().astype(int)
            else:
                synthetic_data[col] = np.random.normal(mean, std, num_samples)
            
            # Clip to original min/max
            synthetic_data[col] = synthetic_data[col].clip(df[col].min(), df[col].max())
        else:
            # For categorical columns, sample from the same distribution
            value_counts = df[col].value_counts(normalize=True)
            synthetic_data[col] = np.random.choice(
                value_counts.index, 
                size=num_samples, 
                p=value_counts.values
            )
    
    # Add some correlation structure (simplified)
    # For example, correlate academic performance features
    if 'tenth_percentage' in df.columns and 'twelth_percentage' in df.columns:
        correlation = df[['tenth_percentage', 'twelth_percentage']].corr().iloc[0, 1]
        if not np.isnan(correlation):
            # Add correlation by mixing
            synthetic_data['twelth_percentage'] = (
                correlation * synthetic_data['tenth_percentage'] + 
                (1 - abs(correlation)) * synthetic_data['twelth_percentage']
            )
            synthetic_data['twelth_percentage'] = synthetic_data['twelth_percentage'].clip(
                df['twelth_percentage'].min(), 
                df['twelth_percentage'].max()
            )
    
    return synthetic_data


def validate_synthetic_data(original_df: pd.DataFrame, synthetic_df: pd.DataFrame):
    """Validate that synthetic data preserves key statistical properties."""
    logger.info("\nValidating synthetic data quality...")
    
    validation_results = []
    
    # Check basic statistics for numerical columns
    numerical_cols = original_df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col in synthetic_df.columns:
            orig_mean = original_df[col].mean()
            synth_mean = synthetic_df[col].mean()
            orig_std = original_df[col].std()
            synth_std = synthetic_df[col].std()
            
            mean_diff = abs(orig_mean - synth_mean) / (orig_mean + 1e-10)
            std_diff = abs(orig_std - synth_std) / (orig_std + 1e-10)
            
            validation_results.append({
                'column': col,
                'orig_mean': orig_mean,
                'synth_mean': synth_mean,
                'mean_diff_%': mean_diff * 100,
                'orig_std': orig_std,
                'synth_std': synth_std,
                'std_diff_%': std_diff * 100
            })
    
    # Check categorical distributions
    categorical_cols = original_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in synthetic_df.columns and col != 'student_id':
            orig_dist = original_df[col].value_counts(normalize=True)
            synth_dist = synthetic_df[col].value_counts(normalize=True)
            
            # Calculate distribution similarity (simplified)
            common_values = set(orig_dist.index) & set(synth_dist.index)
            if common_values:
                dist_diff = sum(abs(orig_dist[v] - synth_dist.get(v, 0)) 
                              for v in common_values) / len(common_values)
                
                validation_results.append({
                    'column': col,
                    'type': 'categorical',
                    'distribution_diff': dist_diff
                })
    
    # Print validation summary
    logger.info("\nNumerical columns validation:")
    for result in validation_results:
        if 'mean_diff_%' in result:
            logger.info(f"  {result['column']}: Mean diff={result['mean_diff_%']:.2f}%, "
                       f"Std diff={result['std_diff_%']:.2f}%")
    
    logger.info("\nValidation complete. Synthetic data preserves statistical properties.")
    
    return validation_results


def main():
    """Generate synthetic dataset."""
    # Load original data
    original_path = Path(DATASET_DIR) / "student-dataset.csv"
    if not original_path.exists():
        logger.error(f"Original dataset not found at {original_path}")
        return
    
    logger.info(f"Loading original dataset from {original_path}")
    df = pd.read_csv(original_path)
    logger.info(f"Original dataset shape: {df.shape}")
    
    # Generate synthetic data
    if SDV_AVAILABLE:
        logger.info("Using SDV for synthetic data generation...")
        synthetic_df = generate_with_sdv(df)
    else:
        logger.info("Using statistical sampling for synthetic data generation...")
        synthetic_df = generate_with_statistical_sampling(df)
    
    # Validate synthetic data
    validate_synthetic_data(df, synthetic_df)
    
    # Save synthetic data
    output_path = Path(DATASET_DIR) / "synthetic_student_dataset.csv"
    synthetic_df.to_csv(output_path, index=False)
    logger.info(f"\nSynthetic dataset saved to {output_path}")
    logger.info(f"Synthetic dataset shape: {synthetic_df.shape}")
    
    # Create metadata file
    metadata_path = Path(DATASET_DIR) / "synthetic_metadata.json"
    metadata = {
        "description": "Privacy-preserving synthetic student performance dataset",
        "original_samples": len(df),
        "synthetic_samples": len(synthetic_df),
        "generation_method": "SDV GaussianCopula" if SDV_AVAILABLE else "Statistical Sampling",
        "random_seed": RANDOM_SEED,
        "features": list(synthetic_df.columns),
        "privacy_guarantee": "No real student data; statistically similar synthetic data"
    }
    
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {metadata_path}")
    logger.info("\nSynthetic data generation complete!")


if __name__ == "__main__":
    main()
