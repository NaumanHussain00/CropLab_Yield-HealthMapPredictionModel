import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_summary_stats(classified_image: np.ndarray) -> dict:
    """
    Compute pixel distribution statistics.
    
    classified_image: RGBA image from classify_pixels()
    Categories:
    - Green [34, 197, 94, 255]: Healthy vegetation
    - Red [239, 68, 68, 255]: Stressed vegetation
    - Gray [156, 163, 175, 200]: Sparse vegetation
    - Transparent [*, *, *, 0]: Non-vegetated/unclassified
    
    Returns:
        {
            "healthy_percent": float (% of vegetated),
            "stressed_percent": float (% of vegetated),
            "sparse_percent": float (% of vegetated),
            "vegetation_coverage_percent": float (% of total),
            "unclassified_percent": float (% of total)
        }
    """
    # Define color masks
    healthy_mask = (
        (classified_image[:, :, 0] == 34) &
        (classified_image[:, :, 1] == 197) &
        (classified_image[:, :, 2] == 94) &
        (classified_image[:, :, 3] == 255)
    )
    
    stressed_mask = (
        (classified_image[:, :, 0] == 239) &
        (classified_image[:, :, 1] == 68) &
        (classified_image[:, :, 2] == 68) &
        (classified_image[:, :, 3] == 255)
    )
    
    sparse_mask = (
        (classified_image[:, :, 0] == 156) &
        (classified_image[:, :, 1] == 163) &
        (classified_image[:, :, 2] == 175)
    )
    
    # Count pixels
    healthy_count = healthy_mask.sum()
    stressed_count = stressed_mask.sum()
    sparse_count = sparse_mask.sum()
    vegetated_count = healthy_count + stressed_count + sparse_count
    total_pixels = classified_image.shape[0] * classified_image.shape[1]
    unclassified_count = total_pixels - vegetated_count
    
    if vegetated_count == 0:
        logger.warning("⚠️ No vegetated pixels detected (NDVI < 0.2 across all pixels)")
        return {
            "healthy_percent": 0.0,
            "stressed_percent": 0.0,
            "sparse_percent": 0.0,
            "vegetation_coverage_percent": 0.0,
            "unclassified_percent": 100.0
        }
    
    # Calculate percentages (relative to vegetated area for health breakdown)
    stats = {
        "healthy_percent": round((healthy_count / vegetated_count) * 100, 2),
        "stressed_percent": round((stressed_count / vegetated_count) * 100, 2),
        "sparse_percent": round((sparse_count / vegetated_count) * 100, 2),
        "vegetation_coverage_percent": round((vegetated_count / total_pixels) * 100, 2),
        "unclassified_percent": round((unclassified_count / total_pixels) * 100, 2)
    }
    
    logger.info("Summary Statistics:")
    logger.info(f"  Vegetation coverage: {stats['vegetation_coverage_percent']:.1f}%")
    logger.info(f"  Health breakdown (of vegetated area):")
    logger.info(f"    Healthy: {stats['healthy_percent']:.1f}%")
    logger.info(f"    Stressed: {stats['stressed_percent']:.1f}%")
    logger.info(f"    Sparse: {stats['sparse_percent']:.1f}%")
    
    return stats
