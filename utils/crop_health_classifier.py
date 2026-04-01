import numpy as np
import ee
import logging
from typing import Tuple
import base64
import io

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PIL not available - heatmap visualization disabled")

logger = logging.getLogger(__name__)

def compute_anomaly(ndvi_rasters: dict) -> np.ndarray:
    """
    Compute anomaly: current NDVI - historical mean
    
    ndvi_rasters: {
        "current": {"raster": ...},
        "year_1": {"raster": ...} or None,
        "year_2": {"raster": ...} or None
    }
    
    Returns: anomaly array (current - mean of available historical years)
    """
    current = ndvi_rasters['current']['raster']
    year_1 = ndvi_rasters['year_1']['raster'] if ndvi_rasters['year_1'] else None
    year_2 = ndvi_rasters['year_2']['raster'] if ndvi_rasters['year_2'] else None
    
    # Compute pixelwise mean of available historical years
    available_years = [y for y in [year_1, year_2] if y is not None]
    
    if len(available_years) == 0:
        # No historical data - use 0 as baseline
        logger.warning("⚠️ No historical NDVI data available. Using 0 as baseline.")
        historical_mean = np.zeros_like(current)
    elif len(available_years) == 1:
        logger.warning(f"⚠️ Only 1 year of historical data available. Using as baseline.")
        historical_mean = available_years[0]
    else:
        # 2 years available
        historical_mean = np.mean(available_years, axis=0)
    
    # Compute anomaly
    anomaly = current - historical_mean
    
    logger.info(f"Anomaly computed from {len(available_years)} historical year(s)")
    logger.info(f"Anomaly range: [{anomaly.min():.3f}, {anomaly.max():.3f}]")
    return anomaly

def classify_pixels(current_ndvi: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
    """
    Classify pixels as healthy/stressed/other.
    
    NDVI ranges:
    - Water/Buildings: < 0.2
    - Bare soil: 0.2 - 0.4
    - Sparse vegetation: 0.4 - 0.6
    - Dense vegetation: > 0.6
    
    Stress indicators:
    - High NDVI + negative anomaly = unhealthy trend
    - Low NDVI + negative anomaly = definitely stressed
    
    Returns: RGBA image for visualization
    """
    height, width = current_ndvi.shape
    classified = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA
    
    # Define vegetated areas (NDVI > 0.4)
    vegetated = current_ndvi > 0.4
    
    # Healthy: Vegetated AND anomaly >= -0.05 (stable or improving)
    healthy_mask = vegetated & (anomaly >= -0.05)
    classified[healthy_mask] = [34, 197, 94, 255]  # Green #22c55e
    
    # Stressed: Vegetated AND anomaly < -0.05 (declining trend)
    stressed_mask = vegetated & (anomaly < -0.05)
    classified[stressed_mask] = [239, 68, 68, 255]  # Red #ef4444
    
    # Sparse vegetation (0.2-0.4): Show as gray
    sparse = (current_ndvi >= 0.2) & (current_ndvi <= 0.4)
    classified[sparse] = [156, 163, 175, 200]  # Gray #9ca3af with alpha
    
    # Non-vegetated (< 0.2): Keep transparent
    
    healthy_pct = (healthy_mask.sum() / vegetated.sum() * 100) if vegetated.sum() > 0 else 0
    stressed_pct = (stressed_mask.sum() / vegetated.sum() * 100) if vegetated.sum() > 0 else 0
    
    logger.info(f"Classification (from {vegetated.sum()} vegetated pixels):")
    logger.info(f"  Healthy: {healthy_mask.sum()} pixels ({healthy_pct:.1f}%)")
    logger.info(f"  Stressed: {stressed_mask.sum()} pixels ({stressed_pct:.1f}%)")
    logger.info(f"  Sparse vegetation: {sparse.sum()} pixels")
    logger.info(f"NDVI range: [{current_ndvi.min():.3f}, {current_ndvi.max():.3f}]")
    logger.info(f"Anomaly range: [{anomaly.min():.3f}, {anomaly.max():.3f}]")
    
    return classified

def generate_tile_urls(
    current_ndvi: np.ndarray,
    anomaly: np.ndarray,
    geometry: list
) -> dict:
    """
    Generate NDVI heatmap visualizations as base64-encoded PNG data URLs.
    
    Returns:
        {
            "ndvi_heatmap": "data:image/png;base64,...",
            "anomaly_heatmap": "data:image/png;base64,...",
            "ndvi_range": [min, max],
            "anomaly_range": [min, max]
        }
    """
    try:
        result = {}
        
        # Log actual data ranges
        ndvi_min, ndvi_max = float(np.nanmin(current_ndvi)), float(np.nanmax(current_ndvi))
        anomaly_min, anomaly_max = float(np.nanmin(anomaly)), float(np.nanmax(anomaly))
        
        logger.info(f"Generating visualization tiles:")
        logger.info(f"  NDVI range: [{ndvi_min:.4f}, {ndvi_max:.4f}]")
        logger.info(f"  Anomaly range: [{anomaly_min:.4f}, {anomaly_max:.4f}]")
        
        result["ndvi_range"] = [ndvi_min, ndvi_max]
        result["anomaly_range"] = [anomaly_min, anomaly_max]
        
        if PILLOW_AVAILABLE:
            # Generate NDVI heatmap (Red-Yellow-Green colormap)
            if ndvi_max <= ndvi_min:
                # All same value - use middle color (yellow)
                ndvi_normalized = np.full_like(current_ndvi, 0.5)
                logger.warning("⚠️ NDVI is constant - using yellow for visualization")
            else:
                ndvi_normalized = np.clip((current_ndvi - ndvi_min) / (ndvi_max - ndvi_min + 1e-6), 0, 1)
            
            ndvi_rgb = ndvi_to_rgb(ndvi_normalized)
            ndvi_img = Image.fromarray(ndvi_rgb, 'RGB')
            
            # Convert to base64
            ndvi_buffer = io.BytesIO()
            ndvi_img.save(ndvi_buffer, format='PNG')
            ndvi_base64 = base64.b64encode(ndvi_buffer.getvalue()).decode('utf-8')
            result["ndvi_heatmap"] = f"data:image/png;base64,{ndvi_base64}"
            logger.info("✅ NDVI heatmap generated")
            
            # Generate anomaly heatmap (Blue-White-Red colormap)
            if anomaly_max <= anomaly_min:
                # All same value - use white (neutral)
                anomaly_normalized = np.full_like(anomaly, 0.5)
                logger.warning("⚠️ Anomaly is constant - using white for visualization")
            else:
                anomaly_normalized = np.clip((anomaly - anomaly_min) / (anomaly_max - anomaly_min + 1e-6), 0, 1)
            
            anomaly_rgb = anomaly_to_rgb(anomaly_normalized, threshold_normalized=0.5)
            anomaly_img = Image.fromarray(anomaly_rgb, 'RGB')
            
            # Convert to base64
            anomaly_buffer = io.BytesIO()
            anomaly_img.save(anomaly_buffer, format='PNG')
            anomaly_base64 = base64.b64encode(anomaly_buffer.getvalue()).decode('utf-8')
            result["anomaly_heatmap"] = f"data:image/png;base64,{anomaly_base64}"
            logger.info("✅ Anomaly heatmap generated")
        else:
            logger.warning("⚠️ PIL not available - returning data ranges only")
            result["note"] = "Heatmap visualization requires Pillow library"
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating tiles: {e}")
        return {"error": str(e)}


def ndvi_to_rgb(ndvi_normalized: np.ndarray) -> np.ndarray:
    """
    Convert normalized NDVI (0-1) to RGB using Red-Yellow-Green colormap.
    
    0.0 (Low NDVI) → Red (255, 0, 0)
    0.5 (Medium NDVI) → Yellow (255, 255, 0)
    1.0 (High NDVI) → Green (0, 255, 0)
    """
    height, width = ndvi_normalized.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Red/Yellow gradient (0.0 - 0.5)
    low_mask = ndvi_normalized < 0.5
    rgb[low_mask, 0] = 255  # Red channel always full
    rgb[low_mask, 1] = (ndvi_normalized[low_mask] * 2 * 255).astype(np.uint8)  # Green increases
    
    # Yellow/Green gradient (0.5 - 1.0)
    high_mask = ndvi_normalized >= 0.5
    rgb[high_mask, 0] = (255 * (1 - (ndvi_normalized[high_mask] - 0.5) * 2)).astype(np.uint8)  # Red decreases
    rgb[high_mask, 1] = 255  # Green always full
    
    return rgb


def anomaly_to_rgb(anomaly_normalized: np.ndarray, threshold_normalized: float = 0.5) -> np.ndarray:
    """
    Convert normalized anomaly (0-1) to RGB using Blue-White-Red colormap.
    
    0.0 → Blue (high negative anomaly - stressed)
    0.5 → White (no anomaly - stable)
    1.0 → Red (high positive anomaly - thriving)
    """
    height, width = anomaly_normalized.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Blue/White gradient (0.0 - 0.5)
    low_mask = anomaly_normalized < threshold_normalized
    rgb[low_mask, 0] = (anomaly_normalized[low_mask] * 2 * 255).astype(np.uint8)  # Red increases
    rgb[low_mask, 1] = (anomaly_normalized[low_mask] * 2 * 255).astype(np.uint8)  # Green increases
    rgb[low_mask, 2] = 255  # Blue always full
    
    # White/Red gradient (0.5 - 1.0)
    high_mask = anomaly_normalized >= threshold_normalized
    rgb[high_mask, 0] = 255  # Red always full
    rgb[high_mask, 1] = (255 * (1 - (anomaly_normalized[high_mask] - threshold_normalized) * 2)).astype(np.uint8)  # Green decreases
    rgb[high_mask, 2] = (255 * (1 - (anomaly_normalized[high_mask] - threshold_normalized) * 2)).astype(np.uint8)  # Blue decreases
    
    return rgb
