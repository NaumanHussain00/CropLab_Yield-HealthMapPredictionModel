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
        "current": {"raster": ...} or None,
        "year_1": {"raster": ...} or None,
        "year_2": {"raster": ...} or None
    }

    Returns: anomaly array (current - mean of available historical years)
    """
    def sanitize_raster(raster: np.ndarray) -> np.ndarray:
        """Replace -inf/-nan values with NaN for proper statistical handling."""
        if raster is None:
            return None
        # Replace -inf and +inf with NaN (nanmean will ignore them)
        sanitized = raster.copy()
        sanitized[~np.isfinite(sanitized)] = np.nan
        return sanitized

    # Handle case where current year data is not available
    current = None
    if ndvi_rasters['current'] is not None:
        current = sanitize_raster(ndvi_rasters['current']['raster'])

    year_1 = sanitize_raster(ndvi_rasters['year_1']['raster']) if ndvi_rasters['year_1'] else None
    year_2 = sanitize_raster(ndvi_rasters['year_2']['raster']) if ndvi_rasters['year_2'] else None

    # If current is None, use the most recent available year (year_1)
    if current is None:
        if year_1 is not None:
            logger.warning("Current year imagery unavailable. Using year_1 as current.")
            current = year_1
            available_years = [year_2] if year_2 is not None else []
        elif year_2 is not None:
            logger.warning("Current and year_1 imagery unavailable. Using year_2 as current.")
            current = year_2
            available_years = []
        else:
            logger.error("No NDVI data available for any year")
            raise ValueError("No NDVI data available for anomaly computation")
    else:
        # Compute pixelwise mean of available historical years
        available_years = [y for y in [year_1, year_2] if y is not None]

    # Ensure all rasters have the same 2D shape
    if current.ndim != 2:
        logger.warning(f"Current raster has {current.ndim} dims, squeezing to 2D")
        current = np.squeeze(current)
    target_shape = current.shape
    logger.info(f"Raster target shape: {target_shape}")

    validated_years = []
    for i, yr in enumerate(available_years):
        if yr.ndim != 2:
            yr = np.squeeze(yr)
        if yr.shape != target_shape:
            logger.warning(f"Year {i} shape {yr.shape} != target {target_shape}, resizing")
            from PIL import Image as PILImage
            yr_img = PILImage.fromarray(yr)
            yr_img = yr_img.resize((target_shape[1], target_shape[0]), PILImage.BILINEAR)
            yr = np.array(yr_img, dtype=np.float32)
        validated_years.append(yr)
    available_years = validated_years

    if len(available_years) == 0:
        logger.warning("No historical NDVI data available. Using 0 as baseline.")
        historical_mean = np.zeros_like(current)
    elif len(available_years) == 1:
        logger.warning("Only 1 year of historical data available. Using as baseline.")
        historical_mean = available_years[0]
    else:
        historical_mean = np.nanmean(available_years, axis=0)

    # Compute anomaly
    anomaly = current - historical_mean

    # Log valid range before sanitization
    valid_anomaly = anomaly[np.isfinite(anomaly)]
    if len(valid_anomaly) > 0:
        logger.info(f"Anomaly computed from {len(available_years)} historical year(s)")
        logger.info(f"Anomaly range (valid pixels): [{valid_anomaly.min():.3f}, {valid_anomaly.max():.3f}]")
        logger.info(f"Anomaly mean: {valid_anomaly.mean():.3f}, std: {valid_anomaly.std():.3f}")
    else:
        logger.warning("All anomaly values are invalid (inf/nan)")

    return anomaly

def classify_pixels(current_ndvi: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
    """
    Classify pixels as healthy/stressed/other.

    Returns: RGBA image for visualization
    """
    # Sanitize anomaly for classification (replace NaN/inf with 0)
    anomaly_clean = np.nan_to_num(anomaly, nan=0.0, posinf=0.0, neginf=0.0)

    height, width = current_ndvi.shape
    classified = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA

    # Define vegetated areas (NDVI > 0.4)
    vegetated = current_ndvi > 0.4

    # Healthy: Vegetated AND anomaly >= -0.05 (stable or improving)
    healthy_mask = vegetated & (anomaly_clean >= -0.05)
    classified[healthy_mask] = [34, 197, 94, 255]  # Green #22c55e

    # Stressed: Vegetated AND anomaly < -0.05 (declining trend)
    stressed_mask = vegetated & (anomaly_clean < -0.05)
    classified[stressed_mask] = [239, 68, 68, 255]  # Red #ef4444

    # Sparse vegetation (0.2-0.4): Show as gray
    sparse = (current_ndvi >= 0.2) & (current_ndvi <= 0.4)
    classified[sparse] = [156, 163, 175, 200]  # Gray #9ca3af with alpha

    healthy_pct = (healthy_mask.sum() / vegetated.sum() * 100) if vegetated.sum() > 0 else 0
    stressed_pct = (stressed_mask.sum() / vegetated.sum() * 100) if vegetated.sum() > 0 else 0

    logger.info(f"Classification (from {vegetated.sum()} vegetated pixels):")
    logger.info(f"  Healthy: {healthy_mask.sum()} pixels ({healthy_pct:.1f}%)")
    logger.info(f"  Stressed: {stressed_mask.sum()} pixels ({stressed_pct:.1f}%)")
    logger.info(f"  Sparse vegetation: {sparse.sum()} pixels")
    logger.info(f"NDVI range: [{current_ndvi.min():.3f}, {current_ndvi.max():.3f}]")
    logger.info(f"Anomaly range: [{anomaly_clean.min():.3f}, {anomaly_clean.max():.3f}]")

    return classified

def generate_tile_urls(
    current_ndvi: np.ndarray,
    anomaly: np.ndarray,
    geometry: list
) -> dict:
    """
    Generate NDVI heatmap visualizations as base64-encoded PNG data URLs.
    """
    try:
        result = {}

        # Determine valid pixel mask (pixels inside the polygon with real data)
        valid_pixel_mask = np.isfinite(current_ndvi)
        valid_count = valid_pixel_mask.sum()
        total_count = valid_pixel_mask.size

        # Sanitize anomaly: replace NaN/inf with 0 for visualization
        anomaly_safe = np.where(np.isfinite(anomaly), anomaly, 0.0).astype(np.float32)

        # Log ranges using only valid pixels
        if valid_count > 0:
            valid_ndvi = current_ndvi[valid_pixel_mask]
            valid_anom = anomaly_safe[valid_pixel_mask]
            ndvi_min, ndvi_max = float(np.min(valid_ndvi)), float(np.max(valid_ndvi))
            anom_min, anom_max = float(np.min(valid_anom)), float(np.max(valid_anom))
        else:
            ndvi_min, ndvi_max = 0.0, 0.0
            anom_min, anom_max = 0.0, 0.0

        logger.info(f"Generating visualization tiles:")
        logger.info(f"  Valid pixels: {valid_count}/{total_count} ({valid_count/max(total_count,1)*100:.1f}%)")
        logger.info(f"  NDVI range (valid): [{ndvi_min:.4f}, {ndvi_max:.4f}]")
        logger.info(f"  Anomaly range (valid): [{anom_min:.4f}, {anom_max:.4f}]")
        logger.info(f"  Array shapes - NDVI: {current_ndvi.shape}, Anomaly: {anomaly.shape}")

        result["ndvi_range"] = [ndvi_min, ndvi_max]
        result["anomaly_range"] = [anom_min, anom_max]

        if not PILLOW_AVAILABLE:
            logger.warning("PIL not available - returning data ranges only")
            result["note"] = "Heatmap visualization requires Pillow library"
            return result

        # ── NDVI heatmap (Red-Yellow-Green) ──
        if ndvi_max <= ndvi_min:
            ndvi_normalized = np.full_like(current_ndvi, 0.5)
        else:
            ndvi_normalized = np.clip((current_ndvi - ndvi_min) / (ndvi_max - ndvi_min + 1e-6), 0, 1)

        ndvi_rgb = ndvi_to_rgb(ndvi_normalized)

        ndvi_img = Image.fromarray(ndvi_rgb, 'RGB')
        ndvi_buffer = io.BytesIO()
        ndvi_img.save(ndvi_buffer, format='PNG')
        result["ndvi_heatmap"] = f"data:image/png;base64,{base64.b64encode(ndvi_buffer.getvalue()).decode('utf-8')}"
        logger.info("NDVI heatmap generated")

        # ── Anomaly heatmap (Blue-White-Red, RGBA with transparency) ──
        if valid_count == 0 or abs(anom_max - anom_min) < 1e-8:
            # No valid data or constant anomaly → neutral
            anomaly_normalized = np.full_like(anomaly_safe, 0.5)
            logger.warning("Anomaly is constant or no valid data - using neutral color")
        else:
            # Use percentile-based normalization on valid pixels for maximum contrast
            valid_anom = anomaly_safe[valid_pixel_mask]
            p2 = float(np.percentile(valid_anom, 2))
            p98 = float(np.percentile(valid_anom, 98))

            logger.info(f"  Anomaly percentiles (valid): p2={p2:.4f}, p98={p98:.4f}")

            # Determine if data is mostly one-sided or spans zero
            spans_zero = (p2 < -0.01) and (p98 > 0.01)

            if spans_zero:
                # Data spans both positive and negative → symmetric around 0
                abs_max = max(abs(p2), abs(p98))
                anomaly_normalized = np.clip((anomaly_safe / abs_max) / 2 + 0.5, 0, 1)
                logger.info(f"  Using symmetric normalization: [-{abs_max:.4f}, +{abs_max:.4f}]")
            else:
                # Data is mostly one-sided → use full colormap range for max contrast
                # Map p2→0.0 (blue), p98→1.0 (red) so spatial variation is visible
                anomaly_normalized = np.clip((anomaly_safe - p2) / (p98 - p2 + 1e-10), 0, 1)
                logger.info(f"  Using range normalization: [{p2:.4f}, {p98:.4f}] → [blue, red]")

        anomaly_rgba = anomaly_to_rgba(anomaly_normalized, valid_pixel_mask)

        anomaly_img = Image.fromarray(anomaly_rgba, 'RGBA')
        anomaly_buffer = io.BytesIO()
        anomaly_img.save(anomaly_buffer, format='PNG')
        result["anomaly_heatmap"] = f"data:image/png;base64,{base64.b64encode(anomaly_buffer.getvalue()).decode('utf-8')}"
        logger.info("Anomaly heatmap generated")

        return result

    except Exception as e:
        logger.error(f"Error generating tiles: {e}")
        return {"error": str(e)}



def ndvi_to_rgb(ndvi_normalized: np.ndarray) -> np.ndarray:
    """
    Convert normalized NDVI (0-1) to RGB using Red-Yellow-Green colormap.

    0.0 (Low NDVI) -> Red (255, 0, 0)
    0.5 (Medium NDVI) -> Yellow (255, 255, 0)
    1.0 (High NDVI) -> Green (0, 255, 0)
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


def anomaly_to_rgba(anomaly_normalized: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Convert normalized anomaly (0-1) to RGBA using Blue-White-Red diverging colormap.
    Pixels outside the valid mask are transparent.

    0.0 -> Dark Blue  (most negative anomaly)
    0.5 -> White      (middle / no change)
    1.0 -> Dark Red   (most positive anomaly)
    """
    height, width = anomaly_normalized.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # Blue gradient (0.0 - 0.5): Dark Blue -> White
    low_mask = anomaly_normalized < 0.5
    t = anomaly_normalized[low_mask] / 0.5  # 0→0, 0.5→1
    rgba[low_mask, 0] = (t * 255).astype(np.uint8)        # R: 0 → 255
    rgba[low_mask, 1] = (t * 255).astype(np.uint8)        # G: 0 → 255
    rgba[low_mask, 2] = (139 + t * 116).astype(np.uint8)  # B: 139 → 255

    # Red gradient (0.5 - 1.0): White -> Dark Red
    high_mask = anomaly_normalized >= 0.5
    t = (anomaly_normalized[high_mask] - 0.5) / 0.5  # 0→0, 0.5→1
    rgba[high_mask, 0] = (255 - t * 77).astype(np.uint8)   # R: 255 → 178
    rgba[high_mask, 1] = (255 - t * 255).astype(np.uint8)  # G: 255 → 0
    rgba[high_mask, 2] = (255 - t * 255).astype(np.uint8)  # B: 255 → 0

    # Only show valid pixels (inside polygon)
    rgba[valid_mask, 3] = 255

    return rgba
