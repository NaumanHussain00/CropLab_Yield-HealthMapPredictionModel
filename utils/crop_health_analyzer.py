from datetime import date
import logging

from .date_resolver import resolve_reference_date, get_historical_dates
from .crop_health_gee import fetch_all_ndvi, ServiceError
from .crop_health_classifier import compute_anomaly, classify_pixels, generate_tile_urls
from .crop_health_stats import compute_summary_stats

logger = logging.getLogger(__name__)

def analyze_crop_health(
    geometry: list,
    cultivation_date: date,
    harvest_date: date
) -> dict:
    """
    Analyze crop health using satellite imagery.
    
    Args:
        geometry: [[lat, lon], [lat, lon], ...] polygon vertices
        cultivation_date: date object
        harvest_date: date object
    
    Returns:
        {
            "reference_date": "YYYY-MM-DD",
            "years_used": ["YYYY-MM-DD", "YYYY-MM-DD", "YYYY-MM-DD"],
            "tile_urls": {
                "current_ndvi": "https://...",
                "anomaly_classified": "https://..."
            },
            "summary": {
                "healthy_percent": float,
                "stressed_percent": float,
                "unclassified_percent": float
            },
            "cache_hits": [bool, bool, bool]
        }
    """
    logger.info("=" * 60)
    logger.info("Starting crop health analysis...")
    logger.info(f"Cultivation: {cultivation_date}, Harvest: {harvest_date}")
    
    # Step 1: Resolve reference date
    reference_date = resolve_reference_date(cultivation_date, harvest_date)
    logger.info(f"✅ Reference date: {reference_date}")
    
    # Step 2: Get historical dates
    dates_to_fetch = get_historical_dates(reference_date)
    dates_str = [d.strftime("%Y-%m-%d") for d in dates_to_fetch]
    logger.info(f"✅ Dates to fetch: {dates_str}")
    
    # Step 3: Fetch NDVI data
    logger.info("Fetching NDVI rasters...")
    ndvi_rasters = fetch_all_ndvi(geometry, dates_str)
    logger.info(f"✅ NDVI fetched (cache hits: {ndvi_rasters['cache_hits']})")
    
    # Step 4: Compute anomaly
    logger.info("Computing anomaly...")
    anomaly = compute_anomaly(ndvi_rasters)
    logger.info("✅ Anomaly computed")
    
    # Step 5: Classify pixels
    logger.info("Classifying pixels...")
    current_ndvi = ndvi_rasters['current']['raster']
    classified_image = classify_pixels(current_ndvi, anomaly)
    logger.info("✅ Pixels classified")
    
    # Step 6: Generate tile URLs
    logger.info("Generating tile URLs...")
    tile_urls = generate_tile_urls(current_ndvi, anomaly, geometry)
    logger.info("✅ Tile URLs generated")
    
    # Step 7: Compute summary stats
    logger.info("Computing statistics...")
    summary = compute_summary_stats(classified_image)
    logger.info("✅ Statistics computed")
    
    # Step 8: Build response
    result = {
        "reference_date": reference_date.strftime("%Y-%m-%d"),
        "years_used": dates_str,
        "tile_urls": tile_urls,
        "summary": summary,
        "cache_hits": ndvi_rasters['cache_hits']
    }
    
    logger.info("=" * 60)
    logger.info("✅ Analysis complete!")
    return result
