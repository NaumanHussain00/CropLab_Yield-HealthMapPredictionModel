import ee
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from .crop_health_cache import get_cached_ndvi, set_cached_ndvi, get_cached_index, set_cached_index

logger = logging.getLogger(__name__)

class ServiceError(Exception):
    pass

def _mask_s2_clouds(image):
    """Apply SCL-based cloud mask to a Sentinel-2 image."""
    scl = image.select('SCL')
    clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(clear)


def _compute_ndvi(image):
    """Compute NDVI from a Sentinel-2 image and return single-band image."""
    return image.normalizedDifference(['B8', 'B4']).rename('ndvi').float()


def _export_raster(ee_image, ee_geometry, max_dim=256):
    """Export a single-band EE image clipped to geometry as a numpy array."""
    clipped = ee_image.clip(ee_geometry)

    bounds = ee_geometry.bounds().getInfo()
    coords = bounds['coordinates'][0]
    min_lon = min(c[0] for c in coords)
    max_lon = max(c[0] for c in coords)
    min_lat = min(c[1] for c in coords)
    max_lat = max(c[1] for c in coords)

    avg_lat = (min_lat + max_lat) / 2
    m_per_deg_lon = 111319 * np.cos(np.radians(avg_lat))
    m_per_deg_lat = 111139

    scale = 10
    width = max(50, int((max_lon - min_lon) * m_per_deg_lon / scale))
    height = max(50, int((max_lat - min_lat) * m_per_deg_lat / scale))

    if max(width, height) > max_dim:
        ratio = max_dim / max(width, height)
        width = max(50, int(width * ratio))
        height = max(50, int(height * ratio))

    scale_x = (max_lon - min_lon) / width
    scale_y = (max_lat - min_lat) / height

    request = {
        'expression': clipped,
        'fileFormat': 'NUMPY_NDARRAY',
        'grid': {
            'dimensions': {'width': width, 'height': height},
            'affineTransform': {
                'scaleX': scale_x, 'shearX': 0, 'translateX': min_lon,
                'shearY': 0, 'scaleY': -scale_y, 'translateY': max_lat
            },
            'crsCode': 'EPSG:4326'
        }
    }

    pixel_data = ee.data.computePixels(request)
    if pixel_data is None:
        raise ValueError("computePixels returned None")

    raster_raw = np.array(pixel_data)
    if raster_raw.dtype.names:
        raster = raster_raw[raster_raw.dtype.names[0]].astype(np.float32)
    else:
        raster = raster_raw.astype(np.float32)

    logger.info(f"Export: {width}x{height}, shape={raster.shape}, "
                f"valid={np.sum(np.isfinite(raster))}/{raster.size}")
    return raster


def fetch_ndvi_for_date(
    geometry: list,
    target_date_str: str,
    window_days: int = 30,
    cloud_threshold: int = 30,
    max_retries: int = 2
) -> Optional[Dict[str, Any]]:
    """
    Fetch NDVI median composite for target date from Sentinel-2.

    Uses a median composite of all cloud-masked images within the time window.
    This eliminates Sentinel-2 detector striping artifacts that appear when
    subtracting single images from different orbital passes.

    geometry: [[lon, lat], [lon, lat], ...] polygon (GeoJSON format)
    target_date_str: "YYYY-MM-DD"
    window_days: search window = target_date +/- window_days (default 30)
    cloud_threshold: max acceptable scene cloud percentage (default 30)
    """
    if max_retries <= 0:
        logger.error("Max retries exhausted")
        return None

    # Check cache first
    cached = get_cached_ndvi(geometry, target_date_str)
    if cached is not None:
        return {"raster": cached, "cache_hit": True, "date_used": target_date_str}

    try:
        ee_geometry = ee.Geometry.Polygon(geometry)

        target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
        start_date = (target_date - timedelta(days=window_days)).strftime("%Y-%m-%d")
        end_date = (target_date + timedelta(days=window_days)).strftime("%Y-%m-%d")

        logger.info(f"Building NDVI composite: {start_date} to {end_date}, "
                     f"window={window_days}d, cloud<{cloud_threshold}%")

        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(ee_geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold)))

        count = collection.size().getInfo()
        if count == 0:
            logger.warning(f"No imagery found ({window_days}d, {cloud_threshold}%)")
            if window_days < 60:
                return fetch_ndvi_for_date(geometry, target_date_str,
                    window_days=60, cloud_threshold=40, max_retries=max_retries - 1)
            return None

        logger.info(f"Found {count} images, building median composite...")

        # Strategy: mosaic tiles from same date FIRST (fills tile gaps),
        # then median across dates (eliminates detector striping).
        # Without this, farms spanning two Sentinel-2 tiles only get data
        # from the tile overlap zone.
        masked = collection.map(_mask_s2_clouds)

        # Get unique acquisition dates
        unique_dates = (masked
            .aggregate_array('system:time_start')
            .map(lambda t: ee.Date(t).format('YYYY-MM-dd'))
            .distinct())

        def _daily_mosaic_ndvi(date_str):
            """Mosaic all tiles from one date, then compute NDVI."""
            d = ee.Date(date_str)
            daily = masked.filterDate(d, d.advance(1, 'day'))
            # .mosaic() merges adjacent tiles into one seamless image
            return (daily.mosaic()
                .normalizedDifference(['B8', 'B4']).float()
                .set('system:time_start', d.millis()))

        daily_mosaics = ee.ImageCollection(unique_dates.map(_daily_mosaic_ndvi))
        n_dates = unique_dates.size().getInfo()
        logger.info(f"Mosaicked into {n_dates} daily composites, computing median...")

        # Median across daily mosaics — each covers full geometry, no tile gaps
        ndvi_composite = daily_mosaics.median()

        # Export to numpy
        raster = _export_raster(ndvi_composite, ee_geometry)

        valid_count = np.sum(np.isfinite(raster))
        if valid_count == 0:
            logger.warning("Composite has no valid pixels")
            if cloud_threshold < 50:
                return fetch_ndvi_for_date(geometry, target_date_str,
                    window_days=60, cloud_threshold=50, max_retries=max_retries - 1)
            return None

        logger.info(f"NDVI composite: min={np.nanmin(raster):.4f}, "
                     f"max={np.nanmax(raster):.4f}, valid={valid_count/raster.size*100:.1f}%")

        set_cached_ndvi(geometry, target_date_str, raster)

        return {"raster": raster, "cache_hit": False, "date_used": target_date_str}

    except Exception as e:
        logger.error(f"Error fetching NDVI: {e}")
        return None

def fetch_all_ndvi(geometry: list, dates: list) -> Dict[str, Any]:
    """
    Fetch NDVI for multiple dates.
    
    dates: ["YYYY-MM-DD", "YYYY-MM-DD", "YYYY-MM-DD"]
    
    Returns:
        {
            "current": {...raster, cache_hit, date_used...},
            "year_1": {...},
            "year_2": {...},
            "cache_hits": [bool, bool, bool]
        }
    """
    logger.info("=" * 60)
    logger.info("Fetching NDVI rasters in parallel (progressive retry per date)...")
    logger.info("Strategy: 7d window (20% cloud) → 15d window → 30d window → 40% cloud threshold")

    def _fetch_one(idx_date):
        i, date_str = idx_date
        logger.info(f"[{i+1}/{len(dates)}] Starting fetch for {date_str}...")
        try:
            result = fetch_ndvi_for_date(geometry, date_str)
            if result:
                logger.info(f"✅ Success for {date_str}")
            else:
                logger.warning(f"❌ Failed for {date_str}")
            return result
        except Exception as e:
            logger.error(f"❌ Exception fetching {date_str}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=len(dates)) as executor:
        results = list(executor.map(_fetch_one, list(enumerate(dates))))
    
    available = [r for r in results if r is not None]
    
    if len(available) == 0:
        raise ServiceError("No cloud-free imagery found for any requested dates")
    elif len(available) == 1:
        logger.warning(f"⚠️ Only 1 year of imagery available (need 2 for best results). Proceeding with limited historical comparison.")
    
    logger.info(f"\n✅ NDVI fetch complete: {len(available)}/3 years available")
    logger.info("=" * 60)
    
    return {
        "current": results[0],
        "year_1": results[1],
        "year_2": results[2],
        "cache_hits": [r['cache_hit'] if r else False for r in results]
    }

def fetch_ndwi_for_date(
    geometry: list,
    target_date_str: str,
    sat_image=None
) -> Optional[Dict[str, Any]]:
    """
    Fetch NDWI (Normalized Difference Water Index) for target date.
    
    Args:
        geometry: [[lon, lat], [lon, lat], ...] polygon (GeoJSON format)
        target_date_str: "YYYY-MM-DD"
        sat_image: Optional pre-fetched satellite image. If None, will fetch from GEE.
    
    Returns:
        {
            "raster": numpy array,
            "cache_hit": bool
        }
    """
    # Check cache first
    cached = get_cached_index(geometry, target_date_str, "NDWI")
    if cached is not None:
        return {
            "raster": cached,
            "cache_hit": True
        }
    
    try:
        # If no pre-fetched image, get one from GEE
        if sat_image is None:
            ee_geometry = ee.Geometry.Polygon(geometry)
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
            start_date = (target_date - timedelta(days=7)).strftime("%Y-%m-%d")
            end_date = (target_date + timedelta(days=7)).strftime("%Y-%m-%d")
            
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(ee_geometry) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            
            if collection.size().getInfo() == 0:
                logger.warning("No imagery found for NDWI")
                return None
            
            sat_image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
        
        # Calculate NDWI: (B8 - B11) / (B8 + B11)
        ndwi = sat_image.normalizedDifference(['B8', 'B11']).float()
        
        # Clip to geometry
        ee_geometry = ee.Geometry.Polygon(geometry)
        ndwi_clipped = ndwi.clip(ee_geometry)
        
        # Export as numpy array
        logger.info("Exporting NDWI raster from GEE...")
        bounds = ee_geometry.bounds().getInfo()
        coords = bounds['coordinates'][0]
        
        min_lon = min(coord[0] for coord in coords)
        max_lon = max(coord[0] for coord in coords)
        min_lat = min(coord[1] for coord in coords)
        max_lat = max(coord[1] for coord in coords)
        
        avg_lat = (min_lat + max_lat) / 2
        meters_per_degree_lon = 111319 * np.cos(np.radians(avg_lat))
        meters_per_degree_lat = 111139
        
        scale = 10
        width = max(50, int((max_lon - min_lon) * meters_per_degree_lon / scale))
        height = max(50, int((max_lat - min_lat) * meters_per_degree_lat / scale))
        width = min(width, 256)
        height = min(height, 256)
        
        scale_x = (max_lon - min_lon) / width
        scale_y = (max_lat - min_lat) / height
        
        request = {
            'expression': ndwi_clipped,
            'fileFormat': 'NUMPY_NDARRAY',
            'grid': {
                'dimensions': {'width': width, 'height': height},
                'affineTransform': {
                    'scaleX': scale_x,
                    'shearX': 0,
                    'translateX': min_lon,
                    'shearY': 0,
                    'scaleY': -scale_y,
                    'translateY': max_lat
                },
                'crsCode': 'EPSG:4326'
            }
        }
        
        raster_raw = np.array(ee.data.computePixels(request))
        if raster_raw.dtype.names:
            raster = raster_raw[raster_raw.dtype.names[0]].astype(np.float32)
        else:
            raster = raster_raw.astype(np.float32)
        logger.info(f"✅ NDWI export successful: min={np.nanmin(raster):.4f}, max={np.nanmax(raster):.4f}")
        
        # Cache it
        set_cached_index(geometry, target_date_str, "NDWI", raster)
        
        return {
            "raster": raster,
            "cache_hit": False
        }
        
    except Exception as e:
        logger.error(f"NDWI fetch error: {e}")
        return None

def fetch_ndre_for_date(
    geometry: list,
    target_date_str: str,
    sat_image=None
) -> Optional[Dict[str, Any]]:
    """
    Fetch NDRE (Normalized Difference Red Edge Index) for target date.
    
    Args:
        geometry: [[lon, lat], [lon, lat], ...] polygon (GeoJSON format)
        target_date_str: "YYYY-MM-DD"
        sat_image: Optional pre-fetched satellite image. If None, will fetch from GEE.
    
    Returns:
        {
            "raster": numpy array,
            "cache_hit": bool
        }
    """
    # Check cache first
    cached = get_cached_index(geometry, target_date_str, "NDRE")
    if cached is not None:
        return {
            "raster": cached,
            "cache_hit": True
        }
    
    try:
        # If no pre-fetched image, get one from GEE
        if sat_image is None:
            ee_geometry = ee.Geometry.Polygon(geometry)
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
            start_date = (target_date - timedelta(days=7)).strftime("%Y-%m-%d")
            end_date = (target_date + timedelta(days=7)).strftime("%Y-%m-%d")
            
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterBounds(ee_geometry) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            
            if collection.size().getInfo() == 0:
                logger.warning("No imagery found for NDRE")
                return None
            
            sat_image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
        
        # Calculate NDRE: (B8 - B5) / (B8 + B5)
        ndre = sat_image.normalizedDifference(['B8', 'B5']).float()
        
        # Clip to geometry
        ee_geometry = ee.Geometry.Polygon(geometry)
        ndre_clipped = ndre.clip(ee_geometry)
        
        # Export as numpy array
        logger.info("Exporting NDRE raster from GEE...")
        bounds = ee_geometry.bounds().getInfo()
        coords = bounds['coordinates'][0]
        
        min_lon = min(coord[0] for coord in coords)
        max_lon = max(coord[0] for coord in coords)
        min_lat = min(coord[1] for coord in coords)
        max_lat = max(coord[1] for coord in coords)
        
        avg_lat = (min_lat + max_lat) / 2
        meters_per_degree_lon = 111319 * np.cos(np.radians(avg_lat))
        meters_per_degree_lat = 111139
        
        scale = 10
        width = max(50, int((max_lon - min_lon) * meters_per_degree_lon / scale))
        height = max(50, int((max_lat - min_lat) * meters_per_degree_lat / scale))
        width = min(width, 256)
        height = min(height, 256)
        
        scale_x = (max_lon - min_lon) / width
        scale_y = (max_lat - min_lat) / height
        
        request = {
            'expression': ndre_clipped,
            'fileFormat': 'NUMPY_NDARRAY',
            'grid': {
                'dimensions': {'width': width, 'height': height},
                'affineTransform': {
                    'scaleX': scale_x,
                    'shearX': 0,
                    'translateX': min_lon,
                    'shearY': 0,
                    'scaleY': -scale_y,
                    'translateY': max_lat
                },
                'crsCode': 'EPSG:4326'
            }
        }
        
        raster_raw = np.array(ee.data.computePixels(request))
        if raster_raw.dtype.names:
            raster = raster_raw[raster_raw.dtype.names[0]].astype(np.float32)
        else:
            raster = raster_raw.astype(np.float32)
        logger.info(f"✅ NDRE export successful: min={np.nanmin(raster):.4f}, max={np.nanmax(raster):.4f}")
        
        # Cache it
        set_cached_index(geometry, target_date_str, "NDRE", raster)
        
        return {
            "raster": raster,
            "cache_hit": False
        }
        
    except Exception as e:
        logger.error(f"NDRE fetch error: {e}")
        return None
