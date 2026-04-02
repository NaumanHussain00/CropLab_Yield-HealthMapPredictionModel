import json
import os
from datetime import datetime
from typing import Optional, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Determine cache directory - use absolute path for reliability
_cache_env = os.getenv("CACHE_DIR", "./cache")
if os.path.isabs(_cache_env):
    CACHE_DIR = _cache_env
else:
    # Convert relative path to absolute path
    CACHE_DIR = os.path.abspath(_cache_env)

USE_REDIS = os.getenv("USE_REDIS", "false").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Try to import redis, fallback to file-based cache
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

redis_client = None
if USE_REDIS and REDIS_AVAILABLE:
    try:
        redis_client = redis.from_url(REDIS_URL)
        redis_client.ping()
        logger.info("✅ Redis cache initialized")
    except Exception as e:
        logger.warning(f"⚠️ Redis not available ({e}), falling back to file cache")
        redis_client = None

# Ensure cache directory exists
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    logger.debug(f"Cache directory: {CACHE_DIR}")
except Exception as e:
    logger.warning(f"⚠️ Could not create cache directory {CACHE_DIR}: {e}")

def get_cache_key(geometry: list, date_str: str) -> str:
    """
    Generate cache key from geometry polygon and date.
    geometry: [[lat, lon], [lat, lon], ...] polygon
    date_str: "YYYY-MM-DD"
    """
    # Compute centroid
    lats = [point[0] for point in geometry if len(point) >= 2]
    lons = [point[1] for point in geometry if len(point) >= 2]
    
    if not lats or not lons:
        raise ValueError("Invalid geometry: must contain lat/lon pairs")
    
    centroid_lat = sum(lats) / len(lats)
    centroid_lon = sum(lons) / len(lons)
    
    # Round to 4 decimal places
    lat_rounded = round(centroid_lat, 4)
    lon_rounded = round(centroid_lon, 4)
    
    key = f"{lat_rounded}:{lon_rounded}:{date_str}"
    logger.debug(f"Cache key: {key}")
    return key

def get_cache_filename(geometry: list, date_str: str) -> str:
    """
    Generate safe cache filename (colons replaced with underscores for Windows compatibility).
    """
    key = get_cache_key(geometry, date_str)
    # Replace colons with underscores for Windows filesystem compatibility
    safe_key = key.replace(":", "_")
    return safe_key

def get_cached_ndvi(geometry: list, date_str: str) -> Optional[np.ndarray]:
    """Retrieve cached NDVI raster (uses same naming as get_cached_index for cross-API caching)"""
    key = get_cache_key(geometry, date_str)
    indexed_key = f"{key}_NDVI"  # Append _NDVI for consistent naming
    
    # Try Redis first
    if redis_client:
        try:
            cached = redis_client.get(indexed_key)
            shape_cached = redis_client.get(f"{indexed_key}_shape")
            if cached and shape_cached:
                shape = tuple(json.loads(shape_cached))
                logger.info(f"✅ Cache hit (Redis): {indexed_key}, shape={shape}")
                return np.frombuffer(cached, dtype=np.float32).reshape(shape)
            elif cached:
                logger.warning(f"⚠️ Redis cache hit but no shape info: {indexed_key}")
        except Exception as e:
            logger.warning(f"Redis read error: {e}")

    # Fall back to file cache
    safe_key = get_cache_filename(geometry, date_str)
    cache_file = os.path.join(CACHE_DIR, f"{safe_key}_NDVI.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                age = (datetime.now() - datetime.fromisoformat(data['timestamp'])).total_seconds()
                if age < 86400:  # 24 hours TTL
                    logger.info(f"✅ Cache hit (File): {indexed_key}")
                    return np.array(data['raster'], dtype=np.float32)
                else:
                    logger.info(f"⏰ Cache expired: {indexed_key}")
                    os.remove(cache_file)
        except Exception as e:
            logger.warning(f"File cache read error: {e}")

    return None

def set_cached_ndvi(geometry: list, date_str: str, raster: np.ndarray) -> None:
    """Store NDVI raster in cache (uses same naming as set_cached_index for cross-API caching)"""
    key = get_cache_key(geometry, date_str)
    indexed_key = f"{key}_NDVI"  # Append _NDVI for consistent naming
    
    # Try Redis first
    if redis_client:
        try:
            raster_arr = np.asarray(raster, dtype=np.float32)
            raster_bytes = raster_arr.tobytes()
            redis_client.setex(indexed_key, 1800, raster_bytes)
            redis_client.setex(f"{indexed_key}_shape", 1800, json.dumps(list(raster_arr.shape)))
            logger.debug(f"Cached to Redis: {indexed_key}, shape={raster_arr.shape}")
            return
        except Exception as e:
            logger.warning(f"Redis write error: {e}")

    # Fall back to file cache
    try:
        safe_key = get_cache_filename(geometry, date_str)
        cache_file = os.path.join(CACHE_DIR, f"{safe_key}_NDVI.json")
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "raster": np.asarray(raster, dtype=np.float32).tolist()
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        logger.debug(f"Cached to file: {indexed_key}")
    except Exception as e:
        logger.warning(f"File cache write error: {e}")

def get_cached_index(geometry: list, date_str: str, index_name: str) -> Optional[np.ndarray]:
    """Retrieve cached spectral index (NDWI, NDRE, etc)"""
    key = f"{get_cache_key(geometry, date_str)}_{index_name}"
    
    # Try Redis first
    if redis_client:
        try:
            cached = redis_client.get(key)
            shape_cached = redis_client.get(f"{key}_shape")
            if cached and shape_cached:
                shape = tuple(json.loads(shape_cached))
                logger.info(f"✅ Cache hit (Redis): {key}, shape={shape}")
                return np.frombuffer(cached, dtype=np.float32).reshape(shape)
            elif cached:
                logger.warning(f"⚠️ Redis cache hit but no shape info: {key}")
        except Exception as e:
            logger.warning(f"Redis read error: {e}")
    
    # Fall back to file cache
    safe_key = get_cache_filename(geometry, date_str)
    cache_file = os.path.join(CACHE_DIR, f"{safe_key}_{index_name}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                age = (datetime.now() - datetime.fromisoformat(data['timestamp'])).total_seconds()
                if age < 1800:  # 30 minutes TTL
                    logger.info(f"✅ Cache hit (File): {key}")
                    return np.array(data['raster'], dtype=np.float32)
                else:
                    logger.info(f"⏰ Cache expired: {key}")
                    os.remove(cache_file)
        except Exception as e:
            logger.warning(f"File cache read error: {e}")
    
    return None

def set_cached_index(geometry: list, date_str: str, index_name: str, raster: np.ndarray) -> None:
    """Store spectral index raster in cache"""
    key = f"{get_cache_key(geometry, date_str)}_{index_name}"
    
    # Try Redis first
    if redis_client:
        try:
            raster_arr = np.asarray(raster, dtype=np.float32)
            redis_client.setex(key, 1800, raster_arr.tobytes())
            redis_client.setex(f"{key}_shape", 1800, json.dumps(list(raster_arr.shape)))
            logger.debug(f"Cached to Redis: {key}, shape={raster_arr.shape}")
            return
        except Exception as e:
            logger.warning(f"Redis write error: {e}")
    
    # Fall back to file cache
    try:
        safe_key = get_cache_filename(geometry, date_str)
        cache_file = os.path.join(CACHE_DIR, f"{safe_key}_{index_name}.json")
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "raster": np.asarray(raster, dtype=np.float32).tolist()
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        logger.debug(f"Cached to file: {key}")
    except Exception as e:
        logger.warning(f"File cache write error: {e}")
