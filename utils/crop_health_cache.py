import json
import os
from datetime import datetime
from typing import Optional, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
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
os.makedirs(CACHE_DIR, exist_ok=True)

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
    """Retrieve cached NDVI raster"""
    key = get_cache_key(geometry, date_str)
    
    # Try Redis first
    if redis_client:
        try:
            cached = redis_client.get(key)
            if cached:
                logger.info(f"✅ Cache hit (Redis): {key}")
                return np.frombuffer(cached, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Redis read error: {e}")
    
    # Fall back to file cache
    safe_key = get_cache_filename(geometry, date_str)
    cache_file = os.path.join(CACHE_DIR, f"{safe_key}.json")
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

def set_cached_ndvi(geometry: list, date_str: str, raster: np.ndarray) -> None:
    """Store NDVI raster in cache"""
    key = get_cache_key(geometry, date_str)
    
    # Try Redis first
    if redis_client:
        try:
            raster_bytes = np.asarray(raster, dtype=np.float32).tobytes()
            redis_client.setex(key, 1800, raster_bytes)  # 30 min TTL
            logger.debug(f"Cached to Redis: {key}")
            return
        except Exception as e:
            logger.warning(f"Redis write error: {e}")
    
    # Fall back to file cache
    try:
        safe_key = get_cache_filename(geometry, date_str)
        cache_file = os.path.join(CACHE_DIR, f"{safe_key}.json")
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "raster": np.asarray(raster, dtype=np.float32).tolist()
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        logger.debug(f"Cached to file: {key}")
    except Exception as e:
        logger.warning(f"File cache write error: {e}")
