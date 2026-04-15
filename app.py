# app.py

from dotenv import load_dotenv
load_dotenv()

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Tuple, Optional
from datetime import datetime, date
import numpy as np
import io
import base64
import logging
import merged_processor
import utils.crop_health_gee
from utils.crop_health_analyzer import analyze_crop_health

app = FastAPI(
    title="Crop Yield Prediction API",
    description="AI-powered crop yield prediction with satellite imagery and soil sensor data",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods including OPTIONS
    allow_headers=["*"],  # Allows all headers
)

# --- Configure logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gee_status = None
gee_error = None

# --- Startup event to initialize GEE ---
@app.on_event("startup")
async def startup_event():
    """Initialize Google Earth Engine when server starts"""
    global gee_status, gee_error
    try:
        logger.info("🚀 Server startup - Initializing Google Earth Engine (this may take a moment)...")
        start_time = datetime.now()
        
        gee_status = merged_processor.initialize_earth_engine()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        if gee_status:
            logger.info(f"✅ Google Earth Engine ready in {elapsed:.2f}s")
        else:
            gee_error = "GEE initialization returned False - check credentials"
            logger.error(f"❌ GEE initialization failed: {gee_error}")
    except Exception as e:
        gee_error = str(e)
        logger.error(f"❌ Error initializing GEE at startup: {e}")
        logger.warning("⚠️  API will still start but GEE-dependent endpoints will fail")
        logger.warning("💡 Check system clock sync: Right-click Windows clock → 'Adjust date/time' → 'Sync now'")

# Pydantic models

def get_corresponding_date():
    """Fetch corresponding date based on current date"""
    current = datetime.now()
    # Assuming corresponding is current year - 3, October 1st
    year = current.year - 3
    return f"{year}-10-01"

def sanitize_json_response(obj):
    """Recursively sanitize response to remove inf/-inf/-nan values that break JSON serialization"""
    if isinstance(obj, dict):
        return {k: sanitize_json_response(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json_response(item) for item in obj]
    elif isinstance(obj, float):
        if np.isinf(obj) or np.isnan(obj):
            return None  # Replace inf/-inf/nan with null
        return obj
    elif isinstance(obj, np.ndarray):
        return sanitize_json_response(obj.tolist())
    else:
        return obj

class HeatmapRequest(BaseModel):
    coordinates: List[List[float]]  # List of [longitude, latitude] points
    t1: float = 3.0  # Threshold for low yield
    t2: float = 4.5  # Threshold for high yield
    cultivation_date: Optional[date] = None  # Optional: crop cultivation start date
    harvest_date: Optional[date] = None  # Optional: crop harvest date
    state: Optional[str] = None  # Optional: state name for news query
    crop: Optional[str] = None  # Optional: crop name for news query

class CropHealthRequest(BaseModel):
    geometry: List[List[float]]  # [[lon, lat], [lon, lat], ...] (GeoJSON format)
    cultivation_date: date
    harvest_date: date

# Health check endpoints
@app.get("/")
@app.get("/health")
async def health_check():
    """Health check and root endpoint for monitoring"""
    return {
        "status": "healthy" if gee_status else "unhealthy",
        "message": "🌾 Crop Yield Prediction API is running",
        "components": {
            "google_earth_engine": "connected" if gee_status else f"error: {gee_error}"
        },
        "version": "1.0.0",
        "endpoints": {
            "/generate_heatmap_lite": "Generate color-coded heatmap (CSV-based yield, no ML)",
            "/export_arrays": "Export NDVI and sensor arrays as .npz file",
            "/crop_health/analyze": "Analyze crop health using multi-year satellite imagery"
        }
    }

@app.get("/debug/env")
async def debug_env():
    """Report whether required env vars are loaded (no values exposed)."""
    import os
    keys = ["NEWSAPI_KEY", "GOVDATA_API_KEY"]
    return {
        k: {
            "set": bool(os.getenv(k)),
            "length": len(os.getenv(k) or ""),
        }
        for k in keys
    }


@app.post("/generate_heatmap_lite")
async def generate_heatmap_lite(request: HeatmapRequest):
    """
    Lightweight heatmap generation without ML model inference or sensor data.
    Uses historical yield from CSV as predicted yield.
    Same request/response format as /generate_heatmap.
    """
    # --- Check GEE initialized ---
    if not gee_status:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Google Earth Engine: {gee_error}")

    try:
        # --- Get corresponding date ---
        date_str = get_corresponding_date()
        logging.info(f"[lite] Using date: {date_str}")
        logger.info(f"[lite] Request fields received: state={request.state!r}, crop={request.crop!r}, cultivation_date={request.cultivation_date}, harvest_date={request.harvest_date}")

        # --- Generate NDVI data only (skip sensor) ---
        geojson_dict = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [request.coordinates]
            }
        }

        ndvi_data, _, sat_image = merged_processor.generate_ndvi_and_sensor_npy(
            geojson_dict, date_str, skip_sensor=True
        )

        if ndvi_data is None:
            raise HTTPException(status_code=400, detail="Failed to generate NDVI data from coordinates")

        # --- Get location and old yield from CSV (no ML model) ---
        centroid_lat, centroid_lon = merged_processor.get_centroid_coordinates(geojson_dict)
        yield_df = merged_processor.load_yield_data()

        location_info = {
            "district": "unknown",
            "state": None,
            "coordinates": {"latitude": None, "longitude": None},
            "complete_address": "Location not available"
        }
        old_yield = 1.0
        predicted_yield = 1.0
        yield_ratio = 1.0
        growth_percentage = 0.0

        if centroid_lat is not None and centroid_lon is not None:
            district, detected_state, complete_location = merged_processor.get_district_and_location_sync(centroid_lat, centroid_lon)
            old_yield = merged_processor.get_old_yield_for_district(district, yield_df)
            predicted_yield = old_yield  # No model — use old yield as predicted

            # Fallback: if frontend didn't send state, use the one detected from coordinates.
            if not request.state and detected_state:
                logger.info(f"[lite] state not in request; using detected state={detected_state!r}")
                request.state = detected_state

            location_info = {
                "district": district,
                "state": detected_state,
                "coordinates": {"latitude": centroid_lat, "longitude": centroid_lon},
                "complete_address": complete_location
            }

            final_ndvi_data, yield_ratio = merged_processor.compare_yields_and_adjust_ndvi(
                ndvi_data, predicted_yield, old_yield
            )

            growth_percentage = ((predicted_yield - old_yield) / old_yield) * 100 if old_yield > 0 else 0.0
            logger.info(f"[lite] District: {district}, Old yield: {old_yield}, Predicted yield: {predicted_yield}")
        else:
            final_ndvi_data = ndvi_data
            logger.warning("[lite] Could not get district information, using original NDVI data")

        # --- Generate separate heatmap masks ---
        red_mask, yellow_mask, green_mask, pixel_counts = merged_processor.create_separate_yield_masks(
            final_ndvi_data, predicted_yield, request.t1, request.t2
        )

        if red_mask is None or yellow_mask is None or green_mask is None:
            raise HTTPException(status_code=500, detail="Failed to generate heatmap masks")

        # --- Apply pixel-based multiplier to yields ---
        green_pixels = pixel_counts.get("green", 0)
        yellow_pixels = pixel_counts.get("yellow", 0)
        red_pixels = pixel_counts.get("red", 0)
        valid_pixels = pixel_counts.get("valid", green_pixels + yellow_pixels + red_pixels)
        multiplier = (green_pixels + 0.5 * yellow_pixels) / valid_pixels if valid_pixels > 0 else 0

        old_yield = old_yield * multiplier
        predicted_yield = predicted_yield * multiplier

        logger.info(f"[lite] Applied multiplier: {multiplier:.4f} (green={green_pixels}, yellow={yellow_pixels})")

        # --- Convert each mask to PNG base64 ---
        import PIL.Image

        def mask_to_base64(mask_array):
            img = PIL.Image.fromarray(mask_array, "RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            png_bytes = buf.read()
            return base64.b64encode(png_bytes).decode('ascii')

        red_base64 = mask_to_base64(red_mask)
        yellow_base64 = mask_to_base64(yellow_mask)
        green_base64 = mask_to_base64(green_mask)

        # --- Run NDWI, NDRE, crop health, news, and rates fetches in parallel ---
        from concurrent.futures import ThreadPoolExecutor
        from utils.news_fetcher import fetch_agri_news
        from utils.rates_fetcher import fetch_govdata_last_n_days, fetch_agmarknet

        def _ndwi_job():
            logger.info("[lite] Calculating NDWI...")
            return utils.crop_health_gee.fetch_ndwi_for_date(request.coordinates, date_str, sat_image)

        def _ndre_job():
            logger.info("[lite] Calculating NDRE...")
            return utils.crop_health_gee.fetch_ndre_for_date(request.coordinates, date_str, sat_image)

        def _anomaly_job():
            if not (request.cultivation_date and request.harvest_date):
                return None
            try:
                logger.info("[lite] Running crop health analysis...")
                data = analyze_crop_health(
                    geometry=request.coordinates,
                    cultivation_date=request.cultivation_date,
                    harvest_date=request.harvest_date
                )
                logger.info("[lite] ✅ Crop health analysis complete")
                return data
            except Exception as e:
                logger.warning(f"[lite] ⚠️  Crop health analysis failed (non-blocking): {e}")
                return None

        def _news_job():
            try:
                return fetch_agri_news(state=request.state, crop=request.crop)
            except Exception as e:
                logger.warning(f"[lite] ⚠️  News fetch failed (non-blocking): {e}")
                return []

        def _govdata_job():
            try:
                if not request.state:
                    return []
                return fetch_govdata_last_n_days(state=request.state, crop=request.crop, days=7)
            except Exception as e:
                logger.warning(f"[lite] ⚠️  govdata fetch failed (non-blocking): {e}")
                return []

        def _agmarknet_job():
            try:
                return fetch_agmarknet(state=request.state, crop=request.crop)
            except Exception as e:
                logger.warning(f"[lite] ⚠️  agmarknet fetch failed (non-blocking): {e}")
                return None

        with ThreadPoolExecutor(max_workers=6) as pool:
            f_ndwi = pool.submit(_ndwi_job)
            f_ndre = pool.submit(_ndre_job)
            f_anom = pool.submit(_anomaly_job)
            f_news = pool.submit(_news_job)
            f_govdata = pool.submit(_govdata_job)
            f_agmarknet = pool.submit(_agmarknet_job)
            ndwi_result = f_ndwi.result()
            ndre_result = f_ndre.result()
            anomaly_data = f_anom.result()
            news_articles = f_news.result()
            govdata_rates = f_govdata.result()
            agmarknet_rates = f_agmarknet.result()

        ndwi_masks_response = {}
        ndwi_pixel_counts = {}
        if ndwi_result is not None:
            ndwi_values = ndwi_result["raster"].astype(np.float32)
            ndwi_brown_mask, ndwi_yellow_mask, ndwi_light_blue_mask, ndwi_pixel_counts = merged_processor.create_separate_ndwi_masks(
                ndwi_values
            )
            if ndwi_brown_mask is not None:
                ndwi_masks_response = {
                    "brown_mask_base64": mask_to_base64(ndwi_brown_mask),
                    "yellow_mask_base64": mask_to_base64(ndwi_yellow_mask),
                    "light_blue_mask_base64": mask_to_base64(ndwi_light_blue_mask)
                }

        ndre_masks_response = {}
        ndre_pixel_counts = {}
        if ndre_result is not None:
            ndre_values = ndre_result["raster"].astype(np.float32)
            ndre_purple_mask, ndre_pink_mask, ndre_light_green_mask, ndre_dark_green_mask, ndre_pixel_counts = merged_processor.create_separate_ndre_masks(
                ndre_values
            )
            if ndre_purple_mask is not None:
                ndre_masks_response = {
                    "purple_mask_base64": mask_to_base64(ndre_purple_mask),
                    "pink_mask_base64": mask_to_base64(ndre_pink_mask),
                    "light_green_mask_base64": mask_to_base64(ndre_light_green_mask),
                    "dark_green_mask_base64": mask_to_base64(ndre_dark_green_mask)
                }

        trimmed_pixel_counts = {
            k: v for k, v in pixel_counts.items()
            if k not in ("transparent", "total_field")
        }

        response = {
            "predicted_yield": predicted_yield,
            "old_yield": old_yield,
            "growth": {
                "percentage": growth_percentage
            },
            "location": location_info,
            "date_used": date_str,
            "masks": {
                "red_mask_base64": red_base64,
                "yellow_mask_base64": yellow_base64,
                "green_mask_base64": green_base64
            },
            "ndwi-masks": ndwi_masks_response,
            "ndre-masks": ndre_masks_response,
            "pixel_counts": trimmed_pixel_counts,
            "ndwi_pixel_counts": ndwi_pixel_counts,
            "ndre_pixel_counts": ndre_pixel_counts,
            "news": news_articles or [],
            "rate": {
                "govdata": govdata_rates or [],
                "agmarknet": agmarknet_rates,
            }
        }

        if anomaly_data:
            response["anomaly"] = anomaly_data

        response = sanitize_json_response(response)

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"[lite] Heatmap generation error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {str(e)}")


@app.post("/export_arrays")
async def export_arrays(request: HeatmapRequest):
    """
    Utility endpoint: generate NDVI and sensor arrays for the provided coordinates
    and return them as a .npz file in-memory (no disk writes).
    """
    # --- Check GEE initialized ---
    if not gee_status:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Google Earth Engine: {gee_error}")

    try:

        date_str = get_corresponding_date()

        geojson_dict = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [request.coordinates]
            }
        }

        ndvi_data, sensor_data, _ = merged_processor.generate_ndvi_and_sensor_npy(geojson_dict, date_str)

        if ndvi_data is None or sensor_data is None:
            raise HTTPException(status_code=400, detail="Failed to generate arrays from coordinates")

        # Pack to in-memory .npz
        buf = io.BytesIO()
        np.savez(buf, ndvi=ndvi_data, sensor=sensor_data)
        buf.seek(0)

        return StreamingResponse(buf, media_type="application/octet-stream",
                                 headers={"Content-Disposition": "attachment; filename=arrays.npz"})

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Export arrays error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Export arrays failed: {str(e)}")


@app.post("/crop_health/analyze")
async def analyze_crop_health_api(request: CropHealthRequest):
    """
    Analyze crop health for given geometry and dates.
    
    Uses satellite imagery (NDVI) to assess crop health with:
    - Reference date determination based on cultivation and harvest dates
    - Multi-year comparison (current year vs. previous 2 years)
    - Anomaly detection and pixel classification (healthy/stressed)
    - Summary statistics and caching for performance
    
    Args:
        geometry: List of [longitude, latitude] pairs forming a polygon (GeoJSON format)
        cultivation_date: Crop cultivation start date (YYYY-MM-DD)
        harvest_date: Crop harvest date (YYYY-MM-DD)
    
    Returns:
        JSON with reference date, analyzed dates, tile URLs, summary stats, and cache info
    """
    try:
        if not gee_status:
            raise HTTPException(
                status_code=500,
                detail=f"Google Earth Engine not initialized: {gee_error}"
            )
        
        logger.info(f"Analyzing crop health for geometry with {len(request.geometry)} vertices")
        
        result = analyze_crop_health(
            geometry=request.geometry,
            cultivation_date=request.cultivation_date,
            harvest_date=request.harvest_date
        )
        
        return JSONResponse(content=result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Crop health analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Crop health analysis failed: {str(e)}")