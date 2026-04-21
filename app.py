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
import os
import json
import requests
import re
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


def _safe_pct(numerator, denominator):
    if denominator is None or denominator <= 0:
        return 0.0
    return round((float(numerator) / float(denominator)) * 100.0, 2)


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _derive_priority(risk_score):
    if risk_score >= 35:
        return "High"
    if risk_score >= 15:
        return "Medium"
    return "Low"


def _derive_overall_health(risk_score):
    if risk_score > 50:
        return "Critical"
    if risk_score > 35:
        return "Poor"
    if risk_score > 20:
        return "Moderate"
    if risk_score > 10:
        return "Good"
    return "Excellent"


def _build_fallback_ai_analysis(metrics):
    ndvi = metrics.get("ndvi", {})
    ndwi = metrics.get("ndwi", {})
    ndre = metrics.get("ndre", {})
    anomaly = metrics.get("anomaly", {})

    ndvi_red = float(ndvi.get("red_pct", 0.0))
    ndvi_yellow = float(ndvi.get("yellow_pct", 0.0))
    ndwi_low = float(ndwi.get("low_water_area_pct", 0.0))
    ndre_low = float(ndre.get("low_nitrogen_area_pct", 0.0))
    anomaly_neg = float(anomaly.get("negative_anomaly_pct", 0.0))

    risk_score = round(
        0.35 * ndvi_red
        + 0.20 * ndvi_yellow
        + 0.20 * ndwi_low
        + 0.15 * ndre_low
        + 0.10 * anomaly_neg,
        2,
    )

    overall_health = _derive_overall_health(risk_score)
    priority = _derive_priority(risk_score)

    issues = []
    if ndwi_low > 0:
        issues.append({
            "id": "water_stress",
            "name": "Water stress",
            "affected_area_pct": round(ndwi_low, 2),
            "priority": "High" if ndwi_low >= 20 else "Medium",
        })
    if ndre_low > 0:
        issues.append({
            "id": "nitrogen_deficiency",
            "name": "Nitrogen deficiency",
            "affected_area_pct": round(ndre_low, 2),
            "priority": "High" if ndre_low >= 25 else "Medium",
        })
    if anomaly_neg > 0:
        issues.append({
            "id": "underperforming_zones",
            "name": "Underperforming zones",
            "affected_area_pct": round(anomaly_neg, 2),
            "priority": "High" if anomaly_neg >= 20 else "Medium",
        })

    if not issues:
        issues.append({
            "id": "no_major_issue_detected",
            "name": "No major issue detected",
            "affected_area_pct": 0,
            "priority": "Low",
        })

    return {
        "summary": "Crop is mostly healthy but early-stage water and nutrient stress detected in specific zones." if risk_score >= 10 else "Crop condition is stable with low stress signals.",
        "overall_health": overall_health,
        "confidence": "High",
        "risk_score": risk_score,
        "priority": priority,
        "issues": issues,
        "why_happening": [
            "Uneven irrigation or high evapotranspiration",
            "Insufficient nitrogen availability",
            "Combined stress causing anomaly",
        ],
        "immediate_actions": [
            "Fix irrigation in stressed zones within 3 days",
            "Apply nitrogen fertilizer in low-NDRE areas",
            "Monitor affected zones weekly",
        ],
        "monitor_next": [
            "NDWI trend",
            "NDRE recovery",
            "Anomaly reduction",
        ],
        "risk_if_ignored": "Stress zones may expand and reduce crop productivity.",
        "simple_advice": {
            "en": "Some zones show early water and nitrogen stress. Address these zones quickly to avoid yield loss.",
            "hi": "Kuch hisson me paani aur nitrogen ki kami hai. Un par turant dhyaan dena zaroori hai.",
        },
    }


def _normalize_ai_analysis(ai_result, metrics):
    base = _build_fallback_ai_analysis(metrics)

    if not isinstance(ai_result, dict):
        return base

    risk_score = _safe_float(ai_result.get("risk_score"), base["risk_score"])
    # Some models return 0-1 instead of 0-100.
    if 0.0 <= risk_score <= 1.0:
        risk_score *= 100.0
    risk_score = round(max(0.0, min(100.0, risk_score)), 2)

    summary = str(ai_result.get("summary") or base["summary"]).strip()
    confidence_raw = str(ai_result.get("confidence") or base["confidence"]).strip().capitalize()
    confidence = confidence_raw if confidence_raw in ("Low", "Medium", "High") else base["confidence"]

    issues_raw = ai_result.get("issues") if isinstance(ai_result.get("issues"), list) else []
    normalized_issues = []
    for index, issue in enumerate(issues_raw):
        if not isinstance(issue, dict):
            continue

        issue_id = str(issue.get("id") or f"issue_{index + 1}").strip()
        issue_name = str(issue.get("name") or issue_id.replace("_", " ").title()).strip()
        affected = _safe_float(issue.get("affected_area_pct"), 0.0)
        if 0.0 <= affected <= 1.0:
            affected *= 100.0
        affected = round(max(0.0, min(100.0, affected)), 2)

        normalized_issues.append({
            "id": issue_id,
            "name": issue_name,
            "affected_area_pct": affected,
            "priority": _derive_priority(affected),
        })

    if not normalized_issues:
        normalized_issues = base["issues"]

    def _clean_string_list(value, fallback):
        if not isinstance(value, list):
            return fallback
        out = [str(item).strip() for item in value if str(item).strip()]
        return out or fallback

    simple_advice_raw = ai_result.get("simple_advice")
    if not isinstance(simple_advice_raw, dict):
        simple_advice_raw = {}
    advice_en = str(simple_advice_raw.get("en") or base["simple_advice"]["en"]).strip()
    advice_hi = str(simple_advice_raw.get("hi") or base["simple_advice"]["hi"]).strip()

    return {
        "summary": summary,
        "overall_health": _derive_overall_health(risk_score),
        "confidence": confidence,
        "risk_score": risk_score,
        "priority": _derive_priority(risk_score),
        "issues": normalized_issues,
        "why_happening": _clean_string_list(ai_result.get("why_happening"), base["why_happening"]),
        "immediate_actions": _clean_string_list(ai_result.get("immediate_actions"), base["immediate_actions"]),
        "monitor_next": _clean_string_list(ai_result.get("monitor_next"), base["monitor_next"]),
        "risk_if_ignored": str(ai_result.get("risk_if_ignored") or base["risk_if_ignored"]).strip(),
        "simple_advice": {
            "en": advice_en,
            "hi": advice_hi,
        },
    }


def _generate_nvidia_ai_analysis(input_payload):
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        raise RuntimeError("NVIDIA_API_KEY is not configured")

    nvidia_model = os.getenv("NVIDIA_MODEL", "meta/llama-4-maverick-17b-128e-instruct").strip()
    nvidia_url = os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions").strip()

    prompt = (
        "You are an agricultural AI analyst. Return ONLY valid JSON with the exact schema and key names below."
        " Do not include markdown or explanation.\n\n"
        "Schema:\n"
        "{\n"
        "  \"summary\": \"string\",\n"
        "  \"overall_health\": \"Excellent|Good|Moderate|Poor|Critical\",\n"
        "  \"confidence\": \"Low|Medium|High\",\n"
        "  \"risk_score\": number,\n"
        "  \"priority\": \"Low|Medium|High\",\n"
        "  \"issues\": [{\"id\":\"string\",\"name\":\"string\",\"affected_area_pct\":number,\"priority\":\"Low|Medium|High\"}],\n"
        "  \"why_happening\": [\"string\"],\n"
        "  \"immediate_actions\": [\"string\"],\n"
        "  \"monitor_next\": [\"string\"],\n"
        "  \"risk_if_ignored\": \"string\",\n"
        "  \"simple_advice\": {\"en\":\"string\",\"hi\":\"string\"}\n"
        "}\n\n"
        "Use only the provided metrics. If uncertain, reduce confidence.\n\n"
        f"Input:\n{json.dumps(input_payload, ensure_ascii=True)}"
    )

    headers = {
        "Authorization": f"Bearer {nvidia_api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    body = {
        "model": nvidia_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 700,
        "temperature": 0.2,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stream": False,
    }

    required_keys = [
        "summary",
        "overall_health",
        "confidence",
        "risk_score",
        "priority",
        "issues",
        "why_happening",
        "immediate_actions",
        "monitor_next",
        "risk_if_ignored",
        "simple_advice",
    ]

    try:
        response = requests.post(nvidia_url, headers=headers, json=body, timeout=25)
        if not response.ok:
            detail = response.text[:1200].replace("\n", " ")
            raise RuntimeError(f"NVIDIA HTTP {response.status_code}: {detail}")

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"NVIDIA returned no choices. Response: {json.dumps(data)[:600]}")

        text = choices[0].get("message", {}).get("content", "")
        if not text:
            raise RuntimeError(f"NVIDIA returned empty content. Response: {json.dumps(data)[:600]}")

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                raise RuntimeError(f"NVIDIA returned non-JSON text: {text[:400]}")
            parsed = json.loads(match.group(0))

        missing_keys = [key for key in required_keys if key not in parsed]
        if missing_keys:
            raise RuntimeError(
                f"NVIDIA response missing required keys: {', '.join(missing_keys)}. "
                f"Response: {json.dumps(parsed)[:500]}"
            )

        logger.info(f"[lite] AI analysis generated with NVIDIA model: {nvidia_model}")
        return _normalize_ai_analysis(parsed, input_payload.get("health_metrics", {}))
    except Exception as exc:
        raise RuntimeError(f"NVIDIA analysis generation failed: {exc}")


def _build_ai_analysis_input(location_info, crop, pixel_counts, ndwi_pixel_counts, ndre_pixel_counts, anomaly_data):
    ndvi_valid = pixel_counts.get("valid", 0)
    ndwi_valid = ndwi_pixel_counts.get("valid", 0) if isinstance(ndwi_pixel_counts, dict) else 0
    ndre_valid = ndre_pixel_counts.get("valid", 0) if isinstance(ndre_pixel_counts, dict) else 0

    ndvi_green_pct = _safe_pct(pixel_counts.get("green", 0), ndvi_valid)
    ndvi_yellow_pct = _safe_pct(pixel_counts.get("yellow", 0), ndvi_valid)
    ndvi_red_pct = _safe_pct(pixel_counts.get("red", 0), ndvi_valid)

    low_water_pixels = 0
    if isinstance(ndwi_pixel_counts, dict):
        low_water_pixels = ndwi_pixel_counts.get("brown", 0) + ndwi_pixel_counts.get("yellow", 0)
    low_water_area_pct = _safe_pct(low_water_pixels, ndwi_valid)

    low_nitrogen_pixels = 0
    if isinstance(ndre_pixel_counts, dict):
        low_nitrogen_pixels = ndre_pixel_counts.get("purple", 0) + ndre_pixel_counts.get("pink", 0)
    low_nitrogen_area_pct = _safe_pct(low_nitrogen_pixels, ndre_valid)

    negative_anomaly_pct = 0.0
    if isinstance(anomaly_data, dict):
        summary = anomaly_data.get("summary", {})
        if isinstance(summary, dict):
            negative_anomaly_pct = round(_safe_float(summary.get("stressed_percent", 0.0), 0.0), 2)

    return {
        "farm_context": {
            "crop": crop,
            "district": location_info.get("district"),
            "state": location_info.get("state"),
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        },
        "health_metrics": {
            "ndvi": {
                "green_pct": ndvi_green_pct,
                "yellow_pct": ndvi_yellow_pct,
                "red_pct": ndvi_red_pct,
            },
            "ndwi": {
                "low_water_area_pct": low_water_area_pct,
            },
            "ndre": {
                "low_nitrogen_area_pct": low_nitrogen_area_pct,
            },
            "anomaly": {
                "negative_anomaly_pct": negative_anomaly_pct,
            },
        },
    }

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

        try:
            ai_input = _build_ai_analysis_input(
                location_info=location_info,
                crop=request.crop,
                pixel_counts=trimmed_pixel_counts,
                ndwi_pixel_counts=ndwi_pixel_counts,
                ndre_pixel_counts=ndre_pixel_counts,
                anomaly_data=anomaly_data,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"AI input build failed: {exc}")
        try:
            response["ai_analysis"] = _generate_nvidia_ai_analysis(ai_input)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"AI analysis failed: {exc}")

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