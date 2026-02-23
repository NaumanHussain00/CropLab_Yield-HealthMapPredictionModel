# app.py

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Tuple, Optional
from datetime import datetime
import numpy as np
import tensorflow as tf
import joblib
import io
import base64
import logging
import merged_processor
from utils.preprocess import preprocess_input

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

# --- Load model and scaler with error handling ---

model = None
scaler = None
model_error = None
scaler_error = None

try:
    # Try loading with different compatibility options
    try:
        model = tf.keras.models.load_model("model.h5", compile=False)
    except Exception as e1:
        logger.warning(f"First load attempt failed: {e1}")
        # Try with custom objects if needed
        model = tf.keras.models.load_model("model.h5", compile=False, 
                                         custom_objects=None)
    logger.info("✅ Model loaded successfully")
except Exception as e:
    model_error = str(e)
    logger.error(f"❌ Error loading model: {e}")
    model = None

try:
    scaler = joblib.load("scaler.save")
    logger.info("✅ Scaler loaded successfully")
except Exception as e:
    scaler_error = str(e)
    logger.error(f"❌ Error loading scaler: {e}")

# Pydantic models

def get_corresponding_date():
    """Fetch corresponding date based on current date"""
    current = datetime.now()
    # Assuming corresponding is current year - 3, October 1st
    year = current.year - 3
    return f"{year}-10-01"

class PredictRequest(BaseModel):
    coordinates: List[List[float]]  # List of [longitude, latitude] points

class HeatmapRequest(BaseModel):
    coordinates: List[List[float]]  # List of [longitude, latitude] points
    t1: float = 3.0  # Threshold for low yield
    t2: float = 4.5  # Threshold for high yield

# Health check endpoints
@app.get("/")
@app.get("/health")
async def health_check():
    """Health check and root endpoint for monitoring"""
    gee_status = merged_processor.initialize_earth_engine()
    
    return {
        "status": "healthy" if model and scaler and gee_status else "unhealthy",
        "message": "🌾 Crop Yield Prediction API is running",
        "components": {
            "model": "loaded" if model else f"error: {model_error}",
            "scaler": "loaded" if scaler else f"error: {scaler_error}",
            "google_earth_engine": "connected" if gee_status else "disconnected"
        },
        "version": "1.0.0",
        "endpoints": {
            "/predict": "Predict crop yield from coordinates",
            "/generate_heatmap": "Generate color-coded heatmap visualization",
            "/export_arrays": "Export NDVI and sensor arrays as .npz file"
        }
    }

@app.post("/predict")
async def predict(request: PredictRequest):
    """
    Predict crop yield from coordinates using data fetched from Google Earth Engine.

    Takes a list of [longitude, latitude] points, generates NDVI and sensor data automatically,
    and returns the predicted yield value.

    """
    # --- Check model and scaler loaded ---
    if model is None or scaler is None:
        msg = "Model or scaler not loaded. "
        if model_error:
            msg += f"Model error: {model_error}. "
        if scaler_error:
            msg += f"Scaler error: {scaler_error}. "
        raise HTTPException(status_code=500, detail=msg.strip())

    try:
        # --- Initialize Earth Engine ---
        if not merged_processor.initialize_earth_engine():
            raise HTTPException(status_code=500, detail="Failed to initialize Google Earth Engine")

        # --- Get corresponding date ---
        date_str = get_corresponding_date()
        logging.info(f"Using date: {date_str}")

        # --- Generate NDVI and Sensor data ---
        geojson_dict = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [request.coordinates]
            }
        }
        try:
            ndvi_data, sensor_data = merged_processor.generate_ndvi_and_sensor_npy(
                geojson_dict, date_str
            )
        except Exception as e:
            import traceback
            raise HTTPException(status_code=400, detail=f"Error generating NDVI and sensor data: {str(e)}\n{traceback.format_exc()}")

        if ndvi_data is None or sensor_data is None:
            raise HTTPException(status_code=400, detail="Failed to generate NDVI and sensor data from coordinates (returned None)")

        # Keep arrays in-memory (do not save .npy files). Use these arrays directly
        logger.info(f"Generated NDVI in-memory with shape: {ndvi_data.shape}")
        logger.info(f"Generated Sensor in-memory with shape: {sensor_data.shape}")

        # --- Prepare data for prediction ---
        # NDVI preprocessing
        if ndvi_data.ndim == 2:
            ndvi_processed = ndvi_data[..., np.newaxis]  # (H, W, 1)
        else:
            ndvi_processed = ndvi_data

        ndvi_processed = np.expand_dims(ndvi_processed, axis=0)    # (1, H, W, C)
        ndvi_processed = np.expand_dims(ndvi_processed, axis=1)    # (1, 1, H, W, C)

        # Sensor preprocessing
        if sensor_data.ndim == 2:
            sensor_processed = sensor_data[..., np.newaxis]  # (H, W, 1)
        else:
            sensor_processed = sensor_data

        sensor_processed = np.expand_dims(sensor_processed, axis=0)
        sensor_processed = np.expand_dims(sensor_processed, axis=1)

        # --- Align sensor channels to scaler expectations to avoid feature mismatches ---
        try:
            expected_features = getattr(scaler, "n_features_in_", None)
            if expected_features is not None:
                expected_features = int(expected_features)
        except Exception:
            expected_features = None

        if expected_features is not None:
            current_channels = sensor_processed.shape[-1]
            if current_channels != expected_features:
                logging.warning(f"Sensor channels ({current_channels}) != scaler expected ({expected_features}); trimming or padding to match.")
                if current_channels > expected_features:
                    # trim extra channels
                    sensor_processed = sensor_processed[..., :expected_features]
                else:
                    # pad with zeros for missing channels
                    pad_width = expected_features - current_channels
                    pad_shape = list(sensor_processed.shape[:-1]) + [pad_width]
                    pad = np.zeros(tuple(pad_shape), dtype=sensor_processed.dtype)
                    sensor_processed = np.concatenate([sensor_processed, pad], axis=-1)

        # --- Preprocess inputs ---
        ndvi_processed, sensor_processed = preprocess_input(ndvi_processed, sensor_processed, scaler)

        # --- Predict yield ---
        prediction = model.predict([ndvi_processed, sensor_processed])
        predicted_yield = float(prediction[0][0])  # Single yield value

        # --- Update GEE with predicted yield ---
        import ee
        polygon = merged_processor.create_geometry_from_geojson(geojson_dict)
        yield_image = ee.Image.constant(predicted_yield).clip(polygon)
        asset_id = f"projects/pk07007/assets/predicted_yield_{int(datetime.now().timestamp())}"
        task = ee.batch.Export.image.toAsset(
            image=yield_image,
            description='Predicted Yield',
            assetId=asset_id,
            scale=10,
            region=polygon,
            maxPixels=1e10
        )
        task.start()
        logger.info(f"Started export to GEE asset: {asset_id}")

        return {"predicted_yield": predicted_yield, "gee_asset_id": asset_id, "ndvi_shape": ndvi_data.shape, "sensor_shape": sensor_data.shape}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/generate_heatmap")
async def generate_heatmap(request: HeatmapRequest):
    """
    Generate yield prediction heatmap overlay from coordinates.

    Takes a list of [longitude, latitude] points, generates NDVI and sensor data,
    predicts yield using the CNN+LSTM model, and returns a color-coded
    heatmap overlay (red/yellow/green based on yield thresholds).
    """
    # --- Check model and scaler loaded ---
    if model is None or scaler is None:
        msg = "Model or scaler not loaded. "
        if model_error:
            msg += f"Model error: {model_error}. "
        if scaler_error:
            msg += f"Scaler error: {scaler_error}. "
        raise HTTPException(status_code=500, detail=msg.strip())

    try:
        # --- Initialize Earth Engine ---
        if not merged_processor.initialize_earth_engine():
            raise HTTPException(status_code=500, detail="Failed to initialize Google Earth Engine")

        # --- Get corresponding date ---
        date_str = get_corresponding_date()
        logging.info(f"Using date: {date_str}")

        # --- Generate NDVI and Sensor data ---
        geojson_dict = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [request.coordinates]
            }
        }
        
        # Generate data once without predicted yield
        ndvi_data, sensor_data = merged_processor.generate_ndvi_and_sensor_npy(
            geojson_dict, date_str
        )

        if ndvi_data is None or sensor_data is None:
            raise HTTPException(status_code=400, detail="Failed to generate NDVI and sensor data from coordinates")

        # --- Prepare data for prediction ---
        # NDVI preprocessing
        if ndvi_data.ndim == 2:
            ndvi_processed = ndvi_data[..., np.newaxis]  # (H, W, 1)
        else:
            ndvi_processed = ndvi_data

        ndvi_processed = np.expand_dims(ndvi_processed, axis=0)    # (1, H, W, C)
        ndvi_processed = np.expand_dims(ndvi_processed, axis=1)    # (1, 1, H, W, C)

        # Sensor preprocessing
        if sensor_data.ndim == 2:
            sensor_processed = sensor_data[..., np.newaxis]  # (H, W, 1)
        else:
            sensor_processed = sensor_data

        sensor_processed = np.expand_dims(sensor_processed, axis=0)
        sensor_processed = np.expand_dims(sensor_processed, axis=1)

        # --- Align sensor channels to scaler expectations to avoid feature mismatches ---
        try:
            expected_features = getattr(scaler, "n_features_in_", None)
            if expected_features is not None:
                expected_features = int(expected_features)
        except Exception:
            expected_features = None

        if expected_features is not None:
            current_channels = sensor_processed.shape[-1]
            if current_channels != expected_features:
                logger.warning(f"Sensor channels ({current_channels}) != scaler expected ({expected_features}); trimming or padding to match.")
                if current_channels > expected_features:
                    # trim extra channels
                    sensor_processed = sensor_processed[..., :expected_features]
                else:
                    # pad with zeros for missing channels
                    pad_width = expected_features - current_channels
                    pad_shape = list(sensor_processed.shape[:-1]) + [pad_width]
                    pad = np.zeros(tuple(pad_shape), dtype=sensor_processed.dtype)
                    sensor_processed = np.concatenate([sensor_processed, pad], axis=-1)

        # --- Preprocess inputs ---
        ndvi_processed, sensor_processed = preprocess_input(ndvi_processed, sensor_processed, scaler)

        # --- Predict yield ---
        prediction = model.predict([ndvi_processed, sensor_processed])[0][0]
        predicted_yield = float(prediction)

        # --- Apply yield comparison and NDVI adjustment directly ---
        # Get district and old yield for comparison
        centroid_lat, centroid_lon = merged_processor.get_centroid_coordinates(geojson_dict)
        yield_df = merged_processor.load_yield_data()
        
        # Initialize location info with defaults
        location_info = {
            "district": "unknown",
            "coordinates": {"latitude": None, "longitude": None},
            "complete_address": "Location not available"
        }
        old_yield = 1.0
        yield_ratio = 1.0
        growth_percentage = 0.0
        
        if centroid_lat is not None and centroid_lon is not None:
            # Get complete location information
            district, complete_location = merged_processor.get_district_and_location_sync(centroid_lat, centroid_lon)
            old_yield = merged_processor.get_old_yield_for_district(district, yield_df)
            
            # Update location info
            location_info = {
                "district": district,
                "coordinates": {"latitude": centroid_lat, "longitude": centroid_lon},
                "complete_address": complete_location
            }
            
            # Apply yield comparison and adjust NDVI
            final_ndvi_data, yield_ratio = merged_processor.compare_yields_and_adjust_ndvi(
                ndvi_data, predicted_yield, old_yield
            )
            
            # Calculate growth percentage
            growth_percentage = ((predicted_yield - old_yield) / old_yield) * 100 if old_yield > 0 else 0.0
            
            logger.info(f"District: {district}, Old yield: {old_yield}, Predicted yield: {predicted_yield}, Ratio: {yield_ratio:.2f}, Growth: {growth_percentage:.2f}%")
        else:
            final_ndvi_data = ndvi_data
            logger.warning("Could not get district information, using original NDVI data")

        # --- Generate separate heatmap masks ---
        red_mask, yellow_mask, green_mask, pixel_counts = merged_processor.create_separate_yield_masks(
            final_ndvi_data, predicted_yield, request.t1, request.t2
        )

        if red_mask is None or yellow_mask is None or green_mask is None:
            raise HTTPException(status_code=500, detail="Failed to generate heatmap masks")

        # --- Generate farmer suggestions ---
        suggestions = merged_processor.generate_farmer_suggestions(
            predicted_yield=predicted_yield,
            old_yield=old_yield,
            pixel_counts=pixel_counts,
            sensor_data=sensor_data,
            location_info=location_info,
            thresholds={"t1": request.t1, "t2": request.t2}
        )

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

        response = {
            "predicted_yield": predicted_yield,
            "old_yield": old_yield,
            "growth": {
                "ratio": yield_ratio,
                "percentage": growth_percentage
            },
            "location": location_info,
            "ndvi_shape": final_ndvi_data.shape,
            "sensor_shape": sensor_data.shape,
            "masks": {
                "red_mask_base64": red_base64,
                "yellow_mask_base64": yellow_base64,
                "green_mask_base64": green_base64
            },
            "pixel_counts": pixel_counts,
            "thresholds": {"t1": request.t1, "t2": request.t2},
            "suggestions": suggestions
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Heatmap generation error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {str(e)}")


@app.post("/export_arrays")
async def export_arrays(request: HeatmapRequest):
    """
    Utility endpoint: generate NDVI and sensor arrays for the provided coordinates
    and return them as a .npz file in-memory (no disk writes).
    """
    try:
        if not merged_processor.initialize_earth_engine():
            raise HTTPException(status_code=500, detail="Failed to initialize Google Earth Engine")

        date_str = get_corresponding_date()

        geojson_dict = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [request.coordinates]
            }
        }

        ndvi_data, sensor_data = merged_processor.generate_ndvi_and_sensor_npy(geojson_dict, date_str)

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