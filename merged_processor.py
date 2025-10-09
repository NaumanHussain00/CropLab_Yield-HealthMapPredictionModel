# merged_processor.py
# Merged functionality from ndvi_heatmap.py and main_sensor.py
# Generates NDVI and Sensor .npy files in memory without saving images

import ee
import numpy as np
import json
import os
import requests
import pandas as pd
import tempfile
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SERVICE_ACCOUNT_PATH = 'earth-engine-service-account.json'
DATE_RANGE_START = '2017-10-01'
DATE_RANGE_END = '2018-03-31'
CLOUD_THRESHOLD = 20
MAX_PIXELS = 1e10

# Sensor assets (same as main_sensor.py)
SENSOR_ASSETS = {
    'ECe': 'projects/pk07007/assets/ECe',
    'N': 'projects/pk07007/assets/N',
    'P': 'projects/pk07007/assets/P',
    'pH': 'projects/pk07007/assets/pH',
    'OC': 'projects/pk07007/assets/OC'
}

def initialize_earth_engine():
    """Initialize Google Earth Engine with service account authentication"""
    try:
        # Check if already initialized
        try:
            ee.Number(1).getInfo()
            logger.info("✅ Google Earth Engine already initialized")
            return True
        except:
            pass

        # Try environment variables first (for production/Render)
        if os.getenv('GEE_SERVICE_ACCOUNT_EMAIL') and os.getenv('GEE_PRIVATE_KEY'):
            logger.info("🌐 Initializing GEE with environment variables (Production mode)")
            
            # Create service account credentials from environment variables
            service_account_info = {
                "type": "service_account",
                "project_id": os.getenv('GEE_PROJECT_ID', 'pk07007'),
                "private_key_id": os.getenv('GEE_PRIVATE_KEY_ID'),
                "private_key": os.getenv('GEE_PRIVATE_KEY').replace('\\n', '\n'),
                "client_email": os.getenv('GEE_SERVICE_ACCOUNT_EMAIL'),
                "client_id": os.getenv('GEE_CLIENT_ID'),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": os.getenv('GEE_CLIENT_CERT_URL'),
                "universe_domain": "googleapis.com"
            }
            
            # Create a temporary file with the credentials
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(service_account_info, temp_file)
                temp_file_path = temp_file.name
            
            try:
                # Initialize with temporary file
                credentials = ee.ServiceAccountCredentials(
                    email=service_account_info['client_email'],
                    key_file=temp_file_path
                )
                ee.Initialize(credentials)
                
                # Test the initialization
                test_result = ee.Number(1).getInfo()
                if test_result == 1:
                    logger.info("✅ Google Earth Engine initialized successfully with environment variables")
                    return True
                else:
                    logger.error("❌ GEE initialization test failed")
                    return False
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
        
        # Fallback to local file (for development)
        elif os.path.exists(SERVICE_ACCOUNT_PATH):
            logger.info("� Initializing GEE with local service account file (Development mode)")
            
            with open(SERVICE_ACCOUNT_PATH, 'r') as f:
                service_account = json.load(f)

            logger.info(f"Initializing GEE with service account: {service_account.get('client_email', 'Unknown')}")

            # Use the key file path directly with ee.Initialize
            ee.Initialize(ee.ServiceAccountCredentials(
                email=service_account['client_email'],
                key_file=SERVICE_ACCOUNT_PATH
            ))
            
            # Test the initialization
            test_result = ee.Number(1).getInfo()
            if test_result == 1:
                logger.info("✅ Google Earth Engine initialized successfully")
                return True
            else:
                logger.error("❌ GEE initialization test failed")
                return False
        else:
            logger.error("❌ No Google Earth Engine credentials found!")
            logger.error("💡 For production: Set GEE_SERVICE_ACCOUNT_EMAIL and GEE_PRIVATE_KEY environment variables")
            logger.error(f"💡 For development: Ensure {SERVICE_ACCOUNT_PATH} exists")
            return False
            
    except Exception as e:
        error_msg = str(e)
        if "Invalid JWT Signature" in error_msg:
            logger.error(f"❌ Failed to initialize Google Earth Engine: {e}")
            logger.error("🕐 JWT SIGNATURE ERROR DETECTED!")
            logger.error("📋 This is typically caused by system clock synchronization issues.")
            logger.error("💡 SOLUTION: Synchronize your system clock:")
            logger.error("   • Windows: Right-click clock → 'Adjust date/time' → 'Sync now'")
            logger.error("   • Or run as Administrator: w32tm /resync")
            logger.error("   • Ensure 'Set time automatically' is enabled")
            logger.error("🔄 After syncing, restart the application")
            return False
        else:
            logger.error(f"❌ Failed to initialize Google Earth Engine: {e}")
            return False

async def get_district_from_coordinates(lat, lon):
    """Get district from coordinates using OpenStreetMap Nominatim API"""
    try:
        logger.info(f"Getting district for coordinates: {lat}, {lon}")
        
        url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}"
        
        headers = {
            'User-Agent': 'AgriProject/1.0'  # Required by Nominatim API
        }
        
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Response: {data}")
            
            # Extract district from address details
            address = data.get('address', {})
            district = address.get('state_district') or address.get('county') or address.get('district') or 'agra'
            
            logger.info(f"Detected district: {district}")
            return district.lower()
        else:
            logger.warning(f"Failed to get location data: {response.status_code}")
            return 'agra'  # Default fallback
            
    except Exception as e:
        logger.error(f"Error getting district: {e}")
        return 'agra'  # Default fallback

def get_district_from_coordinates_sync(lat, lon):
    """Synchronous version of get_district_from_coordinates"""
    try:
        logger.info(f"Getting district for coordinates: {lat}, {lon}")
        
        url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}"
        
        headers = {
            'User-Agent': 'AgriProject/1.0'  # Required by Nominatim API
        }
        
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Response: {data}")
            
            # Extract district from address details
            address = data.get('address', {})
            district = address.get('state_district') or address.get('county') or address.get('district') or 'agra'
            
            logger.info(f"Detected district: {district}")
            return district.lower()
        else:
            logger.warning(f"Failed to get location data: {response.status_code}")
            return 'agra'  # Default fallback
            
    except Exception as e:
        logger.error(f"Error getting district: {e}")
        return 'agra'  # Default fallback

def get_district_and_location_sync(lat, lon):
    """Get both district and complete location information from coordinates"""
    try:
        logger.info(f"Getting district and location for coordinates: {lat}, {lon}")
        
        url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}"
        
        headers = {
            'User-Agent': 'AgriProject/1.0'  # Required by Nominatim API
        }
        
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Location response received")
            
            # Extract district from address details
            address = data.get('address', {})
            district = address.get('state_district') or address.get('county') or address.get('district') or 'agra'
            complete_address = data.get('display_name', 'Address not available')
            
            logger.info(f"Detected district: {district}")
            return district.lower(), complete_address
        else:
            logger.warning(f"Failed to get location data: {response.status_code}")
            return 'agra', 'Location not available'  # Default fallback
            
    except Exception as e:
        logger.error(f"Error getting district and location: {e}")
        return 'agra', 'Location not available'  # Default fallback

def get_district_and_location_sync(lat, lon):
    """Get both district and complete location information from coordinates"""
    try:
        logger.info(f"Getting district and location for coordinates: {lat}, {lon}")
        
        url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}"
        
        headers = {
            'User-Agent': 'AgriProject/1.0'  # Required by Nominatim API
        }
        
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Response: {data}")
            
            # Extract district from address details
            address = data.get('address', {})
            district = address.get('state_district') or address.get('county') or address.get('district') or 'agra'
            
            # Get complete display name
            complete_location = data.get('display_name', 'Location not available')
            
            logger.info(f"Detected district: {district}")
            logger.info(f"Complete location: {complete_location}")
            
            return district.lower(), complete_location
        else:
            logger.warning(f"Failed to get location data: {response.status_code}")
            return 'agra', 'Location not available'  # Default fallback
            
    except Exception as e:
        logger.error(f"Error getting district and location: {e}")
        return 'agra', 'Location not available'  # Default fallback

def load_yield_data(csv_path='district_yield.csv'):
    """Load yield data from CSV file"""
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded yield data with {len(df)} districts")
            return df
        else:
            logger.error(f"Yield CSV file not found: {csv_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading yield data: {e}")
        return None

def get_old_yield_for_district(district_name, yield_df):
    """Get old yield value for a specific district"""
    try:
        if yield_df is None:
            logger.warning("No yield data available")
            return 1.0  # Default yield
            
        # Clean district name for matching
        district_clean = district_name.lower().strip()
        
        # Try exact match first
        match = yield_df[yield_df['district_name'].str.lower().str.strip() == district_clean]
        
        if len(match) > 0:
            old_yield = float(match.iloc[0]['yield'])
            logger.info(f"Found exact match for {district_name}: {old_yield}")
            return old_yield
        
        # Try partial match
        partial_match = yield_df[yield_df['district_name'].str.lower().str.contains(district_clean, na=False)]
        
        if len(partial_match) > 0:
            old_yield = float(partial_match.iloc[0]['yield'])
            logger.info(f"Found partial match for {district_name}: {old_yield}")
            return old_yield
            
        logger.warning(f"No yield data found for district: {district_name}, using default")
        return 1.0  # Default yield
        
    except Exception as e:
        logger.error(f"Error getting old yield for district {district_name}: {e}")
        return 1.0

def compare_yields_and_adjust_ndvi(ndvi_data, predicted_yield, old_yield, improvement_factor=0.1, deterioration_factor=0.05):
    """Compare predicted vs old yield and adjust NDVI pixel parameters accordingly"""
    try:
        logger.info(f"Comparing yields - Predicted: {predicted_yield}, Old: {old_yield}")
        
        yield_ratio = predicted_yield / old_yield if old_yield > 0 else 1.0
        
        # Create a copy of NDVI data to modify
        adjusted_ndvi = np.copy(ndvi_data)
        
        if yield_ratio > 1.0:
            # Growth detected - improve NDVI values
            improvement = (yield_ratio - 1.0) * improvement_factor
            adjusted_ndvi = np.clip(adjusted_ndvi * (1 + improvement), -1, 1)
            logger.info(f"Growth detected ({yield_ratio:.2f}x), improving NDVI by {improvement:.3f}")
            
        elif yield_ratio < 1.0:
            # Decline detected - worsen NDVI values  
            deterioration = (1.0 - yield_ratio) * deterioration_factor
            adjusted_ndvi = np.clip(adjusted_ndvi * (1 - deterioration), -1, 1)
            logger.info(f"Decline detected ({yield_ratio:.2f}x), worsening NDVI by {deterioration:.3f}")
            
        else:
            logger.info("No significant yield change detected, keeping NDVI unchanged")
            
        return adjusted_ndvi, yield_ratio
        
    except Exception as e:
        logger.error(f"Error comparing yields and adjusting NDVI: {e}")
        return ndvi_data, 1.0

def get_centroid_coordinates(geojson_feature):
    """Get centroid coordinates from GeoJSON feature"""
    try:
        coordinates = geojson_feature['geometry']['coordinates']
        geometry_type = geojson_feature['geometry']['type']
        
        if geometry_type == 'Polygon':
            # Calculate centroid of polygon
            if isinstance(coordinates[0], list) and len(coordinates[0]) > 0:
                coords = coordinates[0] if isinstance(coordinates[0][0], list) else coordinates
                lons = [point[0] for point in coords if len(point) >= 2]
                lats = [point[1] for point in coords if len(point) >= 2]
                
                if lons and lats:
                    centroid_lon = sum(lons) / len(lons)
                    centroid_lat = sum(lats) / len(lats)
                    return centroid_lat, centroid_lon
                    
        elif geometry_type == 'Point':
            return coordinates[1], coordinates[0]  # lat, lon
            
        logger.warning(f"Unable to extract centroid from geometry type: {geometry_type}")
        return None, None
        
    except Exception as e:
        logger.error(f"Error getting centroid coordinates: {e}")
        return None, None

def create_geometry_from_geojson(geojson_feature):
    """Create GEE geometry from GeoJSON feature - from ndvi_heatmap.py"""
    try:
        coordinates = geojson_feature['geometry']['coordinates']
        geometry_type = geojson_feature['geometry']['type']

        if geometry_type == 'Polygon':
            if isinstance(coordinates, list) and len(coordinates) > 0:
                if isinstance(coordinates[0], list) and len(coordinates[0]) > 0:
                    if isinstance(coordinates[0][0], list) and len(coordinates[0][0]) == 2:
                        return ee.Geometry.Polygon(coordinates)
                    elif isinstance(coordinates[0][0], (int, float)):
                        if len(coordinates[0]) % 2 == 0:
                            reshaped = [[coordinates[0][i], coordinates[0][i+1]] for i in range(0, len(coordinates[0]), 2)]
                            return ee.Geometry.Polygon([reshaped])
            return ee.Geometry.Polygon(coordinates)
        elif geometry_type == 'MultiPolygon':
            return ee.Geometry.MultiPolygon(coordinates)
        else:
            raise ValueError(f"Unsupported geometry type: {geometry_type}")
    except Exception as e:
        logger.error(f"Error creating geometry: {e}")
        return None

def search_satellite_image(polygon, start_date, end_date, cloud_threshold=20):
    """Search for satellite image in GEE - from ndvi_heatmap.py"""
    try:
        logger.info(f"Searching for Sentinel-2 images from {start_date} to {end_date} with cloud threshold {cloud_threshold}%")

        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(polygon)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
                     .sort('CLOUDY_PIXEL_PERCENTAGE'))

        size = collection.size().getInfo()
        logger.info(f"Found {size} images in collection")

        if size == 0:
            logger.info("No images found with current filters, trying without cloud filter...")
            collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(polygon)
                         .filterDate(start_date, end_date)
                         .sort('CLOUDY_PIXEL_PERCENTAGE'))
            size = collection.size().getInfo()
            logger.info(f"Found {size} images without cloud filter")

            if size == 0:
                logger.info("No images found, trying with buffer periods...")
                try:
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

                    extended_start = start_dt - relativedelta(months=3)
                    extended_start_str = extended_start.strftime('%Y-%m-%d')

                    extended_end = end_dt + relativedelta(months=3)
                    extended_end_str = extended_end.strftime('%Y-%m-%d')

                    logger.info(f"Trying extended date range: {extended_start_str} to {extended_end_str}")
                    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                                 .filterBounds(polygon)
                                 .filterDate(extended_start_str, extended_end_str)
                                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
                                 .sort('CLOUDY_PIXEL_PERCENTAGE'))

                    size = collection.size().getInfo()
                    logger.info(f"Found {size} images in extended date range")

                    if size == 0:
                        logger.info("No images found with cloud filter in extended range, trying without cloud filter...")
                        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                                     .filterBounds(polygon)
                                     .filterDate(extended_start_str, extended_end_str)
                                     .sort('CLOUDY_PIXEL_PERCENTAGE'))
                        size = collection.size().getInfo()
                        logger.info(f"Found {size} images in extended range without cloud filter")

                        if size == 0:
                            logger.info("No images found even with extended date range and no cloud filter")
                            return None
                except Exception as date_error:
                    logger.error(f"Error with date extension: {date_error}")
                    return None

        image = collection.first()
        info = image.getInfo()
        if info and 'id' in info:
            logger.info(f"Selected image: {info['id']}")
            return image
        else:
            logger.info("No suitable image found")
            return None
    except Exception as e:
        logger.error(f"Error searching for satellite image: {e}")
        return None

def select_ndvi_bands(image):
    """Select NDVI bands (B8-NIR, B4-Red) from Sentinel-2 image - from ndvi_heatmap.py"""
    try:
        ndvi_bands = image.select(['B8', 'B4']).rename(['NIR', 'Red'])
        return ndvi_bands
    except Exception as e:
        logger.error(f"Error selecting NDVI bands: {e}")
        return None

def calculate_ndvi(image):
    """Calculate NDVI from NIR and Red bands - from ndvi_heatmap.py"""
    try:
        nir = image.select('NIR')
        red = image.select('Red')
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        return ndvi
    except Exception as e:
        logger.error(f"Error calculating NDVI: {e}")
        return None

def export_image_data(image, region, scale=10, band_names=None):
    """Export image data as numpy array - from ndvi_heatmap.py"""
    try:
        logger.info(f"Exporting image data with scale {scale} meters per pixel...")

        clipped = image.clip(region)

        if band_names is None:
            band_info = image.getInfo()
            if 'bands' in band_info:
                band_names = [band['id'] for band in band_info['bands']]
            else:
                band_names = ['band']

        logger.info(f"Exporting bands: {band_names}")

        bounds = region.bounds().getInfo()
        coords = bounds['coordinates'][0]

        min_lon = min(coord[0] for coord in coords)
        max_lon = max(coord[0] for coord in coords)
        min_lat = min(coord[1] for coord in coords)
        max_lat = max(coord[1] for coord in coords)

        avg_lat = (min_lat + max_lat) / 2
        meters_per_degree_lon = 111319 * np.cos(np.radians(avg_lat))
        meters_per_degree_lat = 111139

        width = int((max_lon - min_lon) * meters_per_degree_lon / scale)
        height = int((max_lat - min_lat) * meters_per_degree_lat / scale)

        logger.info(f"Calculated image dimensions: {width}x{height} pixels")

        scale_x = (max_lon - min_lon) / width
        scale_y = (max_lat - min_lat) / height

        request = {
            'expression': clipped,
            'fileFormat': 'NUMPY_NDARRAY',
            'bandIds': band_names,
            'grid': {
                'dimensions': {'width': width, 'height': height},
                'affineTransform': {
                    'scaleX': scale_x, 'shearX': 0, 'translateX': min_lon,
                    'shearY': 0, 'scaleY': -scale_y, 'translateY': max_lat
                },
                'crsCode': 'EPSG:4326'
            }
        }

        logger.info("Fetching pixel data from GEE...")
        pixel_data = ee.data.computePixels(request)

        if pixel_data is not None:
            logger.info(f"Successfully fetched pixel data with shape: {pixel_data.shape}")
            return pixel_data
        else:
            logger.error("Failed to fetch pixel data")
            return None

    except Exception as e:
        logger.error(f"Error exporting image data: {e}")
        return None

def crop_to_256(data):
    """Crop data to 256x256 pixels, centered if possible - from ndvi_heatmap.py"""
    try:
        if hasattr(data, 'dtype') and data.dtype.names is not None:
            if 'NDVI' in data.dtype.names:
                height, width = data['NDVI'].shape
                start_y = max(0, (height - 256) // 2)
                start_x = max(0, (width - 256) // 2)
                end_y = min(height, start_y + 256)
                end_x = min(width, start_x + 256)

                cropped_data = np.empty((end_y - start_y, end_x - start_x), dtype=data.dtype)
                for name in data.dtype.names:
                    cropped_data[name] = data[name][start_y:end_y, start_x:end_x]
                return cropped_data
        else:
            if data.ndim >= 2:
                height, width = data.shape[:2]
                start_y = max(0, (height - 256) // 2)
                start_x = max(0, (width - 256) // 2)
                end_y = min(height, start_y + 256)
                end_x = min(width, start_x + 256)

                if data.ndim == 2:
                    return data[start_y:end_y, start_x:end_x]
                else:
                    return data[start_y:end_y, start_x:end_x, :]
        return data
    except Exception as e:
        logger.error(f"Error cropping data: {e}")
        return data

def get_sensor_data(region):
    """Get sensor data for all 5 sensors - from main_sensor.py"""
    try:
        logger.info("Fetching sensor data...")
        sensor_data = {}
        valid_sensors = []

        for sensor_name, asset_id in SENSOR_ASSETS.items():
            try:
                logger.info(f"Loading {sensor_name} sensor data from {asset_id}...")
                sensor_image = ee.Image(asset_id)

                image_info = sensor_image.getInfo()
                if not image_info:
                    logger.warning(f"Failed to load {sensor_name}: No image info")
                    continue

                available_bands = image_info.get('bands', [])
                if len(available_bands) == 0:
                    logger.warning(f"No bands available for {sensor_name}")
                    continue

                # Use only the first band of each sensor
                actual_band_ids = [band['id'] for band in available_bands[:1]]
                logger.info(f"Using first band for {sensor_name}: {actual_band_ids}")

                selected_image = sensor_image.select(actual_band_ids)
                valid_sensors.append(selected_image)
                sensor_data[sensor_name] = {'bands': actual_band_ids}

                logger.info(f"✅ {sensor_name}: {len(actual_band_ids)} bands loaded")

            except Exception as e:
                logger.warning(f"⚠️ Failed to load {sensor_name}: {e}")
                continue

        if len(valid_sensors) == 0:
            logger.error("No sensor data could be loaded")
            return None

        combined_sensor_image = ee.Image.cat(valid_sensors)
        combined_sensor_image = combined_sensor_image.reproject('EPSG:4326', scale=10).resample('bilinear')
        combined_sensor_image = combined_sensor_image.clip(region)

        logger.info(f"✅ Combined sensor data: {len(valid_sensors)} sensors loaded")
        return combined_sensor_image

    except Exception as e:
        logger.error(f"Error getting sensor data: {e}")
        return None

def export_sensor_data(image, region, scale=10):
    """Export sensor data as numpy array - from main_sensor.py"""
    try:
        logger.info(f"Exporting sensor data with scale {scale} meters per pixel...")

        clipped = image.clip(region)

        bounds = region.bounds().getInfo()
        coords = bounds['coordinates'][0]

        min_lon = min(coord[0] for coord in coords)
        max_lon = max(coord[0] for coord in coords)
        min_lat = min(coord[1] for coord in coords)
        max_lat = max(coord[1] for coord in coords)

        avg_lat = (min_lat + max_lat) / 2
        meters_per_degree_lon = 111319 * np.cos(np.radians(avg_lat))
        meters_per_degree_lat = 111139

        width = int((max_lon - min_lon) * meters_per_degree_lon / scale)
        height = int((max_lat - min_lat) * meters_per_degree_lat / scale)

        if width * height > MAX_PIXELS:
            logger.warning(f"Image size ({width}x{height}) exceeds GEE limit")
            ratio = width / height
            new_width = int(np.sqrt(MAX_PIXELS * ratio))
            new_height = int(MAX_PIXELS / new_width)
            width, height = new_width, new_height
            logger.info(f"Reduced dimensions to {width}x{height}")

        logger.info(f"Calculated image dimensions: {width}x{height} pixels")

        scale_x = (max_lon - min_lon) / width
        scale_y = (max_lat - min_lat) / height

        image_info = image.getInfo()
        band_names = [band['id'] for band in image_info.get('bands', [])]

        if not band_names:
            logger.error("No bands found in sensor image")
            return None

        logger.info(f"Sensor data bands: {band_names}")

        request = {
            'expression': clipped,
            'fileFormat': 'NUMPY_NDARRAY',
            'bandIds': band_names,
            'grid': {
                'dimensions': {'width': width, 'height': height},
                'affineTransform': {
                    'scaleX': scale_x, 'shearX': 0, 'translateX': min_lon,
                    'shearY': 0, 'scaleY': -scale_y, 'translateY': max_lat
                },
                'crsCode': 'EPSG:4326'
            }
        }

        logger.info("Fetching pixel data from GEE...")
        pixel_data = ee.data.computePixels(request)

        if pixel_data is not None:
            logger.info(f"Successfully fetched sensor data with shape: {pixel_data.shape}")
            return pixel_data
        else:
            logger.error("Failed to fetch sensor data")
            return None

    except Exception as e:
        logger.error(f"Error exporting sensor data: {e}")
        return None

def combine_ndvi_sensor_data(ndvi_data, sensor_data):
    """Combine NDVI and sensor data into a single 3D array - from main_sensor.py"""
    try:
        logger.info("Combining NDVI and sensor data into 21-band array...")

        if hasattr(ndvi_data, 'dtype') and ndvi_data.dtype.names is not None:
            ndvi_values = ndvi_data['NDVI']
            ndvi_3d = np.expand_dims(ndvi_values, axis=2)
        else:
            if len(ndvi_data.shape) == 2:
                ndvi_3d = np.expand_dims(ndvi_data, axis=2)
            else:
                ndvi_3d = ndvi_data[:, :, :1]

        if hasattr(sensor_data, 'dtype') and sensor_data.dtype.names is not None:
            band_names = sensor_data.dtype.names
            sensor_bands = []
            for band_name in band_names:
                band_data = sensor_data[band_name]
                if len(band_data.shape) == 2:
                    band_3d = np.expand_dims(band_data, axis=2)
                else:
                    band_3d = band_data
                sensor_bands.append(band_3d)
            sensor_3d = np.concatenate(sensor_bands, axis=2)
        else:
            sensor_3d = sensor_data

        combined_array = np.concatenate([ndvi_3d, sensor_3d], axis=2)

        logger.info(f"Combined array shape: {combined_array.shape}")
        logger.info(f"✅ Combined NDVI (1 band) + Sensor ({sensor_3d.shape[2]} bands) = {combined_array.shape[2]} bands total")

        return combined_array

    except Exception as e:
        logger.error(f"Error combining and processing data: {e}")
        return None

def generate_ndvi_and_sensor_npy(geojson_feature, date_str="2018-10-01"):
    """Generate NDVI and Sensor .npy data in memory from GeoJSON feature"""
    try:
        logger.info("Generating NDVI and Sensor data from GeoJSON...")

        # Create Earth Engine polygon
        polygon = create_geometry_from_geojson(geojson_feature)
        if polygon is None:
            logger.error("Failed to create polygon")
            return None, None

        # Parse date and create date range
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            start_date = (target_date - timedelta(days=15)).strftime("%Y-%m-%d")
            end_date = (target_date + timedelta(days=15)).strftime("%Y-%m-%d")
        except Exception as e:
            logger.error(f"Error parsing date: {e}")
            start_date = date_str
            end_date = date_str

        # Search for satellite image
        logger.info(f"Searching for satellite image between {start_date} and {end_date}")
        image = search_satellite_image(polygon, start_date, end_date)
        if image is None:
            logger.error("No suitable satellite image found")
            return None, None

        # Generate NDVI data
        logger.info("Calculating NDVI...")
        ndvi_bands = select_ndvi_bands(image)
        if ndvi_bands is None:
            logger.error("Failed to select NDVI bands")
            return None, None

        ndvi_image = calculate_ndvi(ndvi_bands)
        if ndvi_image is None:
            logger.error("Failed to calculate NDVI")
            return None, None

        logger.info("Exporting NDVI data...")
        ndvi_data = export_image_data(ndvi_image, polygon, scale=10, band_names=['NDVI'])
        if ndvi_data is None:
            logger.error("Failed to export NDVI data")
            return None, None

        # Extract NDVI values
        if hasattr(ndvi_data, 'dtype') and ndvi_data.dtype.names is not None:
            if 'NDVI' in ndvi_data.dtype.names:
                ndvi_values = ndvi_data['NDVI']
            else:
                field_name = ndvi_data.dtype.names[0]
                ndvi_values = ndvi_data[field_name]
        else:
            ndvi_values = ndvi_data

        # Ensure proper data type
        if ndvi_values.dtype != np.float32:
            ndvi_values = ndvi_values.astype(np.float32)

        # Get sensor data (now using only first band per sensor)
        logger.info("Fetching sensor data...")
        sensor_image = get_sensor_data(polygon)
        if sensor_image is None:
            logger.error("Failed to get sensor data")
            return None, None

        logger.info("Exporting sensor data...")
        sensor_data = export_sensor_data(sensor_image, polygon, scale=10)
        if sensor_data is None:
            logger.error("Failed to export sensor data")
            return None, None

        # Prepare sensor data as 3D array (full size, no crop)
        if hasattr(sensor_data, 'dtype') and sensor_data.dtype.names is not None:
            band_names = sensor_data.dtype.names
            sensor_bands = []
            for band_name in band_names:
                band_data = sensor_data[band_name]
                if len(band_data.shape) == 2:
                    band_3d = np.expand_dims(band_data, axis=2)
                else:
                    band_3d = band_data
                sensor_bands.append(band_3d)
            sensor_3d = np.concatenate(sensor_bands, axis=2)
        else:
            sensor_3d = sensor_data

        logger.info(f"✅ Successfully generated NDVI data with shape: {ndvi_values.shape}")
        logger.info(f"✅ Successfully generated sensor data with shape: {sensor_3d.shape}")

        return ndvi_values, sensor_3d

    except Exception as e:
        logger.error(f"Error generating NDVI and Sensor data: {e}")
        return None, None

def create_yield_heatmap_overlay(ndvi_data, predicted_yield, t1=30, t2=50):
    """
    Create a heatmap overlay with red, yellow, and green masks based on predicted yield.
    Uses NDVI as base image and applies color coding based on yield thresholds.

    Args:
        ndvi_data: 2D NDVI array
        predicted_yield: Predicted yield value (float)
        t1: Threshold 1 for low yield (default: 30)
        t2: Threshold 2 for high yield (default: 50)

    Returns:
        RGBA numpy array for PNG overlay
    """
    try:
        # Ensure NDVI is float
        nd = np.array(ndvi_data, dtype=float)
        if nd.ndim == 3 and nd.shape[2] == 1:
            nd = nd[..., 0]
        if nd.ndim != 2:
            nd = np.squeeze(nd)
        h, w = nd.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)  # default transparent

        # Mask valid NDVI
        valid_mask = np.isfinite(nd)
        if not np.any(valid_mask):
            return rgba
        # Thresholds
        v1 = t1
        v2 = t2
        # Classify
        low_mask = valid_mask & (nd < v1)
        mid_mask = valid_mask & (nd >= v1) & (nd < v2)
        high_mask = valid_mask & (nd >= v2)
        alpha_val = 200  # overlay alpha
        # Pure Red for low yield
        rgba[low_mask, 0] = 255   # R
        rgba[low_mask, 1] = 0     # G
        rgba[low_mask, 2] = 0     # B
        rgba[low_mask, 3] = alpha_val
        # Pure Yellow for mid yield
        rgba[mid_mask, 0] = 255
        rgba[mid_mask, 1] = 255
        rgba[mid_mask, 2] = 0
        rgba[mid_mask, 3] = alpha_val
        # Pure Green for high yield
        rgba[high_mask, 0] = 0
        rgba[high_mask, 1] = 255
        rgba[high_mask, 2] = 0
        rgba[high_mask, 3] = alpha_val
        return rgba
    except Exception as e:
        logger.error(f"Error creating yield heatmap overlay: {e}")
        return None

def create_separate_yield_masks(ndvi_data, predicted_yield, t1=30, t2=50):
    """
    Create 3 separate yield masks (red, yellow, green) based on NDVI thresholds.
    Each mask contains only one color, with other areas transparent.

    Args:
        ndvi_data: 2D NDVI array
        predicted_yield: Predicted yield value (float)
        t1: Threshold 1 for low yield (default: 30)
        t2: Threshold 2 for high yield (default: 50)

    Returns:
        Tuple of (red_mask, yellow_mask, green_mask, pixel_counts)
        Each mask is an RGBA numpy array
    """
    try:
        # Ensure NDVI is float
        nd = np.array(ndvi_data, dtype=float)
        if nd.ndim == 3 and nd.shape[2] == 1:
            nd = nd[..., 0]
        if nd.ndim != 2:
            nd = np.squeeze(nd)
        h, w = nd.shape
        
        # Create empty masks for each color
        red_mask = np.zeros((h, w, 4), dtype=np.uint8)
        yellow_mask = np.zeros((h, w, 4), dtype=np.uint8)
        green_mask = np.zeros((h, w, 4), dtype=np.uint8)

        # Mask valid NDVI pixels
        valid_mask = np.isfinite(nd)
        if not np.any(valid_mask):
            return red_mask, yellow_mask, green_mask, {"valid": 0, "red": 0, "yellow": 0, "green": 0}

        # Apply thresholds to classify pixels
        v1 = t1
        v2 = t2
        
        # Classify pixels based on NDVI values
        low_mask = valid_mask & (nd < v1)    # Red: low NDVI
        mid_mask = valid_mask & (nd >= v1) & (nd < v2)  # Yellow: medium NDVI
        high_mask = valid_mask & (nd >= v2)  # Green: high NDVI
        
        alpha_val = 255  # Full opacity for visible pixels
        
        # Red mask - only red pixels visible
        red_mask[low_mask, 0] = 255   # R
        red_mask[low_mask, 1] = 0     # G
        red_mask[low_mask, 2] = 0     # B
        red_mask[low_mask, 3] = alpha_val  # A
        
        # Yellow mask - only yellow pixels visible
        yellow_mask[mid_mask, 0] = 255   # R
        yellow_mask[mid_mask, 1] = 255   # G
        yellow_mask[mid_mask, 2] = 0     # B
        yellow_mask[mid_mask, 3] = alpha_val  # A
        
        # Green mask - only green pixels visible
        green_mask[high_mask, 0] = 0     # R
        green_mask[high_mask, 1] = 255   # G
        green_mask[high_mask, 2] = 0     # B
        green_mask[high_mask, 3] = alpha_val  # A
        
        # Calculate pixel counts correctly
        valid_pixels = int(np.sum(valid_mask))
        red_pixels = int(np.sum(low_mask))
        yellow_pixels = int(np.sum(mid_mask))
        green_pixels = int(np.sum(high_mask))
        
        pixel_counts = {
            "valid": valid_pixels,
            "red": red_pixels,
            "yellow": yellow_pixels,
            "green": green_pixels
        }
        
        logger.info(f"Created separate masks - Red: {red_pixels}, Yellow: {yellow_pixels}, Green: {green_pixels}, Valid: {valid_pixels}")
        
        return red_mask, yellow_mask, green_mask, pixel_counts
        
    except Exception as e:
        logger.error(f"Error creating separate yield masks: {e}")
        return None, None, None, None

def generate_farmer_suggestions(predicted_yield, old_yield, pixel_counts, sensor_data, location_info, thresholds):
    """
    Generate simple and easy-to-understand farming suggestions
    based on yield, NDVI colors, soil data, and location.
    """
    try:
        suggestions = {
            "overall_assessment": "",
            "yield_analysis": {},
            "field_management": [],
            "soil_recommendations": [],
            "immediate_actions": [],
            "seasonal_planning": [],
            "risk_alerts": []
        }

        # --- Basic calculations ---
        total_pixels = pixel_counts.get('valid', 1)
        red = (pixel_counts.get('red', 0) / total_pixels) * 100
        yellow = (pixel_counts.get('yellow', 0) / total_pixels) * 100
        green = (pixel_counts.get('green', 0) / total_pixels) * 100

        yield_change = predicted_yield - old_yield
        yield_change_percent = (yield_change / old_yield) * 100 if old_yield > 0 else 0

        # --- Overall condition ---
        if yield_change_percent > 10:
            suggestions["overall_assessment"] = "🌟 Excellent! This season’s yield looks very promising."
        elif yield_change_percent > 0:
            suggestions["overall_assessment"] = "✅ Good! Your field is doing better than last year."
        elif yield_change_percent > -10:
            suggestions["overall_assessment"] = "⚠️ Average. Some areas need improvement."
        else:
            suggestions["overall_assessment"] = "🚨 Poor condition. Immediate action is required."

        # --- Yield details ---
        suggestions["yield_analysis"] = {
            "predicted_yield": round(predicted_yield, 2),
            "previous_yield": round(old_yield, 2),
            "yield_change": round(yield_change, 2),
            "yield_change_percent": round(yield_change_percent, 1),
            "status": "Better" if yield_change > 0 else "Lower" if yield_change < 0 else "Same"
        }

        # --- Field management based on NDVI ---
        fm = []
        if red > 30:
            fm.extend([
                "🔴 More than 30% of your field shows crop stress.",
                "Irrigate these areas as soon as possible.",
                "Do a soil test and apply nitrogen-rich fertilizer."
            ])
            suggestions["risk_alerts"].append("⚠️ High stress areas detected - take quick action.")
        elif red > 15:
            fm.extend([
                "🔴 Some areas are stressed.",
                "Check irrigation and fertilizer in those zones."
            ])

        if yellow > 40:
            fm.extend([
                "🟡 Medium zones need attention.",
                "Balance water and fertilizer evenly across the field."
            ])

        if green > 60:
            fm.extend([
                "🟢 Great! Most of your field looks healthy.",
                "Keep following your current farming practices."
            ])
        elif green < 30:
            fm.append("🟢 Very few healthy areas — full-field improvement is needed.")

        suggestions["field_management"] = fm

        # --- Soil recommendations ---
        soil_reco = []
        if sensor_data is not None and sensor_data.size > 0 and sensor_data.shape[-1] >= 5:
            avg_ece = np.nanmean(sensor_data[..., 0])
            avg_n = np.nanmean(sensor_data[..., 1])
            avg_p = np.nanmean(sensor_data[..., 2])
            avg_ph = np.nanmean(sensor_data[..., 3])
            avg_oc = np.nanmean(sensor_data[..., 4])

            # Soil pH
            if avg_ph < 6:
                soil_reco.append("🧪 Soil is acidic — apply lime to improve pH.")
            elif avg_ph > 8:
                soil_reco.append("🧪 Soil is alkaline — use gypsum or organic manure.")
            else:
                soil_reco.append("🧪 Soil pH is in good condition.")

            # Nutrients
            if avg_n < 0.3:
                soil_reco.append("🌿 Nitrogen is low — apply urea or ammonium sulfate.")
            if avg_p < 0.3:
                soil_reco.append("🦴 Phosphorus is low — apply DAP or super phosphate.")
            if avg_oc < 0.3:
                soil_reco.append("🍃 Organic matter is low — add compost or crop residues.")
            if avg_ece > 2:
                soil_reco.append("🧂 High soil salinity — improve drainage and apply gypsum.")

        if not soil_reco:
            soil_reco.append("📊 Soil looks balanced — keep following current practices.")

        suggestions["soil_recommendations"] = soil_reco

        # --- Immediate actions ---
        actions = []
        if red > 20:
            actions.extend([
                "💧 Give water immediately to red (stressed) zones.",
                "🔬 Do a soil test in those areas."
            ])
        if yellow > 50:
            actions.append("🎯 Manage fertilizer and water properly in yellow zones.")
        if yield_change_percent < -15:
            actions.append("🚨 Yield dropping — consult your local agriculture officer.")
        if not actions:
            actions.append("✅ No urgent action required — continue regular monitoring.")

        suggestions["immediate_actions"] = actions

        # --- Seasonal planning ---
        sp = []
        loc = location_info.get("complete_address", "").lower()
        dist = location_info.get("district", "").lower()

        if "punjab" in loc or "punjab" in dist:
            sp.extend(["🌾 Punjab: Plan better wheat-rice rotation.", "💧 Prepare for water management before monsoon."])
        elif "haryana" in loc or "haryana" in dist:
            sp.extend(["🌾 Haryana: Use heat-tolerant crop varieties.", "💧 Adopt drip irrigation for water saving."])
        elif "uttar pradesh" in loc or "up" in dist:
            sp.extend(["🌾 UP: Practice crop rotation and pest management.", "🐛 Keep an eye on insects and diseases."])

        sp.extend(["📅 Maintain yield records.", "🌱 Try intercropping to improve productivity."])
        suggestions["seasonal_planning"] = sp

        # --- Risk alerts ---
        risks = []
        if red > 25:
            risks.append("🚨 High stress detected — irrigate and check soil health.")
        if yield_change_percent < -20:
            risks.append("📉 Major yield loss expected — take quick measures.")
        if green < 25:
            risks.append("⚠️ Very few healthy areas — needs improvement.")
        if not risks:
            risks.append("✅ No major problems detected — field is in good condition.")

        suggestions["risk_alerts"] = risks

        return suggestions

    except Exception as e:
        return {"error": f"Error while generating suggestions: {str(e)}"}

    """
    Generate comprehensive farming suggestions based on yield analysis, NDVI distribution, 
    soil conditions, and location data.
    """
    try:
        suggestions = {
            "overall_assessment": "",
            "yield_analysis": {},
            "field_management": [],
            "soil_recommendations": [],
            "immediate_actions": [],
            "seasonal_planning": [],
            "risk_alerts": []
        }
        
        # Calculate key metrics
        total_pixels = pixel_counts.get('valid', 1)
        red_percentage = (pixel_counts.get('red', 0) / total_pixels) * 100
        yellow_percentage = (pixel_counts.get('yellow', 0) / total_pixels) * 100
        green_percentage = (pixel_counts.get('green', 0) / total_pixels) * 100
        
        yield_change = predicted_yield - old_yield
        yield_change_percent = (yield_change / old_yield) * 100 if old_yield > 0 else 0
        
        # Overall Assessment
        if yield_change_percent > 10:
            suggestions["overall_assessment"] = "🌟 Excellent! Your field shows strong potential for above-average yields."
        elif yield_change_percent > 0:
            suggestions["overall_assessment"] = "✅ Good! Your field is performing better than historical averages."
        elif yield_change_percent > -10:
            suggestions["overall_assessment"] = "⚠️ Moderate performance. Some areas need attention for optimal yields."
        else:
            suggestions["overall_assessment"] = "🚨 Concerning. Immediate intervention needed to improve yields."
        
        # Yield Analysis
        suggestions["yield_analysis"] = {
            "predicted_yield": round(predicted_yield, 3),
            "historical_yield": round(old_yield, 3),
            "yield_change": round(yield_change, 3),
            "yield_change_percent": round(yield_change_percent, 2),
            "performance_status": "Above Average" if yield_change > 0 else "Below Average" if yield_change < 0 else "Average"
        }
        
        # Field Management based on NDVI zones
        field_management = []
        
        if red_percentage > 30:
            field_management.extend([
                "🔴 Critical: 30%+ of your field shows stressed vegetation",
                "Priority irrigation needed in red zones",
                "Consider immediate soil testing for nutrient deficiencies",
                "Apply nitrogen-rich fertilizers to stressed areas"
            ])
            suggestions["risk_alerts"].append("High stress zones detected - immediate action required")
        elif red_percentage > 15:
            field_management.extend([
                "🔴 Attention: Some areas show vegetation stress",
                "Monitor irrigation in red zones closely",
                "Consider targeted fertilizer application"
            ])
        
        if yellow_percentage > 40:
            field_management.extend([
                "🟡 Moderate zones need attention for optimal growth",
                "Implement precision farming techniques",
                "Consider variable rate fertilizer application"
            ])
        
        if green_percentage > 60:
            field_management.extend([
                "🟢 Excellent! Major portions showing healthy vegetation",
                "Maintain current practices in high-performing areas",
                "Use these areas as reference for field management"
            ])
        elif green_percentage < 30:
            field_management.append("🟢 Limited healthy zones - comprehensive field improvement needed")
        
        suggestions["field_management"] = field_management
        
        # Soil Recommendations based on sensor data analysis
        soil_recommendations = []
        
        if sensor_data is not None and sensor_data.size > 0:
            # Analyze soil parameters (assuming order: ECe, N, P, pH, OC)
            if sensor_data.shape[-1] >= 5:
                avg_ece = np.nanmean(sensor_data[..., 0])  # Electrical Conductivity
                avg_n = np.nanmean(sensor_data[..., 1])    # Nitrogen
                avg_p = np.nanmean(sensor_data[..., 2])    # Phosphorus
                avg_ph = np.nanmean(sensor_data[..., 3])   # pH
                avg_oc = np.nanmean(sensor_data[..., 4])   # Organic Carbon
                
                # Normalize values to 0-1 range for analysis
                ece_norm = (avg_ece - np.nanmin(sensor_data[..., 0])) / (np.nanmax(sensor_data[..., 0]) - np.nanmin(sensor_data[..., 0]) + 1e-8)
                n_norm = (avg_n - np.nanmin(sensor_data[..., 1])) / (np.nanmax(sensor_data[..., 1]) - np.nanmin(sensor_data[..., 1]) + 1e-8)
                p_norm = (avg_p - np.nanmin(sensor_data[..., 2])) / (np.nanmax(sensor_data[..., 2]) - np.nanmin(sensor_data[..., 2]) + 1e-8)
                ph_norm = (avg_ph - np.nanmin(sensor_data[..., 3])) / (np.nanmax(sensor_data[..., 3]) - np.nanmin(sensor_data[..., 3]) + 1e-8)
                oc_norm = (avg_oc - np.nanmin(sensor_data[..., 4])) / (np.nanmax(sensor_data[..., 4]) - np.nanmin(sensor_data[..., 4]) + 1e-8)
                
                # pH recommendations (assuming normalized pH where 0.5 = neutral)
                if ph_norm < 0.3:
                    soil_recommendations.extend([
                        "🧪 Soil appears acidic - consider lime application to raise pH",
                        "Consider dolomitic limestone for calcium and magnesium"
                    ])
                elif ph_norm > 0.7:
                    soil_recommendations.extend([
                        "🧪 Soil appears alkaline - consider sulfur or organic matter application",
                        "Consider gypsum application for calcium without raising pH"
                    ])
                else:
                    soil_recommendations.append("🧪 Soil pH appears to be in good range")
                
                # Nutrient recommendations
                if n_norm < 0.4:
                    soil_recommendations.append("🌿 Nitrogen levels appear low - consider nitrogen-rich fertilizers (urea, ammonium sulfate)")
                elif n_norm > 0.8:
                    soil_recommendations.append("🌿 Nitrogen levels appear high - reduce nitrogen application, focus on phosphorus and potassium")
                
                if p_norm < 0.3:
                    soil_recommendations.append("🦴 Phosphorus levels appear low - consider DAP or single superphosphate")
                
                if oc_norm < 0.3:
                    soil_recommendations.extend([
                        "🍃 Organic matter appears low - incorporate crop residues and compost",
                        "Consider cover cropping to improve soil organic content"
                    ])
                
                # Salinity check
                if ece_norm > 0.7:
                    soil_recommendations.extend([
                        "🧂 High soil salinity detected - improve drainage",
                        "Consider salt-tolerant crop varieties",
                        "Apply gypsum to help leach salts"
                    ])
        
        if not soil_recommendations:
            soil_recommendations.append("📊 Soil parameters appear balanced - maintain current soil management practices")
        
        suggestions["soil_recommendations"] = soil_recommendations
        
        # Immediate Actions
        immediate_actions = []
        
        if red_percentage > 20:
            immediate_actions.extend([
                "💧 Prioritize irrigation in stressed areas (red zones)",
                "📱 Set up soil moisture monitoring",
                "🔬 Conduct detailed soil testing in problem areas"
            ])
        
        if yellow_percentage > 50:
            immediate_actions.extend([
                "🎯 Plan targeted fertilizer application for moderate zones",
                "📋 Schedule weekly field monitoring"
            ])
        
        if yield_change_percent < -15:
            immediate_actions.extend([
                "🚨 Emergency consultation with agricultural extension officer",
                "💊 Consider foliar feeding for quick nutrient uptake",
                "🔍 Investigate pest and disease issues"
            ])
        
        if not immediate_actions:
            immediate_actions.append("✅ No immediate critical actions required - continue regular monitoring")
        
        suggestions["immediate_actions"] = immediate_actions
        
        # Seasonal Planning
        seasonal_planning = []
        location_str = location_info.get('complete_address', '').lower()
        district = location_info.get('district', '').lower()
        
        # Location-specific recommendations
        if 'punjab' in location_str or 'punjab' in district:
            seasonal_planning.extend([
                "🌾 Punjab region: Consider wheat-rice rotation optimization",
                "💧 Plan for efficient water management during monsoon",
                "🚜 Schedule pre-monsoon soil preparation"
            ])
        elif 'haryana' in location_str or 'haryana' in district:
            seasonal_planning.extend([
                "🌾 Haryana region: Focus on heat-resistant crop varieties",
                "💧 Plan drip irrigation for water conservation"
            ])
        elif 'uttar pradesh' in location_str or 'up' in district:
            seasonal_planning.extend([
                "🌾 UP region: Consider diverse crop rotation",
                "🐛 Plan integrated pest management strategy"
            ])
        
        # General seasonal recommendations
        seasonal_planning.extend([
            "📅 Plan crop rotation to maintain soil health",
            "🌱 Consider intercropping for better land utilization",
            "📊 Set up yield monitoring and record keeping"
        ])
        
        suggestions["seasonal_planning"] = seasonal_planning
        
        # Risk Alerts
        risk_alerts = []
        
        if red_percentage > 25:
            risk_alerts.append("🚨 HIGH RISK: Significant crop stress detected")
        
        if yield_change_percent < -20:
            risk_alerts.append("📉 YIELD RISK: Substantial yield decline predicted")
        
        if green_percentage < 25:
            risk_alerts.append("⚠️ HEALTH RISK: Limited healthy vegetation zones")
        
        suggestions["risk_alerts"] = risk_alerts if risk_alerts else ["✅ No major risks detected"]
        
        logger.info(f"Generated {len(field_management + soil_recommendations + immediate_actions)} recommendations for farmer")
        return suggestions
        
    except Exception as e:
        logger.error(f"Error generating farmer suggestions: {e}")
        return {
            "overall_assessment": "Unable to generate assessment",
            "yield_analysis": {},
            "field_management": [],
            "soil_recommendations": [],
            "immediate_actions": [],
            "seasonal_planning": [],
            "risk_alerts": ["Error generating recommendations"]
        }