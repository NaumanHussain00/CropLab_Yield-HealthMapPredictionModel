# 🚀 Deployment Guide for Render

This guide will help you deploy the Crop Yield Prediction API to Render.

## 📋 Pre-deployment Checklist

1. ✅ Code is ready for production
2. ✅ Environment variables are configured
3. ✅ Dependencies are listed in requirements.txt
4. ✅ Dockerfile is available (optional)
5. ✅ Service account credentials are secured

## 🌍 Environment Variables Setup

You need to set these environment variables in Render:

### Required Google Earth Engine Credentials:

```
GEE_SERVICE_ACCOUNT_EMAIL=farm-monitoring-service@pk07007.iam.gserviceaccount.com
GEE_PROJECT_ID=pk07007
GEE_PRIVATE_KEY_ID=a1967074c8dfb1aa1502222cde67f755938a9a6e
GEE_CLIENT_ID=109079035971556167013
GEE_CLIENT_CERT_URL=https://www.googleapis.com/robot/v1/metadata/x509/farm-monitoring-service%40pk07007.iam.gserviceaccount.com
```

### Private Key (IMPORTANT):

```
GEE_PRIVATE_KEY=-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDDFDQ0t/P9X77V
f5T589xQ2C4l1NteusBzklnHNjzswSHIb7xf/HbathfWb7MtF+GHFSPl8l3hEf9q
JXymIeeGwVk1vrGA4GPhwJCnTicTXvv49PE4EvJFhEJhqLGZLBM5CFyw9XxoJdHk
lVIFOab7T2T8TOjfv8BUbuiidihRgwcu/uydf9bEUMW2PeUIyQu4hlO8ptgaP10x
lMmVYiMY1jsUgXYrjbUTno8GKLuGVWnMUKEHJN951bJJi+qfBHzf+eiGUXZzisga
nBEqtF+xKHN8TpvhoLBolpmKwt4HpJnC88FeAPmNcpooAtWOeAKLxiHjOLA7zaqG
FemVZDWBAgMBAAECggEADYNLw88Y6UHa7+Jg3m45olFR47BP6yBCjn7iayiNpdS8
U1cHacViMkTkzSd69jJY8/N57hwQzAi5dDqD3OTXqypKKgKQkch5WBJFCDcjVnXW
PGjd1gWHSgYfAbF0GT5yrRkFhViBNUb8DWUZLmAk+fQw9EJw6yGBisJiqPDzBVpQ
J5cHf56Gkf+NKCKKr+H2mRXpIbvtYgML4uCe6D8/jr6jqQJBbw6wjpNytMQSq/3m
HgcpyLtpux62DiLQCmFlQ8w+0ufO7gJDi8wiaiwa9CEEp0Gg2Ixm3yb73HqRAllN
gFjTIOHWdq3R7S/gD77vWeuj1QYs44YkLuZ20hFNMQKBgQDuLFzYOpwN07iL0gSm
4LtaibCpHymUZZr5kbZtuV9g8bk/9+oopKar1FOuBAfgMwPvKxh7x0Hjc/95Acor
zo6E00o2LeRH16y5gIZwC/diOMtREts2PBh916VjU8qPYh6uwpf/6sTq0FCCXsgF
ibBCQyCIFqQDzAdPAxy31NeKLQKBgQDRrhw4I8Rmzq1wg+eUJhl4Pkauo5hYQSV0
+oltFc25OXePl6SHqwYcZ8ziPKNNzBPMjWj0LQDKJHOQ2LqS2vtC9u2KzmZsoaCT
1V7LkNiPARPzO+08NRxCOwv/Q1VOsF7PLrGwUtR74lrFJyIeJIdwr3y7DTlhUiYN
VNiT+blRJQKBgQDAl+kABOzaFYGettaCJyn5PWw4ZfWWhU77TsPpGEQzPWRigNO9
x97rBcgg6CuCNV6SERQ8S1VRWySTsknMgLwoVj8lbpixK4sFehO8GUETEP//8DKJ
ObFIWY/osQIUTfCnur10+WAAzTC9K22tZLi97FArG7vHQj4Ku2aGlLsyhQKBgEW+
eyZ1RzPHFUuypEnT7m7fkBUw11CrwrDJUQLW2Mn+gnVhxFlYGr1CDZVHBC6xbfiB
JOLrQTL7svEAFfcZHbBlgBfGla0WidoCg/iEInRWsHMcgMmhBNhG2bO7itmUssSJ
TJNQydq3LOgdHy0Vi3OA+6UgPfDQKVYP3cZCr1B9AoGAVjv00q+fOY8jdhjYNtQC
iGyCxIkJvYQckwVnYz4tBrMCvpHOAK5mXFNzTQ6w2Zr803UFQKXtiKSprI5Piym/
5N8EulSjVtOocuEYOKEfsKG2pzWjP5LkvJyUyPWvySQYsdWxST4oaBbIWhoq2Rzo
LD7JFIMAWJ7Yb/e658+laNc=
-----END PRIVATE KEY-----
```

⚠️ **IMPORTANT**: When setting GEE_PRIVATE_KEY in Render, copy the ENTIRE private key including the BEGIN and END lines.

## 📦 Deployment Steps

### Option 1: Direct Git Deploy (Recommended)

1. **Push to GitHub**:

   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Create Render Service**:

   - Go to [render.com](https://render.com)
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Choose the repository with your code

3. **Configure Service**:

   - **Name**: `croplab-ml-api`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

4. **Set Environment Variables**:

   - Go to "Environment" tab
   - Add all the GEE variables listed above
   - ⚠️ Make sure to paste the private key exactly as shown

5. **Deploy**:
   - Click "Create Web Service"
   - Wait for deployment to complete (~5-10 minutes)

### Option 2: Docker Deploy

If you prefer Docker:

1. **Build and test locally**:

   ```bash
   docker build -t croplab-ml-api .
   docker run -p 8000:8000 --env-file .env croplab-ml-api
   ```

2. **Deploy to Render**:
   - Use "Dockerfile" in Render settings
   - Set environment variables as above

## 🔍 Testing Deployment

Once deployed, test your endpoints:

1. **Health Check**:

   ```bash
   curl https://your-app-name.onrender.com/health
   ```

2. **Generate Heatmap**:
   ```bash
   curl -X POST https://your-app-name.onrender.com/generate_heatmap \
     -H "Content-Type: application/json" \
     -d '{
       "coordinates": [
         [74.0546207396482, 30.4382853694864],
         [74.0596207396482, 30.4382853694864],
         [74.0596207396482, 30.4432853694864],
         [74.0546207396482, 30.4432853694864],
         [74.0546207396482, 30.4382853694864]
       ],
       "t1": 3.0,
       "t2": 4.5
     }'
   ```

## 🚨 Troubleshooting

### Common Issues:

1. **FastAPI + Gunicorn Compatibility Error**:

   - **Error**: `FastAPI.__call__() missing 1 required positional argument: 'send'`
   - **Cause**: Gunicorn uses WSGI by default, but FastAPI needs ASGI
   - **Solution**: Updated Procfile to use uvicorn workers with gunicorn
   - **Fix**: `web: gunicorn app:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT`

2. **TensorFlow Model Loading Error**:

   - **Error**: `Unrecognized keyword arguments: ['batch_shape']`
   - **Cause**: Version mismatch between model (TF 2.13) and runtime (TF 2.15+)
   - **Solution**: Use exact TensorFlow version (2.13.0) that matches the model

3. **Scikit-learn Version Mismatch**:
   - **Error**: `InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from version 1.2.2 when using version 1.3.2`
   - **Solution**: Use exact scikit-learn version (1.2.2) that matches the scaler

### Common Issues:

1. **FastAPI + Gunicorn Compatibility Error**:

   - **Error**: `FastAPI.__call__() missing 1 required positional argument: 'send'`
   - **Cause**: FastAPI needs ASGI workers, not WSGI workers
   - **Solution**: Updated Procfile and render.yaml to use `uvicorn.workers.UvicornWorker`

2. **Scikit-learn Version Mismatch**:

   - **Error**: `InconsistentVersionWarning: Trying to unpickle estimator StandardScaler from version 1.2.2 when using version 1.7.2`
   - **Cause**: Model files were saved with older scikit-learn version
   - **Solution**: Pinned scikit-learn==1.2.2 in requirements.txt

3. **Python Version Compatibility Error**:

   - **Error**: `Cannot import 'setuptools.build_meta'` or similar build errors
   - **Cause**: Render defaults to Python 3.13, but some packages aren't compatible
   - **Solution**: The deployment is configured to use Python 3.11.9 via:
     - `PYTHON_VERSION=3.11.9` in render.yaml
     - `.python-version` file specifying 3.11.9
     - Updated requirements.txt with compatible versions

4. **Package Version Not Found**:

   - **Error**: `Could not find a version that satisfies the requirement earthengine-api==X.X.X`
   - **Cause**: Package version specified doesn't exist
   - **Solution**: Updated to use latest stable version (1.6.12)

5. **GEE Authentication Failed**:

   - Check that all environment variables are set correctly
   - Verify private key format (include BEGIN/END lines)
   - Ensure no extra spaces or newlines

6. **Build Fails**:

   - Check requirements.txt format
   - Ensure all dependencies are compatible with Python 3.11
   - Check build logs in Render dashboard

7. **App Crashes**:
   - Check application logs in Render
   - Verify model.h5 and scaler.save files are included
   - Test locally first

### Performance Tips:

1. **Cold Start**: First request might be slow (~30s) due to model loading
2. **Memory**: Ensure sufficient memory (recommendation: 1GB+)
3. **Timeout**: Set longer timeout for satellite data fetching

## 📊 Monitoring

1. **Health Endpoint**: Monitor `/health` for service status
2. **Logs**: Check Render logs for errors
3. **Metrics**: Monitor response times and error rates

## 🔐 Security

1. ✅ Service account credentials are in environment variables
2. ✅ No sensitive files in git repository
3. ✅ Non-root user in Docker container
4. ✅ HTTPS enabled by default on Render

Your API should now be live and ready for production use! 🎉
