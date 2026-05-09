
---
title: Car Recommender
emoji: ⚡
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
license: mit
---


<<<<<<< HEAD
# Automobile ML Recommender System

Analyze and recommend vehicles using machine learning-powered content-based filtering to provide personalized car suggestions based on user preferences.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Data Pipeline](#data-pipeline)
- [Machine Learning Architecture](#machine-learning-architecture)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Configuration](#configuration)
- [Authors](#authors)

---

## 📌 Overview

This project presents an **end-to-end ML-powered recommendation system** for automobile discovery and personalization. It leverages real-world vehicle data to deliver accurate, content-based car recommendations using **cosine similarity on engineered feature vectors**.

The solution integrates the complete machine learning lifecycle:
- **Data Collection**: Web scraping and data ingestion from multiple sources
- **Data Processing**: Cleaning, normalization, and feature transformation
- **Feature Engineering**: Extracting meaningful predictive indicators from raw vehicle attributes
- **Model Development**: Training scikit-learn pipelines with optimized transformers
- **Evaluation**: Cross-validation and performance metrics
- **Deployment**: Production-ready FastAPI server with Docker containerization
- **Monitoring**: Health checks and artifact reloading without downtime

The system enables **data-driven vehicle discovery**, helping users find cars matching their preferences through intelligent fuzzy matching and similarity-based recommendations.

---

## 📂 Project Structure

```
automobile-ML-recommender-system/
│
├── api/                          # FastAPI application
│   ├── app.py                    # Main API server (single-file production app)
│   └── README.md                 # API documentation
│
├── data/
│   ├── raw/                      # Original unprocessed data
│   │   ├── car_links.csv         # Vehicle image URLs and metadata
│   │   └── cars_data.csv         # Raw vehicle specifications
│   │
│   ├── interim/                  # Intermediate processing stages
│   │   └── cars_cleaned.parquet  # Cleaned and deduplicated data
│   │
│   └── processed/                # Final datasets for modeling
│       ├── pipeline.joblib       # Trained sklearn Pipeline (serialized)
│       └── feature_matrix.npz    # Sparse feature matrix (scipy CSR, 4928×1117)
│
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── 01_pakwheels_webscraping.py      # Web scraping data collection
│   ├── 02_pakwheels_html_to_dataframe.ipynb  # HTML parsing and extraction
│   ├── 03_Cars_Data_Cleaning.ipynb      # Data cleaning and preprocessing
│   ├── 04_EDA.ipynb                     # Exploratory data analysis
│   └── 05_Feature_Transformation_and_model.ipynb  # Feature engineering & training
│
├── src/                          # Reusable Python modules
│   ├── __init__.py
│   ├── components/
│   │   ├── data_cleaning.py      # Data preprocessing functions
│   │   ├── pipeline_builder.py   # sklearn Pipeline construction
│   │   └── train_pipeline.py     # Model training orchestration
│   ├── config/
│   │   └── config.py             # Configuration management
│   └── utils/
│       ├── common.py             # Helper utilities
│       └── logger.py             # Logging setup
│
├── models/                       # Trained model artifacts
│   └── (models stored via dvc/versioning)
│
├── reports/                      # Model performance reports and figures
│   └── figures/
│
├── Dockerfile                    # Multi-stage production Docker image
├── docker-compose.yml            # Docker Compose configuration
├── .dockerignore                 # Files excluded from Docker build
├── requirements.txt              # Python package dependencies (pinned versions)
├── params.yaml                   # Model hyperparameters
├── dvc.yaml                      # Data Version Control pipeline configuration
├── setup.py                      # Package installation configuration
├── Makefile                      # Build automation commands
├── tox.ini                       # Test environment configuration
├── LICENSE                       # Project license
└── README.md                     # This file
```

---

## 🎯 Key Features

### Recommendation Engine
- **Content-Based Filtering**: Recommends vehicles similar to user input using cosine similarity
- **Fuzzy Matching**: Intelligent query-to-vehicle name matching with configurable thresholds
- **Sparse Matrix Optimization**: Efficient similarity computation on 4928×1117 feature matrix

### Search & Discovery
- **Multi-field Search**: Full-text fuzzy search across name, brand, model, fuel type, transmission
- **Structured Filtering**: Advanced filtering by brand, price, year, fuel type, kilometers, transmission
- **Metadata Endpoints**: Quick access to unique brands, cities, fuel types

### Production-Ready API
- **Single-File Architecture**: All endpoints in one maintainable `app.py` file
- **Lifespan Context Manager**: Efficient artifact loading and caching at startup
- **Hot Reloading**: POST `/reload` endpoint to refresh models without server restart
- **Health Monitoring**: Comprehensive `/health` endpoint with artifact validation
- **Structured Responses**: Pydantic models for all API responses
- **Logging**: Detailed operation logging with timestamps

### Deployment
- **Multi-Stage Docker Build**: Optimized image with compiler-free runtime (~500MB)
- **Layer Caching**: Efficient builds with requirements.txt before code
- **Non-Root User**: Security-hardened container execution
- **Memory Limits**: 512MB memory constraint for horizontal scaling
- **Health Checks**: Automated container health monitoring with 60s startup grace period

---

## 🔄 Data Pipeline

```
Raw Data Collection
        ↓
    Web Scraping (notebooks/01_pakwheels_webscraping.py)
        ↓
    HTML Parsing & Extraction (notebooks/02_pakwheels_html_to_dataframe.ipynb)
        ↓
    Data Cleaning & Preprocessing (notebooks/03_Cars_Data_Cleaning.ipynb)
        ↓
    Interim Storage (data/interim/cars_cleaned.parquet)
        ↓
    Exploratory Data Analysis (notebooks/04_EDA.ipynb)
        ↓
    Feature Engineering & Transformation (notebooks/05_Feature_Transformation_and_model.ipynb)
        ↓
    sklearn Pipeline Training (src/components/pipeline_builder.py)
        ↓
    Feature Matrix Generation (data/processed/feature_matrix.npz)
        ↓
    Model Serialization (data/processed/pipeline.joblib)
        ↓
    API Deployment (api/app.py)
```

### Data Processing Steps

1. **Collection**: Web scraping from PakWheels to gather vehicle listings with images
2. **Extraction**: Parse HTML and extract structured vehicle data
3. **Cleaning**: 
   - Handle missing values
   - Remove duplicates and outliers
   - Standardize field formats (price, kilometers, engine size)
4. **Consolidation**: Merge data from multiple sources into unified dataset
5. **Feature Engineering**:
   - Categorical encoding (Brand, Fuel Type, Transmission, City)
   - Numerical scaling (Price, Kilometers, Engine, Model Year)
   - Feature interactions and derived indicators
6. **Matrix Generation**: Transform engineered features into sparse CSR matrix (4928×1117)
7. **Pipeline Serialization**: Save fitted sklearn Pipeline for reproducible predictions

---

## 🤖 Machine Learning Architecture

### Model Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Preprocessing** | sklearn.preprocessing | Standardization, encoding, scaling |
| **Feature Engineering** | pandas, numpy | Feature creation and transformation |
| **Similarity Metric** | cosine_similarity (sklearn) | Measure car similarity |
| **Serialization** | joblib | Pipeline persistence |
| **Matrix Format** | scipy.sparse.csr_matrix | Memory-efficient storage |

### Recommendation Workflow

```python
1. Input: User query (e.g., "Honda Civic turbo")
   ↓
2. Fuzzy Matching: Match query to existing car in dataset
   - Use rapidfuzz.process.extractOne with token_set_ratio
   - Score cutoff: 60 (configurable)
   ↓
3. Feature Extraction: Transform matched car using fitted pipeline
   - Apply same preprocessing as training data
   - Generate 1117-dimensional feature vector
   ↓
4. Similarity Computation: Calculate cosine similarity against all cars
   - sim(matched_car, all_cars) = cosine(feature_vector, feature_matrix)
   - Output: Similarity scores for each car (0-1 range)
   ↓
5. Ranking: Sort by similarity score descending
   - Exclude matched car itself
   - Return top_n results
   ↓
6. Output: List of recommendations with scores
```

### Key Metrics

- **Dataset Size**: 4,928 vehicles
- **Feature Dimensions**: 1,117 engineered features
- **Matrix Sparsity**: Sparse CSR format for memory efficiency
- **Search Latency**: <100ms for recommendations (single worker)
- **Null Links**: 30 records (handled gracefully)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Git & DVC (for versioning)
- Docker & Docker Compose (for containerized deployment)
- 512MB RAM (for model artifacts in memory)

### Local Installation

1. **Clone repository**
   ```bash
   git clone https://github.com/kamran-khalil/automobile-ML-recommender-system.git
   cd automobile-ML-recommender-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure data artifacts exist**
   ```
   data/processed/
   ├── pipeline.joblib
   ├── feature_matrix.npz
   └── cars_cleaned.parquet (or in data/interim/)
   ```

5. **Run local development server**
   ```bash
   cd api
   python app.py
   ```
   Server starts at `http://localhost:8000`

### Using Docker Compose

```bash
# Build and start containerized service
docker-compose up --build

# Service will be available at http://localhost:8000
# Health check runs automatically every 30 seconds

# View logs
docker-compose logs -f car-recommender

# Stop service
docker-compose down
```

---

## 📡 API Documentation

### Base URL
```
http://localhost:8000
```

### Authentication
Public endpoints (no authentication required)

### Response Format
All responses include proper HTTP status codes and Pydantic-validated JSON.

---

### Recommendation Endpoints

#### 1. **GET /recommend** - ML-Based Recommendations

Get cars most similar to your query using cosine similarity.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | string | Required | Car name or description (e.g., "honda civic turbo") |
| `top_n` | integer | 10 | Number of recommendations to return |
| `threshold` | integer | 60 | Fuzzy match threshold (0-100) |

**Request:**
```bash
curl -X GET "http://localhost:8000/recommend?q=honda%20civic&top_n=5&threshold=60"
```

**Response (200 OK):**
```json
{
  "query": "honda civic",
  "top_n": 5,
  "threshold": 60,
  "latency_ms": 45.32,
  "results": [
    {
      "name": "Honda Civic 2021 1.5 RS Turbo",
      "brand": "Honda",
      "city": "Karachi",
      "fuel_type": "Petrol",
      "transmission": "Automatic",
      "price": 2500000.0,
      "model": 2021,
      "kilometer": 50000.0,
      "engine": 1.5,
      "link": "https://cache1.pakwheels.com/...",
      "similarity_score": 0.92,
      "match_score": 95.0,
      "matched_input": "Honda Civic 2021 1.5 RS Turbo"
    }
  ]
}
```

**Error Responses:**
- `404 Not Found`: No matching car found for query
- `503 Service Unavailable`: Artifacts not loaded

---

#### 2. **GET /search** - Fuzzy Text Search

Search for cars using fuzzy text matching (no ML similarity).

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | string | Required | Search query |
| `top_n` | integer | 10 | Maximum results to return |
| `threshold` | integer | 50 | Match threshold (0-100) |

**Request:**
```bash
curl -X GET "http://localhost:8000/search?q=honda&top_n=10"
```

**Response (200 OK):**
```json
{
  "query": "honda",
  "results": [
    {
      "name": "Honda Civic 2021 1.5 RS Turbo",
      "brand": "Honda",
      "match_score": 95.0,
      "similarity_score": 0.0,
      "matched_input": "Honda Civic 2021 1.5 RS Turbo"
    }
  ]
}
```

**Error Responses:**
- `404 Not Found`: No results above threshold

---

#### 3. **GET /car** - Single Car Lookup

Get detailed information for a single car by fuzzy name match.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | Required | Car name to search |
| `threshold` | integer | 60 | Match threshold |

**Request:**
```bash
curl -X GET "http://localhost:8000/car?name=civic"
```

**Response (200 OK):**
```json
{
  "name": "Honda Civic 2021 1.5 RS Turbo",
  "brand": "Honda",
  "price": 2500000.0,
  "match_score": 90.5,
  "message": "Car found"
}
```

---

#### 4. **GET /filter** - Advanced Filtering

Filter cars by structured attributes (brand, price, year, fuel type, transmission, kilometers).

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `brand` | string | null | Brand (case-insensitive contains) |
| `city` | string | null | City (case-insensitive contains) |
| `fuel_type` | string | null | Fuel type (Petrol, Diesel, Hybrid, Electric) |
| `transmission` | string | null | Transmission (Automatic, Manual) |
| `year_min` | integer | null | Minimum model year |
| `year_max` | integer | null | Maximum model year |
| `price_min` | float | null | Minimum price |
| `price_max` | float | null | Maximum price |
| `km_max` | float | null | Maximum kilometers |
| `top_n` | integer | 20 | Maximum results |

**Request:**
```bash
curl -X GET "http://localhost:8000/filter?brand=Honda&fuel_type=Petrol&price_max=3000000&top_n=10"
```

**Response (200 OK):**
```json
{
  "filters_applied": {
    "brand": "Honda",
    "fuel_type": "Petrol",
    "price_max": 3000000
  },
  "total_found": 15,
  "results": [...]
}
```

**Sorting**: Results sorted by price (ascending)

---

### Metadata Endpoints

#### 5. **GET /brands**
Get sorted list of unique car brands.

```bash
curl -X GET "http://localhost:8000/brands"
```

**Response:**
```json
{
  "brands": ["Honda", "Toyota", "BMW", ...]
}
```

---

#### 6. **GET /cities**
Get sorted list of unique cities.

```bash
curl -X GET "http://localhost:8000/cities"
```

---

#### 7. **GET /fuel-types**
Get sorted list of unique fuel types.

```bash
curl -X GET "http://localhost:8000/fuel-types"
```

---

### Health & Admin Endpoints

#### 8. **GET /health** - Health Check

Check API status and artifact loading state.

**Request:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Response (200 OK - Artifacts Loaded):**
```json
{
  "status": "ok",
  "artifacts_loaded": true,
  "rows": 4928,
  "matrix_shape": [4928, 1117]
}
```

**Response (200 OK - Loading):**
```json
{
  "status": "loading",
  "artifacts_loaded": false,
  "rows": null,
  "matrix_shape": null
}
```

---

#### 9. **POST /reload** - Hot Reload Artifacts

Reload all models without restarting the server.

**Request:**
```bash
curl -X POST "http://localhost:8000/reload"
```

**Response (200 OK):**
```json
{
  "status": "reloaded",
  "rows": 4928
}
```

**Error Responses:**
- `500 Internal Server Error`: Reload failed (check logs)

---

### API Documentation UI

- **Swagger UI (Recommended)**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🐳 Docker Deployment

### Multi-Stage Build

The Dockerfile uses a two-stage build process:

1. **Builder Stage**: 
   - Installs `build-essential`, `gcc` for compilation
   - Installs all Python packages into `/install` prefix
   - Generates wheels and compiled extensions

2. **Runtime Stage**:
   - Minimal `python:3.11-slim` base image
   - Only includes `libgomp1` (scikit-learn OpenMP) and `curl` (healthcheck)
   - Copies compiled packages from builder
   - No compilers or build tools in final image
   - Result: ~500MB final image (vs ~1.5GB with full dependencies)

### Building Manually

```bash
# Build image
docker build -t car-recommender:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/data/processed:/app/data/processed:ro \
  -v $(pwd)/data/interim:/app/data/interim:ro \
  -e LOG_LEVEL=info \
  car-recommender:latest
```

### Using Docker Compose (Recommended)

```bash
# Start service
docker-compose up --build

# Stop service
docker-compose down

# View logs
docker-compose logs -f

# Scale horizontally (requires load balancer)
docker-compose up -d --scale car-recommender=3
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_HOST` | 0.0.0.0 | Bind address |
| `APP_PORT` | 8000 | API port |
| `APP_WORKERS` | 1 | Uvicorn worker processes |
| `LOG_LEVEL` | info | Logging level (debug, info, warning, error) |

### Volume Mounts

| Path | Container Path | Mode | Purpose |
|------|-----------------|------|---------|
| `./data/processed` | `/app/data/processed` | ro | Feature matrix & pipeline |
| `./data/interim` | `/app/data/interim` | ro | Cleaned dataset |

### Memory Management

- Container memory limit: **512MB** (adjustable in docker-compose.yml)
- Feature matrix in-memory: ~200MB (sparse CSR)
- DataFrame in-memory: ~150MB (4928 rows × 40 columns)
- Single worker recommended per container

---

## ⚙️ Configuration

### Parameters (params.yaml)

```yaml
model:
  type: "sklearn_pipeline"
  hyperparameters:
    # Configure based on notebook experiments
    
features:
  n_features: 1117
  matrix_format: "scipy_sparse_csr"

api:
  default_top_n: 10
  default_threshold: 60
  workers: 1
  log_level: "info"
```

### Environment Variables (.env)

```bash
# API Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
APP_WORKERS=1
LOG_LEVEL=info

# Data Paths (optional, defaults to workspace paths)
# PARQUET_PATH=/custom/path/cars_cleaned.parquet
# PIPELINE_PATH=/custom/path/pipeline.joblib
# MATRIX_PATH=/custom/path/feature_matrix.npz
```

---

## 📊 Model Performance

### Recommendation Quality

- **Cosine Similarity**: Measures feature-space distance
- **Precision@10**: Percentage of top-10 results that are relevant
- **Coverage**: Percentage of catalog recommendable (active learning)
- **Cold Start**: Fuzzy matching handles new or misspelled queries

### API Performance

- **Recommendation Latency**: 30-100ms (depending on top_n)
- **Search Latency**: 20-50ms
- **Filter Latency**: 10-40ms
- **Throughput**: ~100 req/s (single worker, 512MB memory)

---

## 🔧 Development

### Running Notebooks

```bash
jupyter notebook notebooks/
```

**Execution Order:**
1. `01_pakwheels_webscraping.py` - Data collection
2. `02_pakwheels_html_to_dataframe.ipynb` - Data extraction
3. `03_Cars_Data_Cleaning.ipynb` - Preprocessing
4. `04_EDA.ipynb` - Exploration
5. `05_Feature_Transformation_and_model.ipynb` - Modeling

### Building from Scratch

```bash
# Run DVC pipeline
dvc repro

# Or manually:
python src/components/data_cleaning.py
python src/components/pipeline_builder.py
python src/components/train_pipeline.py
```

### Code Quality

```bash
# Formatting
make format

# Linting
make lint

# Testing
make test
```

---

## 📦 Dependencies

See [requirements.txt](requirements.txt) for pinned versions:

- **fastapi** 0.104.1 - Web framework
- **uvicorn** 0.24.0 - ASGI server
- **scikit-learn** 1.3.2 - ML pipelines
- **pandas** 2.1.3 - Data manipulation
- **numpy** 1.26.2 - Numerical computing
- **scipy** 1.11.4 - Sparse matrices
- **rapidfuzz** 3.4.0 - Fuzzy matching
- **joblib** 1.3.2 - Model serialization
- **dask** 2023.12.0 - Large dataframe processing
- **pyarrow** 13.0.0 - Parquet format

---

## 🐛 Troubleshooting

### Container fails to start
**Error**: `ModuleNotFoundError: No module named 'rapidfuzz'`
**Solution**: Rebuild image with `docker-compose up --build`

### Slow recommendations
**Cause**: Multiple workers competing for sparse matrix memory
**Solution**: Keep `APP_WORKERS=1`, scale horizontally with load balancer

### `/health` returns 503
**Cause**: Artifacts still loading or missing
**Solution**: Check docker logs, verify mount paths, wait 60s for startup

### High memory usage
**Cause**: Feature matrix or dataframe not cleared
**Solution**: POST `/reload` endpoint to refresh, or restart container

---

## 📝 Logging

Logs include:
- Artifact loading lifecycle (startup, errors, row counts)
- Every fuzzy match: matched car name, image URL, score
- Every recommendation request: query, result count, latency
- Every API error with full traceback

**Format:**
```
22:34:21  INFO      Starting artifact load
22:34:22  INFO      Artifacts loaded: 4928 rows, matrix=(4928, 1117), null_links=30
22:34:23  INFO      Recommend: query='honda civic', results=5, latency=45.2ms
```

---

## 🔐 Security

- **Non-root user**: API runs as uid 1001, no login shell
- **Read-only data**: Volumes mounted as `:ro` (read-only)
- **No credentials in image**: Environment variables or .env files only
- **Input validation**: Pydantic models validate all parameters
- **Error messages**: No sensitive data leakage in responses

---

## 🚢 Production Deployment

### Recommended Setup

```
                    ┌─────────────┐
                    │ Load Balancer
                    │ (nginx/HAProxy)
                    └────────┬────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
        ┌───▼──┐         ┌───▼──┐         ┌──▼───┐
        │  Pod │         │  Pod │         │ Pod  │
        │ :8000│         │ :8000│         │:8000 │
        └──────┘         └──────┘         └──────┘
            │                │                │
            └────────────────┼────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Shared NFS/EBS  │
                    │ /data/processed │
                    │ /data/interim   │
                    └─────────────────┘
```

### Kubernetes Example

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: car-recommender
spec:
  replicas: 3
  selector:
    matchLabels:
      app: car-recommender
  template:
    metadata:
      labels:
        app: car-recommender
    spec:
      containers:
      - name: api
        image: car-recommender:latest
        ports:
        - containerPort: 8000
        env:
        - name: APP_WORKERS
          value: "1"
        - name: LOG_LEVEL
          value: "info"
        volumeMounts:
        - name: data
          mountPath: /app/data
          readOnly: true
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        resources:
          requests:
            memory: "256Mi"
          limits:
            memory: "512Mi"
      volumes:
      - name: data
        nfs:
          server: nfs-server.example.com
          path: /exports/car-recommender
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Authors

### Kamran Khan Orakzai
**Data Science & MLOps Engineer**

- GitHub: [@kamran-khalil](https://github.com/kamran-khalil)
- LinkedIn: [Kamran Khan Orakzai](https://linkedin.com/in/kamran-khan-orakzai)
- Email: [kamran@example.com](mailto:kamran@example.com)

### Khalil (Author)
**Machine Learning & Data Engineering**

- GitHub: [@khalil](https://github.com/khalil)
- LinkedIn: [Khalil](https://linkedin.com/in/khalil)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support

- **Issues**: Report bugs on [GitHub Issues](https://github.com/kamran-khalil/automobile-ML-recommender-system/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/kamran-khalil/automobile-ML-recommender-system/discussions)
- **Email**: kamran@example.com

---

## 🙏 Acknowledgments

- PakWheels for dataset access and inspiration
- scikit-learn community for excellent ML tools
- FastAPI framework for production-grade web development
- DVC and MLflow for ML lifecycle management

---

**Last Updated**: May 2026  
**Version**: 1.0.0
=======
