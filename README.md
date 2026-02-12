🐳 Docker Ready
TENSORFLOW DOES NOT WORK WITH PYTHON 3.14. MAKE SURE YOU HAVE 3.12 INSTALLED IN YOUR PATH TO RUN IT.

# Smart Ambulance - Anomaly Detection & Risk Scoring System

##  Project Overview

Real-time anomaly detection and risk scoring system for ambulance patient vitals using ensemble machine learning (Isolation Forest + One-Class SVM + LSTM Autoencoder).

**Key Features:**
- Multi-model ensemble for robust detection
- Clinical severity scoring with domain knowledge
- Motion artifact suppression
- Temporal persistence checking
- REST API for real-time predictions
- Explainable alerts with reasoning

---

##  Project Structure

```
smart-ambulance/
├── data/
│   ├── raw/                          # Original generated data
│   └── processed/                    # Cleaned, windowed data
├── src/
│   ├── generate_synthetic.py         # Synthetic data generation
│   ├── inject_artifacts.py           # Artifact injection
│   ├── clean_artifacts.py            # Artifact removal
│   ├── Eval and plots                # Plotting utilities folder
│   ├── preprocessing.py              #  Feature engineering
│   ├── inference.py                  #  Risk scoring logic
│   └── api.py                        #  FastAPI service
├── models/
│   ├── iso_forest.pkl                # Trained Isolation Forest + scaler
│   ├── ocsvm.pkl                     # Trained One-Class SVM
│   └── lstm_autoencoder.keras        # Trained LSTM Autoencoder
├── notebooks/
│   ├── 01_classical_ml.ipynb         # ISO + SVM training
│   ├── 02_deep_learning.ipynb        # LSTM training
│   └── 03_risk_scoring.ipynb         # Risk scoring pipeline
├── requirements.txt
├── test_api.py                       # API test suite
└── README.md
```

---

##  Quick Start

🐳 Run with Docker (Recommended)

This project is fully Dockerized for easy and reproducible setup.

Prerequisites

Install Docker Desktop: https://www.docker.com/products/docker-desktop/

Run the API (one command)
docker compose up

Build image manually (first time or after changes)
docker compose build
docker compose up

API Endpoints

Health check:

http://localhost:8000/health


Interactive API docs (Swagger UI):

http://localhost:8000/docs

Notes

Models are mounted from ./models

Data is mounted from ./data

No local Python or TensorFlow install required

### Alternative(local) 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd smart-ambulance

# Create virtual environment 
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the API

```bash
# Make sure models are in models/ directory
python src/api.py
```

The API will start at `http://localhost:8000`

**API Documentation:** http://localhost:8000/docs (Interactive Swagger UI)

### 3. Test the API

```bash
# In a new terminal
python test_api.py
```

---

##  API Usage

### Endpoint 1: Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "vitals": {
      "hr": [75, 76, 74, 75, 77, ...],      # 30 values
      "spo2": [98, 98, 97, 98, 98, ...],    # 30 values
      "sbp": [120, 121, 119, 120, 122, ...], # 30 values
      "dbp": [80, 80, 79, 80, 81, ...],     # 30 values
      "motion": [0.1, 0.2, 0.1, 0.15, ...]  # 30 values
    },
    "patient_id": "PATIENT_001"
  }'
```

**Response:**
```json
{
  "risk_score": 0.234,
  "risk_level": "NORMAL",
  "confidence": 0.92,
  "reasoning": [
    "High model agreement"
  ],
  "model_scores": {
    "isolation_forest": 0.15,
    "one_class_svm": 0.18,
    "lstm_autoencoder": 0.22
  },
  "timestamp": "2024-02-12T10:30:45.123456"
}
```

### Endpoint 2: Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "vitals_list": [
      {"hr": [...], "spo2": [...], ...},
      {"hr": [...], "spo2": [...], ...}
    ],
    "patient_id": "PATIENT_001"
  }'
```

### Endpoint 3: Health Check

```bash
curl http://localhost:8000/health
```

### Endpoint 4: Reset Session

```bash
curl -X POST http://localhost:8000/reset
```

---

##  Model Architecture

### Ensemble Approach

The system uses **three complementary models**:

1. **Isolation Forest** (30% weight)
   - Tree-based anomaly detection
   - Fast, interpretable
   - Good for point anomalies

2. **One-Class SVM** (30% weight)
   - Boundary-based detection
   - Captures complex patterns
   - Flexible with RBF kernel

3. **LSTM Autoencoder** (40% weight)
   - Sequence-based reconstruction
   - Best for temporal patterns
   - Detects gradual deterioration

**Final Score:**
```
ensemble_score = 0.30 × ISO + 0.30 × SVM + 0.40 × LSTM
risk_score = ensemble_score × severity × persistence × confidence
```

### Features (21 total)

| Category | Features |
|----------|----------|
| **Mean values** | hr_mean, spo2_mean, sbp_mean, dbp_mean |
| **Variability** | hr_std, spo2_std, sbp_std, dbp_std |
| **Trends** | hr_slope, spo2_slope, sbp_slope, dbp_slope |
| **Extremes** | hr_min, hr_max, spo2_min, spo2_max |
| **Correlation** | hr_spo2_corr |
| **Pulse Pressure** | pp_mean, pp_std |
| **Motion** | motion_mean, motion_max |

---

##  Risk Scoring Logic

### Clinical Severity Multipliers

| Condition | Multiplier | Example |
|-----------|-----------|---------|
| SpO2 < 88% | +2.0 | Critical oxygen desaturation |
| SpO2 < 92% | +1.0 | Low oxygen saturation |
| HR > 140 | +1.0 | Severe tachycardia |
| HR > 120 | +0.5 | Tachycardia |
| HR + SpO2 both abnormal | +1.5 | Multi-vital distress |
| Rapid HR increase | +0.5 | Early warning signal |

### Temporal Persistence

- Checks last 5 windows for sustained anomalies
- 4/5 anomalous → 1.5× multiplier
- 3/5 anomalous → 1.3× multiplier

### Motion Artifact Suppression

- High motion (>0.8) → 0.4× confidence
- SpO2 drop during motion → reduced confidence
- Alerts suppressed if motion + no severe vitals

### Risk Levels

```python
if risk_score > 1.5 and confidence > 0.6:
    return "CRITICAL"  # Immediate attention
elif risk_score > 0.8 and confidence > 0.5:
    return "WARNING"   # Monitor closely
else:
    return "NORMAL"    # Routine monitoring
```

---

##  Model Training

### 1. Data Generation

```bash
python src/generation.py
```

Generates 50 patients × 30 minutes of synthetic vitals (HR, SpO2, BP, motion)

### 2. Artifact Injection & Cleaning

```bash
python src/artifacts.py    # Add motion artifacts
python src/cleaner.py      # Remove artifacts
```

**Cleaning Performance:** Precision=0.58, Recall=0.94

### 3. Train Classical Models

Open `notebooks/01_classical_ml.ipynb` and run all cells.

**Outputs:**
- `models/iso_forest.pkl` (contains scaler + model)
- `models/ocsvm.pkl`

### 4. Train LSTM Autoencoder

Open `notebooks/02_deep_learning.ipynb` and run all cells.

**Output:**
- `models/lstm_autoencoder.keras`

### 5. Risk Scoring Pipeline

Open `notebooks/03_risk_scoring.ipynb` to:
- Combine all three models
- Generate risk scores
- Analyze alerts
- Produce summary report

---

## Testing

### Run Test Suite

```bash
python test_api.py
```

**Tests included:**
1. Health check
2. Normal vitals → NORMAL
3. Distress scenario → CRITICAL
4. Motion artifact → Suppressed
5. Gradual deterioration → Detected
6. Batch prediction
7. Session reset
8. Invalid input validation

### Expected Results

```
 Normal vitals: risk_level = NORMAL, confidence > 0.8
 distress: risk_level = CRITICAL, confidence > 0.7
 Motion artifact: Low confidence or suppressed
 Batch: All windows processed
 Validation: Invalid vitals rejected
```

---

##  Development

### Running Locally

```bash
# Start API
python src/api.py

# In another terminal - test
python test_api.py

# Or use interactive docs
open http://localhost:8000/docs
```

### Code Style

```bash
# Format code
black src/

# Run tests
pytest tests/
```

---

##  Configuration

### Adjusting Thresholds

Edit `src/inference.py`:

```python
# Line ~115: Clinical thresholds
if row['spo2_min'] < 88:  # Adjust critical SpO2 threshold
    severity += 2.0

# Line ~325: Risk level thresholds  
if risk_score > 1.5:      # Adjust CRITICAL threshold
    risk_level = 'CRITICAL'
```

### Changing Model Weights

Edit `src/inference.py`:

```python
# Line ~165: Ensemble weights
ensemble_score = (
    0.30 * iso_score +     # Increase if ISO performs better
    0.30 * svm_score +     # Increase if SVM performs better
    0.40 * lstm_score      # Increase if LSTM performs better
)
```

---

##  Performance Metrics

### Alert Distribution (Expected)

```
Total Alerts: 3-5% of windows
  🔴 CRITICAL: 0.5-1.5%
  🟠 WARNING:  2-3.5%
  🟢 NORMAL:   95-97%
```

### False Positive Rate

Target: <5% (acceptable in ambulance context)

### Alert Latency

Target: <15 seconds (within one window)

---

##  Known Limitations

1. **LSTM requires sequences**: First 9 windows use median score
2. **Motion artifacts**: Not 100% suppressed, but flagged with low confidence
3. **Patient-specific baselines**: System uses population averages
4. **No integration with medical devices**: Currently accepts manual input

---

##  Future Improvements

- [ ] Patient-specific baseline calibration
- [ ] Long-term trend analysis (5-10 minute windows)
- [ ] Integration with real medical devices (ECG, pulse oximeter)
- [ ] Multi-patient monitoring dashboard
- [ ] Alert escalation logic (notify paramedic vs hospital)
- [ ] Model retraining pipeline

---

##  References

- Isolation Forest: Liu et al., 2008
- One-Class SVM: Schölkopf et al., 2001
- LSTM Autoencoders: Malhotra et al., 2015

---

##  Contributors

Yuvraj Singh -yuvraj17nov@gmail.com

---


##  Acknowledgments

- Gray mobility's Smart Ambulance Platform
- Clinical domain experts for vital sign thresholds


---

**Last Updated:** February 2026










TO CHECK IN SWAGGER- COPY THESE- {
  "vitals": {
    "hr":     [145, 147, 144, 146, 148, 145, 147, 144, 146, 148, 145, 147, 144, 146, 148, 145, 147, 144, 146, 148, 145, 147, 144, 146, 148, 145, 147, 144, 146, 148],
    "spo2":   [86, 85, 87, 86, 85, 86, 85, 87, 86, 85, 86, 85, 87, 86, 85, 86, 85, 87, 86, 85, 86, 85, 87, 86, 85, 86, 85, 87, 86, 85],
    "sbp":    [110, 111, 109, 110, 112, 110, 111, 109, 110, 112, 110, 111, 109, 110, 112, 110, 111, 109, 110, 112, 110, 111, 109, 110, 112, 110, 111, 109, 110, 112],
    "dbp":    [75, 75, 74, 75, 76, 75, 75, 74, 75, 76, 75, 75, 74, 75, 76, 75, 75, 74, 75, 76, 75, 75, 74, 75, 76, 75, 75, 74, 75, 76],
    "motion": [0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1]
  },
  "patient_id": "CRITICAL_TEST"
}

{
  "vitals": {
    "hr":     [78, 79, 77, 78, 80, 78, 79, 77, 78, 80, 78, 79, 77, 78, 80, 78, 79, 77, 78, 80, 78, 79, 77, 78, 80, 78, 79, 77, 78, 80],
    "spo2":   [89, 88, 90, 89, 88, 89, 88, 90, 89, 88, 89, 88, 90, 89, 88, 89, 88, 90, 89, 88, 89, 88, 90, 89, 88, 89, 88, 90, 89, 88],
    "sbp":    [120, 121, 119, 120, 122, 120, 121, 119, 120, 122, 120, 121, 119, 120, 122, 120, 121, 119, 120, 122, 120, 121, 119, 120, 122, 120, 121, 119, 120, 122],
    "dbp":    [80, 80, 79, 80, 81, 80, 80, 79, 80, 81, 80, 80, 79, 80, 81, 80, 80, 79, 80, 81, 80, 80, 79, 80, 81, 80, 80, 79, 80, 81],
    "motion": [0.85, 0.90, 0.88, 0.92, 0.87, 0.85, 0.90, 0.88, 0.92, 0.87, 0.85, 0.90, 0.88, 0.92, 0.87, 0.85, 0.90, 0.88, 0.92, 0.87, 0.85, 0.90, 0.88, 0.92, 0.87, 0.85, 0.90, 0.88, 0.92, 0.87]
  },
  "patient_id": "MOTION_TEST"
}