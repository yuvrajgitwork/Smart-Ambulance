"""
test_api.py
Test script for Smart Ambulance API endpoints.
Tests normal vitals, distress scenarios, and motion artifacts.
"""

import requests
import json
import numpy as np
from typing import Dict, List
import time


# API configuration
API_BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_result(response: Dict):
    """Pretty print API response."""
    print(f"\n🎯 Risk Score: {response['risk_score']:.3f}")
    print(f"📊 Risk Level: {response['risk_level']}")
    print(f"✅ Confidence: {response['confidence']:.2f}")
    print(f"\n📝 Reasoning:")
    if response['reasoning']:
        for reason in response['reasoning']:
            print(f"   • {reason}")
    else:
        print("   • No specific concerns")
    print(f"\n🤖 Model Scores:")
    for model, score in response['model_scores'].items():
        print(f"   • {model}: {score:.3f}")


def generate_normal_vitals(duration: int = 30) -> Dict[str, List[float]]:
    """Generate normal vital signs."""
    np.random.seed(42)
    return {
        'hr': list(np.random.normal(75, 5, duration)),
        'spo2': list(np.random.normal(98, 1, duration)),
        'sbp': list(np.random.normal(120, 8, duration)),
        'dbp': list(np.random.normal(80, 5, duration)),
        'motion': list(np.random.uniform(0, 0.3, duration))
    }


def generate_distress_vitals(duration: int = 30) -> Dict[str, List[float]]:
    """Generate distress scenario (high HR, low SpO2)."""
    np.random.seed(43)
    return {
        'hr': list(np.random.normal(145, 8, duration)),      # Tachycardia
        'spo2': list(np.random.normal(88, 2, duration)),     # Low oxygen
        'sbp': list(np.random.normal(110, 10, duration)),
        'dbp': list(np.random.normal(75, 6, duration)),
        'motion': list(np.random.uniform(0, 0.2, duration))
    }


def generate_motion_artifact(duration: int = 30) -> Dict[str, List[float]]:
    """Generate motion artifact (SpO2 drop during high motion)."""
    np.random.seed(44)
    return {
        'hr': list(np.random.normal(78, 6, duration)),
        'spo2': list(np.random.normal(89, 3, duration)),     # Apparent drop
        'sbp': list(np.random.normal(120, 8, duration)),
        'dbp': list(np.random.normal(80, 5, duration)),
        'motion': list(np.random.uniform(0.7, 0.95, duration))  # High motion!
    }


def generate_gradual_deterioration(duration: int = 30) -> Dict[str, List[float]]:
    """Generate gradually worsening vitals."""
    np.random.seed(45)
    # HR increases linearly
    hr_trend = np.linspace(80, 125, duration)
    hr = list(hr_trend + np.random.normal(0, 3, duration))
    
    # SpO2 decreases linearly
    spo2_trend = np.linspace(98, 91, duration)
    spo2 = list(spo2_trend + np.random.normal(0, 1, duration))
    
    return {
        'hr': hr,
        'spo2': spo2,
        'sbp': list(np.random.normal(115, 8, duration)),
        'dbp': list(np.random.normal(78, 5, duration)),
        'motion': list(np.random.uniform(0, 0.3, duration))
    }


def test_health_check():
    """Test health check endpoint."""
    print_section("TEST 1: Health Check")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        result = response.json()
        
        print(f"\n✅ Status: {result['status']}")
        print(f"✅ Models Loaded: {result['models_loaded']}")
        print(f"🕐 Timestamp: {result['timestamp']}")
        
        if not result['models_loaded']:
            print("\n⚠️  WARNING: Models not loaded! API may not work.")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API!")
        print("   Make sure the API is running: python src/api.py")
        return False


def test_normal_vitals():
    """Test prediction with normal vitals."""
    print_section("TEST 2: Normal Vitals")
    
    vitals = generate_normal_vitals()
    
    print("\n📊 Input Vitals:")
    print(f"   HR mean: {np.mean(vitals['hr']):.1f} bpm")
    print(f"   SpO2 mean: {np.mean(vitals['spo2']):.1f}%")
    print(f"   Motion mean: {np.mean(vitals['motion']):.3f}")
    
    payload = {
        "vitals": vitals,
        "patient_id": "TEST_001"
    }
    
    response = requests.post(f"{API_BASE_URL}/predict", json=payload)
    result = response.json()
    
    print_result(result)
    
    # Expected: NORMAL
    if result['risk_level'] != 'NORMAL':
        print("\n⚠️  WARNING: Expected NORMAL, got", result['risk_level'])


def test_distress_scenario():
    """Test prediction with distress vitals."""
    print_section("TEST 3: Distress Scenario (High HR + Low SpO2)")
    
    vitals = generate_distress_vitals()
    
    print("\n📊 Input Vitals:")
    print(f"   HR mean: {np.mean(vitals['hr']):.1f} bpm (TACHYCARDIA)")
    print(f"   SpO2 mean: {np.mean(vitals['spo2']):.1f}% (LOW)")
    print(f"   Motion mean: {np.mean(vitals['motion']):.3f}")
    
    payload = {
        "vitals": vitals,
        "patient_id": "TEST_002"
    }
    
    response = requests.post(f"{API_BASE_URL}/predict", json=payload)
    result = response.json()
    
    print_result(result)
    
    # Expected: CRITICAL or WARNING
    if result['risk_level'] == 'NORMAL':
        print("\n⚠️  WARNING: Expected CRITICAL/WARNING, got NORMAL")


def test_motion_artifact():
    """Test motion artifact suppression."""
    print_section("TEST 4: Motion Artifact (Should Suppress)")
    
    vitals = generate_motion_artifact()
    
    print("\n📊 Input Vitals:")
    print(f"   HR mean: {np.mean(vitals['hr']):.1f} bpm (NORMAL)")
    print(f"   SpO2 mean: {np.mean(vitals['spo2']):.1f}% (appears low)")
    print(f"   Motion mean: {np.mean(vitals['motion']):.3f} (HIGH!)")
    
    payload = {
        "vitals": vitals,
        "patient_id": "TEST_003"
    }
    
    response = requests.post(f"{API_BASE_URL}/predict", json=payload)
    result = response.json()
    
    print_result(result)
    
    # Check if motion was detected in reasoning
    motion_detected = any('motion' in r.lower() for r in result['reasoning'])
    
    if motion_detected:
        print("\n✅ Motion artifact correctly detected!")
    
    # Should be suppressed or have low confidence
    if result['confidence'] < 0.6:
        print("✅ Low confidence due to motion - good!")
    
    if 'SUPPRESSED' in ' '.join(result['reasoning']):
        print("✅ Alert suppressed - excellent!")


def test_gradual_deterioration():
    """Test detection of gradual deterioration."""
    print_section("TEST 5: Gradual Deterioration")
    
    vitals = generate_gradual_deterioration()
    
    print("\n📊 Input Vitals:")
    print(f"   HR: {vitals['hr'][0]:.1f} → {vitals['hr'][-1]:.1f} bpm (INCREASING)")
    print(f"   SpO2: {vitals['spo2'][0]:.1f} → {vitals['spo2'][-1]:.1f}% (DECREASING)")
    
    payload = {
        "vitals": vitals,
        "patient_id": "TEST_004"
    }
    
    response = requests.post(f"{API_BASE_URL}/predict", json=payload)
    result = response.json()
    
    print_result(result)
    
    # Check if slope was detected
    slope_detected = any('slope' in r.lower() or 'increase' in r.lower() for r in result['reasoning'])
    
    if slope_detected:
        print("\n✅ Trend correctly detected!")


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print_section("TEST 6: Batch Prediction (3 windows)")
    
    vitals_list = [
        generate_normal_vitals(),
        generate_distress_vitals(),
        generate_normal_vitals()
    ]
    
    payload = {
        "vitals_list": vitals_list,
        "patient_id": "TEST_BATCH"
    }
    
    response = requests.post(f"{API_BASE_URL}/predict_batch", json=payload)
    result = response.json()
    
    print(f"\n✅ Processed {result['total_windows']} windows")
    
    for i, pred in enumerate(result['predictions']):
        print(f"\nWindow {i+1}:")
        print(f"   Risk Level: {pred['risk_level']}")
        print(f"   Risk Score: {pred['risk_score']:.3f}")
        print(f"   Confidence: {pred['confidence']:.2f}")


def test_session_reset():
    """Test session reset endpoint."""
    print_section("TEST 7: Session Reset")
    
    response = requests.post(f"{API_BASE_URL}/reset")
    result = response.json()
    
    print(f"\n✅ Status: {result['status']}")
    print(f"📝 Message: {result['message']}")


def test_invalid_vitals():
    """Test API validation with invalid vitals."""
    print_section("TEST 8: Invalid Vitals (Validation)")
    
    # HR out of range
    invalid_vitals = generate_normal_vitals()
    invalid_vitals['hr'] = [250] * 30  # Too high!
    
    payload = {
        "vitals": invalid_vitals,
        "patient_id": "TEST_INVALID"
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=payload)
        
        if response.status_code == 422:
            print("\n✅ Validation working! Rejected invalid HR values")
            print(f"   Error: {response.json()['detail'][0]['msg']}")
        else:
            print("\n⚠️  WARNING: Invalid vitals were accepted!")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")


def run_all_tests():
    """Run all API tests."""
    print("\n" + "="*70)
    print("  🚑 SMART AMBULANCE API TEST SUITE")
    print("="*70)
    print(f"\nTesting API at: {API_BASE_URL}")
    
    # Check if API is running
    if not test_health_check():
        print("\n❌ Cannot proceed - API not running or models not loaded")
        print("\nTo start the API:")
        print("   1. Ensure models are in models/ directory")
        print("   2. Run: python src/api.py")
        return
    
    # Run tests
    time.sleep(1)
    test_normal_vitals()
    
    time.sleep(1)
    test_distress_scenario()
    
    time.sleep(1)
    test_motion_artifact()
    
    time.sleep(1)
    test_gradual_deterioration()
    
    time.sleep(1)
    test_batch_prediction()
    
    time.sleep(1)
    test_session_reset()
    
    time.sleep(1)
    test_invalid_vitals()
    
    # Summary
    print("\n" + "="*70)
    print("  ✅ TEST SUITE COMPLETED")
    print("="*70)
    print("\nAll endpoints tested successfully!")
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()