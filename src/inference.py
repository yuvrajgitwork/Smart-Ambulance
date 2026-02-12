"""
inference.py
Risk scoring inference module for Smart Ambulance platform.
Combines Isolation Forest, One-Class SVM, and LSTM Autoencoder predictions.
"""

import numpy as np
import joblib
from tensorflow.keras.models import load_model
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class RiskScorer:
    """
    Risk scoring system combining multiple anomaly detection models.
    
    Models:
    - Isolation Forest: Tree-based anomaly detection
    - One-Class SVM: Boundary-based anomaly detection
    - LSTM Autoencoder: Sequence-based reconstruction error
    
    Output:
    - risk_score: Continuous score (0+, higher = more critical)
    - risk_level: NORMAL / WARNING / CRITICAL
    - confidence: Model certainty (0-1)
    - reasoning: List of triggered conditions
    """
    
    def __init__(
        self,
        iso_forest_path: str,
        ocsvm_path: str,
        lstm_path: str,
        sequence_length: int = 10
    ):
        """
        Initialize risk scorer with trained models.
        
        Args:
            iso_forest_path: Path to iso_forest.pkl (contains scaler + model)
            ocsvm_path: Path to ocsvm.pkl
            lstm_path: Path to lstm_autoencoder.keras
            sequence_length: Number of windows for LSTM sequences
        """
        # Load classical ML models
        self.scaler, self.iso_forest = joblib.load(iso_forest_path)
        self.ocsvm = joblib.load(ocsvm_path)
        
        # Load deep learning model
        self.lstm_model = load_model(lstm_path, compile=False)
        
        self.sequence_length = sequence_length
        
        # Feature names (must match training)
        self.feature_names = [
            'hr_mean', 'spo2_mean', 'sbp_mean', 'dbp_mean',
            'hr_std', 'spo2_std', 'sbp_std', 'dbp_std',
            'hr_slope', 'spo2_slope', 'sbp_slope', 'dbp_slope',
            'hr_min', 'hr_max', 'spo2_min', 'spo2_max',
            'hr_spo2_corr', 'pp_mean', 'pp_std',
            'motion_mean', 'motion_max'
        ]
        
        # Historical windows for LSTM (stored per patient/session)
        self.window_history: List[Dict[str, float]] = []
        
        print(f"✅ RiskScorer initialized")
        print(f"   - Isolation Forest: {self.iso_forest.n_estimators} trees")
        print(f"   - One-Class SVM: {self.ocsvm.kernel} kernel")
        print(f"   - LSTM Autoencoder: loaded")
    
    def _normalize_score(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range using min-max scaling."""
        min_score = scores.min()
        max_score = scores.max()
        return (scores - min_score) / (max_score - min_score + 1e-8)
    
    def _get_classical_scores(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Get anomaly scores from Isolation Forest and One-Class SVM.
        
        Args:
            features: Feature vector (21 features)
        
        Returns:
            (iso_score, svm_score) - both normalized to [0, 1]
        """
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # ISO Forest score (higher = more anomalous)
        iso_raw = -self.iso_forest.decision_function(features_scaled)[0]
        
        # One-Class SVM score (higher = more anomalous)
        svm_raw = -self.ocsvm.decision_function(features_scaled)[0]
        
        # Normalize to [0, 1] (rough normalization)
        iso_norm = 1 / (1 + np.exp(-iso_raw))  # Sigmoid
        svm_norm = 1 / (1 + np.exp(-svm_raw / 10))  # Sigmoid with scaling
        
        return iso_norm, svm_norm
    
    def _get_lstm_score(self, features: Dict[str, float]) -> float:
        """
        Get LSTM reconstruction error.
        
        Args:
            features: Feature dictionary for current window
        
        Returns:
            lstm_score: Normalized reconstruction error [0, 1]
        """
        # Add to history
        self.window_history.append(features)
        
        # Need at least sequence_length windows
        if len(self.window_history) < self.sequence_length:
            # Not enough history - return median score
            return 0.5
        
        # Keep only recent windows
        if len(self.window_history) > self.sequence_length:
            self.window_history = self.window_history[-self.sequence_length:]
        
        # Create sequence
        sequence = np.array([
            [w[name] for name in self.feature_names]
            for w in self.window_history
        ])
        sequence = sequence.reshape(1, self.sequence_length, len(self.feature_names))
        
        # Get reconstruction
        reconstructed = self.lstm_model.predict(sequence, verbose=0)
        
        # Calculate MSE
        mse = np.mean((sequence - reconstructed) ** 2)
        
        # Normalize (rough normalization based on typical MSE range)
        lstm_norm = 1 / (1 + np.exp(-10 * (mse - 0.01)))
        
        return np.clip(lstm_norm, 0, 1)
    
    def calculate_risk_score(
        self,
        features: Dict[str, float],
        lookback_features: Optional[List[Dict[str, float]]] = None
    ) -> Dict:
        """
        Calculate comprehensive risk score for a window.
        
        Args:
            features: Current window features
            lookback_features: Previous 5 windows for persistence check (optional)
        
        Returns:
            Dictionary with:
            - risk_score: Final risk score (0+)
            - risk_level: 'NORMAL', 'WARNING', or 'CRITICAL'
            - confidence: 0-1
            - ensemble_score: Combined ML score
            - reasoning: List of triggered conditions
            - model_scores: Individual model scores
        """
        reasoning = []
        
        # Convert features dict to numpy array
        feature_vector = np.array([features[name] for name in self.feature_names])
        
        # ========================================
        # 1. GET MODEL SCORES
        # ========================================
        iso_score, svm_score = self._get_classical_scores(feature_vector)
        lstm_score = self._get_lstm_score(features)
        
        # Ensemble score (weighted average)
        ensemble_score = (
            0.30 * iso_score +
            0.30 * svm_score +
            0.40 * lstm_score
        )
        
        # ========================================
        # 2. CLINICAL SEVERITY MULTIPLIERS
        # ========================================
        severity = 1.0
        
        # Critical SpO2
        if features['spo2_min'] < 88:
            severity += 2.0
            reasoning.append(f"CRITICAL SpO2: {features['spo2_min']:.1f}% < 88%")
        elif features['spo2_min'] < 92:
            severity += 1.0
            reasoning.append(f"Low SpO2: {features['spo2_min']:.1f}% < 92%")
        elif features['spo2_mean'] < 94:
            severity += 0.3
            reasoning.append(f"Borderline SpO2: {features['spo2_mean']:.1f}% < 94%")
        
        # Rapid HR increase
        if features['hr_slope'] > 5:
            severity += 0.5
            reasoning.append(f"Rapid HR increase: +{features['hr_slope']:.1f} bpm/window")
        elif features['hr_slope'] > 2:
            severity += 0.2
            reasoning.append(f"Moderate HR increase: +{features['hr_slope']:.1f} bpm/window")
        
        # Tachycardia
        if features['hr_mean'] > 140:
            severity += 1.0
            reasoning.append(f"Severe tachycardia: HR={features['hr_mean']:.0f} > 140")
        elif features['hr_mean'] > 120:
            severity += 0.5
            reasoning.append(f"Tachycardia: HR={features['hr_mean']:.0f} > 120")
        
        # Bradycardia
        if features['hr_mean'] < 45:
            severity += 1.0
            reasoning.append(f"Severe bradycardia: HR={features['hr_mean']:.0f} < 45")
        elif features['hr_mean'] < 55:
            severity += 0.3
            reasoning.append(f"Bradycardia: HR={features['hr_mean']:.0f} < 55")
        
        # Multi-vital distress
        if features['hr_mean'] > 120 and features['spo2_mean'] < 92:
            severity += 1.5
            reasoning.append(f"Multi-vital distress: HR={features['hr_mean']:.0f} + SpO2={features['spo2_mean']:.1f}%")
        
        # High variability
        if features['hr_std'] > 15:
            severity += 0.3
            reasoning.append(f"High HR variability: std={features['hr_std']:.1f}")
        
        # Abnormal blood pressure
        if features['sbp_mean'] > 180 or features['sbp_mean'] < 90:
            severity += 0.5
            reasoning.append(f"Abnormal BP: SBP={features['sbp_mean']:.0f}")
        
        # ========================================
        # 3. TEMPORAL PERSISTENCE
        # ========================================
        persistence_score = 1.0
        
        if lookback_features and len(lookback_features) >= 5:
            # Check how many recent windows have high ensemble scores
            recent_ensemble = [
                0.30 * self._get_classical_scores(
                    np.array([w[name] for name in self.feature_names])
                )[0] +
                0.30 * self._get_classical_scores(
                    np.array([w[name] for name in self.feature_names])
                )[1]
                for w in lookback_features[-5:]
            ]
            
            persistent_anomalies = sum(1 for s in recent_ensemble if s > 0.6)
            
            if persistent_anomalies >= 4:
                persistence_score = 1.5
                reasoning.append(f"Sustained anomaly: {persistent_anomalies}/5 windows")
            elif persistent_anomalies >= 3:
                persistence_score = 1.3
                reasoning.append(f"Recurring anomaly: {persistent_anomalies}/5 windows")
        
        # ========================================
        # 4. MODEL AGREEMENT (Confidence)
        # ========================================
        model_scores = [iso_score, svm_score, lstm_score]
        score_std = np.std(model_scores)
        
        if score_std < 0.1:
            model_confidence = 1.0
            reasoning.append("High model agreement")
        elif score_std < 0.2:
            model_confidence = 0.9
        elif score_std < 0.3:
            model_confidence = 0.7
        else:
            model_confidence = 0.5
            reasoning.append("Low model agreement (uncertain)")
        
        # ========================================
        # 5. MOTION ARTIFACT SUPPRESSION
        # ========================================
        motion_confidence = 1.0
        
        if features['motion_max'] > 0.8:
            motion_confidence = 0.4
            reasoning.append(f"High motion detected: {features['motion_max']:.2f} (likely artifact)")
        elif features['motion_max'] > 0.6:
            motion_confidence = 0.7
            reasoning.append(f"Moderate motion: {features['motion_max']:.2f}")
        
        # SpO2 drop during high motion = likely artifact
        if features['motion_max'] > 0.6 and features['spo2_slope'] < -2:
            motion_confidence *= 0.5
            reasoning.append("SpO2 drop during motion (possible artifact)")
        
        # ========================================
        # 6. FINAL RISK SCORE
        # ========================================
        base_risk = ensemble_score * severity * persistence_score
        overall_confidence = model_confidence * motion_confidence
        
        risk_score = base_risk * overall_confidence
        
        # ========================================
        # 7. RISK LEVEL CLASSIFICATION
        # ========================================
        if risk_score > 1.5 and overall_confidence > 0.6:
            risk_level = 'CRITICAL'
        elif risk_score > 0.8 and overall_confidence > 0.5:
            risk_level = 'WARNING'
        else:
            risk_level = 'NORMAL'
        
        # ========================================
        # 8. ALERT SUPPRESSION
        # ========================================
        if motion_confidence < 0.5 and severity < 2.0:
            risk_level = 'NORMAL'
            reasoning.append("ALERT SUPPRESSED: Motion artifact without severe vitals")
        
        if overall_confidence < 0.4 and risk_score < 2.0:
            risk_level = 'NORMAL'
            reasoning.append("ALERT SUPPRESSED: Low confidence")
        
        return {
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'confidence': float(overall_confidence),
            'ensemble_score': float(ensemble_score),
            'severity': float(severity),
            'persistence': float(persistence_score),
            'reasoning': reasoning,
            'model_scores': {
                'isolation_forest': float(iso_score),
                'one_class_svm': float(svm_score),
                'lstm_autoencoder': float(lstm_score)
            }
        }
    
    def reset_history(self):
        """Reset LSTM window history (call for new patient/session)."""
        self.window_history = []


if __name__ == "__main__":
    print("Risk Scorer - Test Mode\n")
    
    # This would normally load actual models
    print("⚠️  To test, provide actual model paths:")
    print("   - iso_forest.pkl")
    print("   - ocsvm.pkl")
    print("   - lstm_autoencoder.keras")
    print("\n✅ Inference module ready!")