"""
preprocessing.py
Feature engineering module for Smart Ambulance vitals data.
Converts raw time-series vitals into windowed features for ML models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats


class VitalsPreprocessor:
    """
    Preprocesses raw vital signs into windowed features.
    
    Features extracted (21 total):
    - Mean values (4): hr_mean, spo2_mean, sbp_mean, dbp_mean
    - Standard deviations (4): hr_std, spo2_std, sbp_std, dbp_std
    - Slopes/trends (4): hr_slope, spo2_slope, sbp_slope, dbp_slope
    - Min/max values (6): hr_min, hr_max, spo2_min, spo2_max
    - Correlation (1): hr_spo2_corr
    - Pulse pressure (2): pp_mean, pp_std
    - Motion (2): motion_mean, motion_max
    """
    
    def __init__(self, window_size: int = 30, step_size: int = 5, fs: int = 1):
        """
        Initialize preprocessor.
        
        Args:
            window_size: Window duration in seconds (default: 30)
            step_size: Step size in seconds for sliding window (default: 5)
            fs: Sampling frequency in Hz (default: 1)
        """
        self.window_size = window_size
        self.step_size = step_size
        self.fs = fs
        
        # Expected vital signs columns
        self.required_cols = ['hr', 'spo2', 'sbp', 'dbp', 'motion']
        
        # Feature names (21 features)
        self.feature_names = [
            'hr_mean', 'spo2_mean', 'sbp_mean', 'dbp_mean',
            'hr_std', 'spo2_std', 'sbp_std', 'dbp_std',
            'hr_slope', 'spo2_slope', 'sbp_slope', 'dbp_slope',
            'hr_min', 'hr_max', 'spo2_min', 'spo2_max',
            'hr_spo2_corr', 'pp_mean', 'pp_std',
            'motion_mean', 'motion_max'
        ]
    
    def extract_window_features(self, window_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features from a single window of vital signs.
        
        Args:
            window_data: DataFrame with columns [hr, spo2, sbp, dbp, motion]
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Check if window has enough data
        if len(window_data) < 5:
            # Return NaN features if insufficient data
            return {name: np.nan for name in self.feature_names}
        
        # Extract each vital sign
        hr = window_data['hr'].values
        spo2 = window_data['spo2'].values
        sbp = window_data['sbp'].values
        dbp = window_data['dbp'].values
        motion = window_data['motion'].values
        
        # === MEAN VALUES ===
        features['hr_mean'] = np.mean(hr)
        features['spo2_mean'] = np.mean(spo2)
        features['sbp_mean'] = np.mean(sbp)
        features['dbp_mean'] = np.mean(dbp)
        
        # === STANDARD DEVIATIONS ===
        features['hr_std'] = np.std(hr)
        features['spo2_std'] = np.std(spo2)
        features['sbp_std'] = np.std(sbp)
        features['dbp_std'] = np.std(dbp)
        
        # === SLOPES (linear regression) ===
        t = np.arange(len(hr))
        
        # HR slope
        if np.std(hr) > 0:
            slope_hr, _, _, _, _ = stats.linregress(t, hr)
            features['hr_slope'] = slope_hr
        else:
            features['hr_slope'] = 0.0
        
        # SpO2 slope
        if np.std(spo2) > 0:
            slope_spo2, _, _, _, _ = stats.linregress(t, spo2)
            features['spo2_slope'] = slope_spo2
        else:
            features['spo2_slope'] = 0.0
        
        # SBP slope
        if np.std(sbp) > 0:
            slope_sbp, _, _, _, _ = stats.linregress(t, sbp)
            features['sbp_slope'] = slope_sbp
        else:
            features['sbp_slope'] = 0.0
        
        # DBP slope
        if np.std(dbp) > 0:
            slope_dbp, _, _, _, _ = stats.linregress(t, dbp)
            features['dbp_slope'] = slope_dbp
        else:
            features['dbp_slope'] = 0.0
        
        # === MIN/MAX VALUES ===
        features['hr_min'] = np.min(hr)
        features['hr_max'] = np.max(hr)
        features['spo2_min'] = np.min(spo2)
        features['spo2_max'] = np.max(spo2)
        
        # === HR-SPO2 CORRELATION ===
        if np.std(hr) > 0 and np.std(spo2) > 0:
            corr = np.corrcoef(hr, spo2)[0, 1]
            features['hr_spo2_corr'] = corr if not np.isnan(corr) else 0.0
        else:
            features['hr_spo2_corr'] = 0.0
        
        # === PULSE PRESSURE ===
        pp = sbp - dbp
        features['pp_mean'] = np.mean(pp)
        features['pp_std'] = np.std(pp)
        
        # === MOTION ===
        features['motion_mean'] = np.mean(motion)
        features['motion_max'] = np.max(motion)
        
        return features
    
    def create_windows(self, vitals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sliding windows from continuous vital signs data.
        
        Args:
            vitals_df: DataFrame with columns [hr, spo2, sbp, dbp, motion]
                      Index should be time in seconds or sequential
        
        Returns:
            DataFrame with windowed features (one row per window)
        """
        # Validate input columns
        missing_cols = set(self.required_cols) - set(vitals_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate window parameters
        window_samples = self.window_size * self.fs
        step_samples = self.step_size * self.fs
        
        # Generate windows
        features_list = []
        window_indices = []
        
        for start_idx in range(0, len(vitals_df) - window_samples + 1, step_samples):
            end_idx = start_idx + window_samples
            window_data = vitals_df.iloc[start_idx:end_idx]
            
            # Extract features
            features = self.extract_window_features(window_data)
            features_list.append(features)
            
            # Store window metadata
            window_indices.append({
                'window_start': start_idx,
                'window_end': end_idx,
                'time_start': start_idx / self.fs,
                'time_end': end_idx / self.fs
            })
        
        # Create DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Add window metadata
        for key in window_indices[0].keys():
            features_df[key] = [w[key] for w in window_indices]
        
        return features_df
    
    def process_single_window(self, vitals_dict: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Process a single window provided as a dictionary.
        Useful for API requests.
        
        Args:
            vitals_dict: Dictionary with keys [hr, spo2, sbp, dbp, motion]
                        Each value is a list of measurements
        
        Returns:
            Dictionary of extracted features
        
        Example:
            >>> vitals = {
            ...     'hr': [75, 76, 74, 75, 77],
            ...     'spo2': [98, 98, 97, 98, 98],
            ...     'sbp': [120, 121, 119, 120, 122],
            ...     'dbp': [80, 80, 79, 80, 81],
            ...     'motion': [0.1, 0.2, 0.1, 0.15, 0.1]
            ... }
            >>> processor = VitalsPreprocessor()
            >>> features = processor.process_single_window(vitals)
        """
        # Convert to DataFrame
        window_df = pd.DataFrame(vitals_dict)
        
        # Validate columns
        missing_cols = set(self.required_cols) - set(window_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required vitals: {missing_cols}")
        
        # Extract features
        features = self.extract_window_features(window_df)
        
        return features
    
    def get_feature_vector(self, features_dict: Dict[str, float]) -> np.ndarray:
        """
        Convert feature dictionary to ordered numpy array.
        
        Args:
            features_dict: Dictionary of features
        
        Returns:
            Numpy array in the correct order for ML models
        """
        return np.array([features_dict[name] for name in self.feature_names])


# Utility function for quick processing
def extract_features_from_vitals(
    hr: List[float],
    spo2: List[float],
    sbp: List[float],
    dbp: List[float],
    motion: List[float]
) -> Dict[str, float]:
    """
    Quick feature extraction from vitals lists.
    
    Args:
        hr: List of heart rate values
        spo2: List of SpO2 values
        sbp: List of systolic BP values
        dbp: List of diastolic BP values
        motion: List of motion values
    
    Returns:
        Dictionary of extracted features
    """
    preprocessor = VitalsPreprocessor()
    vitals_dict = {
        'hr': hr,
        'spo2': spo2,
        'sbp': sbp,
        'dbp': dbp,
        'motion': motion
    }
    return preprocessor.process_single_window(vitals_dict)


if __name__ == "__main__":
    # Example usage
    print("VitalsPreprocessor - Example Usage\n")
    
    # Create sample data (30 seconds at 1 Hz)
    np.random.seed(42)
    sample_vitals = pd.DataFrame({
        'hr': np.random.normal(75, 5, 30),
        'spo2': np.random.normal(98, 1, 30),
        'sbp': np.random.normal(120, 10, 30),
        'dbp': np.random.normal(80, 5, 30),
        'motion': np.random.uniform(0, 0.3, 30)
    })
    
    # Initialize preprocessor
    preprocessor = VitalsPreprocessor(window_size=30, step_size=5)
    
    # Extract features from single window
    features = preprocessor.extract_window_features(sample_vitals)
    
    print("Extracted Features:")
    print("-" * 50)
    for name, value in features.items():
        print(f"{name:20s}: {value:.3f}")
    
    print("\n✅ Preprocessing module ready!")