import numpy as np
import pandas as pd
import os
import logging
from scipy.optimize import curve_fit
from numpy.linalg import svd
from typing import Dict
from sklearn.metrics import r2_score
import gsw


# --- Entropy Value Method ---
def calculate_entropy(data):
    """
    Calculate the entropy of each column in a DataFrame.
    Normalizes the data before entropy calculation.
    Args:
        data (pd.DataFrame): The input data frame where each column is treated as a feature.
    Returns:
        weight (np.ndarray): Normalized weights based on entropy for each feature.
    """
    # Normalize the data
    data_normalized = (data - data.min()) / np.where(
        (data.max() - data.min()) == 0, 1, (data.max() - data.min())
    )

    entropy = []
    for column in data_normalized.columns:
        # Calculate probability distribution
        prob = data_normalized[column] / data_normalized[column].sum()
        # Calculate entropy for each column
        entropy_value = -np.sum(prob * np.log2(prob + 1e-10))
        entropy.append(entropy_value)

    # Calculate the gain and normalized weight
    gain = 1 - np.array(entropy)
    weight = gain / np.sum(gain)
    return weight


# --- Tensor Analysis Method ---
class ThermoclineDetector:
    """
    Example usage:
        detector = ThermoclineDetector(input_dir="your_input_directory", output_path="output_file.csv")
        detector.process_all_files()
    """
    def __init__(self, input_dir, output_path):
        self.input_dir = input_dir
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def process_all_files(self):
        """
        Process all valid files in the input directory and save the results.
        """
        files = self._get_valid_files()
        if not files:
            print("No valid files found")
            return

        # Load a sample file for initial setup
        sample_data = self._load_data(files[0])
        if sample_data.empty:
            print("Sample file is empty or corrupted")
            return

        # Get all unique depths
        all_depths = np.unique(sample_data['depth'].values)
        all_depths.sort()

        # Process each file
        results = []
        for file_path in files:
            result = self._process_file(file_path, all_depths)
            if result:
                results.extend(result)

        # Save results to a CSV file
        self._save_results(results)

    def _get_valid_files(self):
        """
        Retrieve all valid files from the input directory.
        """
        files = []
        for year in range(2010, 2025):
            for month in range(1, 13):
                if year == 2024 and month > 3:
                    break
                path = os.path.join(self.input_dir, f"BOA_Argo_{year}_{month:02d}.csv")
                if os.path.exists(path):
                    files.append(path)
        return sorted(files)

    def _load_data(self, file_path):
        """
        Load data from a CSV file.
        """
        try:
            df = pd.read_csv(file_path)
            required_cols = ['Latitude', 'Longitude', 'depth', 'temp']
            if not all(col in df.columns for col in required_cols):
                print(f"Missing columns in {os.path.basename(file_path)}")
                return pd.DataFrame()

            df = df[required_cols].copy()
            for col in required_cols:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    print(f"Error converting {col} in {os.path.basename(file_path)}: {str(e)}")
                    return pd.DataFrame()

            df = df.dropna()
            if df.empty:
                print(f"No valid data after conversion in {os.path.basename(file_path)}")
                return pd.DataFrame()

            return df
        except Exception as e:
            print(f"Error loading {os.path.basename(file_path)}: {str(e)}")
            return pd.DataFrame()


# --- Inflection Point Method ---
def calculate_inflection_point_method(depth: np.ndarray, temp: np.ndarray) -> Dict:
    """
    Args:
        depth (np.ndarray): Array of depth values.
        temp (np.ndarray): Array of temperature values corresponding to each depth.
    Returns:
        dict: A dictionary containing thermocline parameters: TTD, TD, Strength, and Thickness.
    """
    max_departure = 0.03
    inflection_points = [(depth[0], temp[0])]
    current_segment = [(depth[0], temp[0])]
    segment_slopes = []

    for i in range(1, len(depth)):
        current_departure = np.abs(temp[i] - np.interp(
            depth[i],
            [p for p, _ in current_segment],
            [t for _, t in current_segment]
        ))
        if current_departure > max_departure:
            inflection_points.append((depth[i - 1], temp[i - 1]))
            current_segment = [(depth[i - 1], temp[i - 1]), (depth[i], temp[i])]
            segment_slopes.append((temp[i] - temp[i - 1]) / (depth[i] - depth[i - 1]))
        else:
            current_segment.append((depth[i], temp[i]))

    if len(inflection_points) > 2:
        for i in range(1, len(inflection_points) - 1):
            if (np.abs(segment_slopes[i]) > np.abs(segment_slopes[i - 1]) and
                    np.abs(segment_slopes[i]) > np.abs(segment_slopes[i + 1])):
                thermocline_segment = inflection_points[i - 1:i + 2]
                break
        else:
            thermocline_segment = inflection_points

        # Calculate thermocline parameters
        TTD = thermocline_segment[1][0]
        TD = np.mean([p for p, _ in thermocline_segment])
        Strength = (thermocline_segment[-1][1] - thermocline_segment[0][1]) / \
                   (thermocline_segment[-1][0] - thermocline_segment[0][0])
        BTD = thermocline_segment[-1][0]
        thickness = BTD - TTD

        return {'TTD': TTD, 'TD': TD, 'Strength': Strength, 'Thickness': thickness}
    else:
        return {}


# --- Variable Representative Isotherm (VRI) Method ---
def calculate_vri_thermocline(depth, temperature):
    """
    Args:
        depth (np.ndarray): Array of depth values.
        temperature (np.ndarray): Array of temperature values corresponding to each depth.
    Returns:
        tuple: Calculated parameters (MLD, TD, TS, thickness, SST, deep_water_temp).
    """
    sst = temperature[0]  # Sea Surface Temperature

    # Mixed layer temperature threshold
    MLD_TEMP_DIFF = 0.8
    mld_temp_threshold = sst - MLD_TEMP_DIFF

    # Find Mixed Layer Depth (MLD)
    MLD_idx = np.where(temperature <= mld_temp_threshold)[0][0]
    MLD = depth[MLD_idx]

    # Calculate temperature at 400m depth
    deep_water_temp = np.interp(400, depth, temperature)

    # Calculate thermocline temperature (TT)
    TT = temperature[MLD_idx] - 0.25 * (temperature[MLD_idx] - deep_water_temp)

    # Calculate thermocline depth (TD)
    TD_idx = np.where(temperature <= TT)[0][0]
    TD = depth[TD_idx]

    # Calculate thermocline strength (TS)
    TS = np.abs((temperature[MLD_idx] - temperature[TD_idx]) / (depth[TD_idx] - depth[MLD_idx]))

    # Calculate thermocline thickness
    thickness = TD - MLD

    return MLD, TD, TS, thickness, sst, deep_water_temp


# --- Sigmoid Function Method ---
def fsigmoid(x, a, b):
    """
    Args:
        x (np.ndarray): Depth values for fitting.
        a, b (float): Parameters for the sigmoid function.
    Returns:
        np.ndarray: Fitted values using the sigmoid function.
    """
    return 1.0 / (1.0 + np.exp(-a * (x - b)))


def thermocline_sigmoid(df):
    """
    Fit thermocline data using a sigmoid function.
    Args:
        df (pd.DataFrame): DataFrame containing pressure (pres) and temperature (ctemp).
    Returns:
        tuple: Thermocline depth, temperature, and R2 score from the sigmoid fit.
    """
    x_true = df.pres.to_numpy()
    y_true = df.ctemp.to_numpy()

    try:
        popt, _ = curve_fit(fsigmoid, x_true, y_true, method='dogbox')
    except:
        return [np.nan] * 8

    y_pred = fsigmoid(x_true, *popt)
    r2 = r2_score(y_true, y_pred)

    # TTD and BTD based on gradient changes
    pres_ttd = np.nan
    temp_ttd = np.nan
    for i in range(1, len(x_true)):
        temp_gradient = np.abs(y_pred[i] - y_pred[i - 1]) / (x_true[i] - x_true[i - 1])
        if temp_gradient > 0.1:  # Threshold for thermocline detection
            pres_ttd = x_true[i]
            temp_ttd = y_pred[i]
            break

    pres_btd = x_true[-1]
    temp_btd = y_pred[-1]

    thickness = pres_btd - pres_ttd
    return pres_btd, temp_btd, pres_ttd, temp_ttd, r2, y_pred


# --- Hyperbolic Tangent Fitting Method ---
def ftanh(x, a, b, c, d):
    """
    Args:
        x (np.ndarray): Depth values.
        a, b, c, d (float): Parameters for the hyperbolic tangent function.
    Returns:
        np.ndarray: Fitted values using the hyperbolic tangent function.
    """
    return a * np.tanh(b * (x - c)) + d


def thermocline_tanh(df):
    """
    Args:
        df (pd.DataFrame): DataFrame containing pressure (pres) and temperature (ctemp).
    Returns:
        tuple: Thermocline depth, temperature, and R2 score from the fitting.
    """
    x_true = df.pres.to_numpy()
    y_true = df.ctemp.to_numpy()

    try:
        popt, _ = curve_fit(ftanh, x_true, y_true)
    except:
        return [np.nan] * 8

    y_pred = ftanh(x_true, *popt)
    r2 = r2_score(y_true, y_pred)

    # TTD and BTD based on gradient changes
    pres_ttd = np.nan
    temp_ttd = np.nan
    for i in range(1, len(x_true)):
        temp_gradient = np.abs(y_pred[i] - y_pred[i - 1]) / (x_true[i] - x_true[i - 1])
        if temp_gradient > 0.1:
            pres_ttd = x_true[i]
            temp_ttd = y_pred[i]
            break

    pres_btd = x_true[-1]
    temp_btd = y_pred[-1]

    thickness = pres_btd - pres_ttd
    return pres_btd, temp_btd, pres_ttd, temp_ttd, r2, y_pred
