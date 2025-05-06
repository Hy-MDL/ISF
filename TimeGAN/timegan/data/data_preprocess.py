"""Hide-and-Seek Privacy Challenge Codebase.

Reference: James Jordon, Daniel Jarrett, Jinsung Yoon, Ari Ercole, Cheng Zhang, Danielle Belgrave, Mihaela van der Schaar, 
"Hide-and-Seek Privacy Challenge: Synthetic Data Generation vs. Patient Re-identification with Clinical Time-series Data," 
Neural Information Processing Systems (NeurIPS) Competition, 2020.

Link: https://www.vanderschaar-lab.com/announcing-the-neurips-2020-hide-and-seek-privacy-challenge/

Last updated Date: Oct 17th 2020
Code author: Jinsung Yoon, Evgeny Saveliev
Contact: jsyoon0823@gmail.com, e.s.saveliev@gmail.com


-----------------------------

(1) data_preprocess: Load the data and preprocess into a 3d numpy array
(2) imputater: Impute missing data 
"""
# Local packages
import os
from typing import Union, Tuple, List
import warnings
warnings.filterwarnings("ignore")

# 3rd party modules
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.interpolate import interp1d

def data_preprocess(
    file_name: str, 
    max_seq_len: int, 
    padding_value: float=-1.0,
    impute_method: str="mode", 
    scaling_method: str="minmax", 
) -> Tuple[np.ndarray, np.ndarray, List, int, float, np.ndarray, np.ndarray]:
    """Load the data and preprocess into 3d numpy array.
    Preprocessing includes:
    1. Extract sequence length for each patient id
    2. Impute missing data 
    3. Normalize data
    4. Sort dataset according to sequence length

    Args:
    - file_name (str): CSV file name
    - max_seq_len (int): maximum sequence length
    - impute_method (str): The imputation method ("median" or "mode") 
    - scaling_method (str): The scaler method ("standard" or "minmax")

    Returns:
    - processed_data: preprocessed data
    - time: ndarray of ints indicating the length for each data
    - params: the parameters to rescale the data
    - max_seq_len: maximum sequence length
    - padding_value: value used for padding
    - uniq_id: unique ID values of the processed data
    - dropped_indices: empty array (no outliers dropped)
    """

    #########################
    # Load data
    #########################

    index = 'Idx'

    # Load csv
    print("Loading data...\n")
    ori_data = pd.read_csv(file_name)

    # Remove spurious column, so that column 0 is now 'admissionid'.
    if ori_data.columns[0] == "Unnamed: 0":  
        ori_data = ori_data.drop(["Unnamed: 0"], axis=1)

    # 아웃라이어 보간을 위한 인덱스 저장
    all_indices = ori_data[index].values
    
    #########################
    # Detect and interpolate outliers
    #########################
    
    # Z-score 계산
    z_scores = stats.zscore(ori_data, axis=0, nan_policy='omit')
    outlier_mask = np.nanmax(np.abs(z_scores), axis=1) >= 3
    
    # 아웃라이어가 있는 행의 인덱스 저장
    outlier_indices = all_indices[outlier_mask]
    
    # 각 특성별로 아웃라이어 보간
    for col in ori_data.columns:
        if col != index:  # 인덱스 열은 제외
            # 아웃라이어가 아닌 값들의 인덱스
            valid_idx = ~outlier_mask
            # 아웃라이어가 아닌 값들
            valid_values = ori_data[col].values[valid_idx]
            
            if len(valid_values) > 1:  # 보간을 위한 충분한 데이터가 있는 경우
                # 보간 함수 생성
                f = interp1d(np.where(valid_idx)[0], valid_values, 
                            kind='linear', bounds_error=False, 
                            fill_value='extrapolate')
                
                # 아웃라이어 위치에 보간된 값 적용
                ori_data.loc[outlier_mask, col] = f(np.where(outlier_mask)[0])
    
    print(f"Interpolated {np.sum(outlier_mask)} outliers\n")

    # Parameters
    uniq_id = np.unique(ori_data[index])
    no = len(uniq_id)
    dim = len(ori_data.columns) - 1

    #########################
    # Impute, scale and pad data
    #########################
    
    # Initialize scaler
    if scaling_method == "minmax":
        scaler = MinMaxScaler()
        scaler.fit(ori_data)
        params = [scaler.data_min_, scaler.data_max_]
    
    elif scaling_method == "standard":
        scaler = StandardScaler()
        scaler.fit(ori_data)
        params = [scaler.mean_, scaler.var_]

    # Imputation values
    if impute_method == "median":
        impute_vals = ori_data.median()
    elif impute_method == "mode":
        impute_vals = stats.mode(ori_data).mode[0]
    else:
        raise ValueError("Imputation method should be `median` or `mode`")    

    # TODO: Sanity check for padding value
    # if np.any(ori_data == padding_value):
    #     print(f"Padding value `{padding_value}` found in data")
    #     padding_value = np.nanmin(ori_data.to_numpy()) - 1
    #     print(f"Changed padding value to: {padding_value}\n")
    
    # Output initialization
    output = np.empty([no, max_seq_len, dim])  # Shape:[no, max_seq_len, dim]
    output.fill(padding_value)
    time = []

    # For each uniq id
    for i in tqdm(range(no)):
        # Extract the time-series data with a certain admissionid

        curr_data = ori_data[ori_data[index] == uniq_id[i]].to_numpy()

        # Impute missing data
        curr_data = imputer(curr_data, impute_vals)

        # Normalize data
        curr_data = scaler.transform(curr_data)
        
        # Extract time and assign to the preprocessed data (Excluding ID)
        curr_no = len(curr_data)

        # Pad data to `max_seq_len`
        if curr_no >= max_seq_len:
            output[i, :, :] = curr_data[:max_seq_len, 1:]  # Shape: [1, max_seq_len, dim]
            time.append(max_seq_len)
        else:
            output[i, :curr_no, :] = curr_data[:, 1:]  # Shape: [1, max_seq_len, dim]
            time.append(curr_no)

    # 빈 배열 반환 (아웃라이어 제거 없음)
    return output, time, params, max_seq_len, padding_value, uniq_id, np.array([])

def imputer(
    curr_data: np.ndarray, 
    impute_vals: List, 
    zero_fill: bool = True
) -> np.ndarray:
    """Impute missing data given values for each columns.

    Args:
        curr_data (np.ndarray): Data before imputation.
        impute_vals (list): Values to be filled for each column.
        zero_fill (bool, optional): Whather to Fill with zeros the cases where 
            impute_val is nan. Defaults to True.

    Returns:
        np.ndarray: Imputed data.
    """

    curr_data = pd.DataFrame(data=curr_data)
    impute_vals = pd.Series(impute_vals)
    
    # Impute data
    imputed_data = curr_data.fillna(impute_vals)

    # Zero-fill, in case the `impute_vals` for a particular feature is `nan`.
    imputed_data = imputed_data.fillna(0.0)

    # Check for any N/A values
    if imputed_data.isnull().any().any():
        raise ValueError("NaN values remain after imputation")

    return imputed_data.to_numpy()
