# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import numpy as np
from typing import Dict, List


def split_text(text: str) -> Dict:
    """
    Extracts sections containing NMR data from the input text.

    Args:
        text (str): Text containing NMR data.

    Returns:
        dict: A dictionary with keys 'H' and 'C', containing respective NMR data.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Find 1H NMR and 13C NMR sections
    h_pattern = r'1H NMR(.*?)(?=13C NMR|$)'
    c_pattern = r'13C NMR(.*?)$'
    
    result = {}
    
    h_match = re.search(h_pattern, text, re.DOTALL)
    if h_match:
        result['H'] = h_match.group(1).strip()
    
    c_match = re.search(c_pattern, text, re.DOTALL)
    if c_match:
        result['C'] = c_match.group(1).strip()
    
    return result


def parse_h_raw(nmr_data: str):
    """
    Parses raw 1H NMR data into structured format.

    Args:
        nmr_data (str): Raw 1H NMR data.

    Returns:
        list: Parsed NMR data as a list of tuples.
    """
    pattern = re.compile(
        r'(?P<shift_left_1>-?\d+\.\d+)\s*(?:–|-)\s*(?P<shift_right_1>-?\d+\.\d+)\s*\((?P<type_1>[a-zA-Z\s\.,]*?)?,?\s*(?:(?:J\s*=\s*(?P<hz_1>[\d\.\s,]+)\s*Hz,\s*)?(?P<count_1>\d+(\.\d+)?)\s*H(?:,\s*(?P<annotation_1>[^)]+))?|(?P<count_alt_1>\d+(\.\d+)?)\s*H,\s*J\s*=\s*(?P<hz_alt_1>[\d\.\s,]+)\s*Hz(?:,\s*(?P<annotation_alt_1>[^)]+))?)\)'
        r'|(?P<shift_2>-?\d+\.\d+)\s*\((?P<type_2>[a-zA-Z\s\.,]*?)?,?\s*(?:(?:J\s*=\s*(?P<hz_2>[\d\.\s,]+)\s*Hz,\s*)?(?P<count_2>\d+(\.\d+)?)\s*H(?:,\s*(?P<annotation_2>[^)]+))?|(?P<count_alt_2>\d+(\.\d+)?)\s*H,\s*J\s*=\s*(?P<hz_alt_2>[\d\.\s,]+)\s*Hz(?:,\s*(?P<annotation_alt_2>[^)]+))?)\)'
        # r'|(?P<shift_left_3>-?\d+\.\d+)\s*(?:–|-)\s*(?P<shift_right_3>-?\d+\.\d+)\s*\((?P<count_3>\d+(\.\d+)?)\s*H(?:,\s*(?P<type_3>[a-zA-Z\s\.,]*?)?)?(?:,\s*J\s*=\s*(?P<hz_3>[\d\.\s,]+)\s*Hz)?(?:,\s*(?P<annotation_3>[^)]+))?\)'
        # r'|(?P<shift_4>-?\d+\.\d+)\s*\((?P<count_4>\d+(\.\d+)?)\s*H(?:,\s*(?P<type_4>[a-zA-Z\s\.,]*?)?)?(?:,\s*J\s*=\s*(?P<hz_4>[\d\.\s,]+)\s*Hz)?(?:,\s*(?P<annotation_4>[^)]+))?\)'
    )

    def parse_j_values(j_string):
        if j_string is None:
            return None
        return [float(j) for j in re.findall(r'-?\d+\.\d+', j_string)]

    H_raw = []
    
    for match in pattern.finditer(nmr_data):
        if match.group("shift_left_1") and match.group("shift_right_1"):  
            # Parse range shifts
            H_raw.append((
                match.group("type_1"), 
                parse_j_values(match.group("hz_1") or match.group("hz_alt_1")),  # Support hz and hz_alt
                int(match.group("count_1") or match.group("count_alt_1")),  # Support count and count_alt
                float(match.group("shift_left_1")), 
                float(match.group("shift_right_1"))
            ))
        elif match.group("shift_2"):  
            # Parse single shift
            H_raw.append((
                match.group("type_2"), 
                parse_j_values(match.group("hz_2") or match.group("hz_alt_2")),  # Support hz2 and hz_alt2
                int(match.group("count_2") or match.group("count_alt_2")),  # Support count2 and count_alt2
                float(match.group("shift_2")), 
                float(match.group("shift_2"))
            ))
        # elif match.group("shift_left_3") and match.group("shift_right_3"):  
        #     # Parse range shifts with new group names
        #     H_raw.append((
        #         match.group("type_3"), 
        #         parse_j_values(match.group("hz_3")), 
        #         int(match.group("count_3")), 
        #         float(match.group("shift_left_3")), 
        #         float(match.group("shift_right_3"))
        #     ))
        # elif match.group("shift_4"):  
        #     # Parse single shift with new group names
        #     H_raw.append((
        #         match.group("type_4"), 
        #         parse_j_values(match.group("hz_4")), 
        #         int(match.group("count_4")), 
        #         float(match.group("shift_4")), 
        #         float(match.group("shift_4"))
        #     ))
    
    return H_raw


def parse_h_split(h_raw: List) -> List:
    """
    Splits 1H NMR data into individual components.

    Args:
        h_nmr (list): Parsed 1H NMR data.

    Returns:
        list: List of individual components.
    """
    h_split = []
    shifts = []
    
    for item in h_raw:
        avg_shift = (item[-2] + item[-1]) / 2
        shifts.extend([avg_shift] * item[2])
        h_split.extend([item[0]] * item[2])
    
    sorted_indices = np.argsort(shifts)
    sorted_h_split = [h_split[i] for i in sorted_indices]
    
    return sorted_h_split


def parse_h_shifts(h_nmr: List) -> np.ndarray:
    """
    Extracts chemical shifts from 1H NMR data.

    Args:
        h_nmr (list): Parsed 1H NMR data.

    Returns:
        np.ndarray: Array of sorted chemical shifts.
    """
    shifts = []
    for item in h_nmr:
        avg_shift = (item[-2] + item[-1]) / 2
        shifts.extend([avg_shift] * item[2])
    return np.array(sorted(shifts))


def parse_c_shifts(nmr_data: str) -> np.ndarray:
    """
    Extracts chemical shifts from 13C NMR data, excluding parts with "J =".

    Args:
        nmr_data (str): Raw 13C NMR data.

    Returns:
        np.ndarray: Array of sorted chemical shifts.
    """
    # Extract all floating-point numbers
    all_shifts = re.findall(r"-?\d+\.\d+", nmr_data)
    # Exclude parts containing "J ="
    filtered_shifts = [
        shift for shift in all_shifts if not re.search(rf"J\s*=\s*{shift}", nmr_data) and not re.search(rf"{shift}\s*Hz", nmr_data)
    ]
    return np.array(sorted(map(float, filtered_shifts)))


def test():
    """
    Test function for validating NMR parsing functionality.
    """
    input = """
    1H NMR (400 MHz, Chloroform-d) δ 7.68 – 7.60 (m, 1H), 7.42 – 7.36 (m, 1H), 7.24 – 7.10 (m, 6H), 5.63 (t, J = 2.1 Hz, 1H), 3.58 (d, J = 2.1 Hz, 2H), 2.41 (s, 3H), 2.29 (s, 3H), 2.16 (br, 1H). 
    13C NMR (101 MHz, Chloroform-d) δ 138.9, 136.02, 135.98, 134.8(J=1.1 Hz), 130.8, 130.2, 128.41, 128.36, 127.0, 126.5, 126.30, 126.27, 84.5, 82.1, 62.7, 23.5, 19.4, 19.1.
    """
    # input = """1H NMR (CDCl3, 500 MHz): δ 7.94 (dd, 1H, J = 8.0, 1.3 Hz), 7.43 (ddd, 1H, J = 8.0, 8.0, 1.3 Hz), 7.24–7.30 (m, 1H), 7.10 (ddd, 1H, J = 8.0, 8.0, 1.3 Hz), 5.62 (d, 1H, J = 10.2 Hz), 4.32 (d, 1H, J = 10.2 Hz), 4.07 (q, 2H, J = 7.2 Hz), 3.45 (s, 3H), 2.25–2.39 (m, 2H), 1.87–2.11 (m, 4H), 1.21 (t, 3H, J = 7.2 Hz); 13C NMR (CDCl3, 126 MHz): δ 173.2 (1C), 172.9 (1C), 143.3 (1C), 140.1 (1C), 131.0 (1C), 130.1 (1C), 129.7 (1C), 100.5 (1C), 78.1 (1C), 60.3 (1C), 56.9 (1C), 33.7 (1C), 33.4 (1C), 20.0 (1C), 14.2 (1C);"""
    result = split_text(input)
    print(result)
    H_raw = parse_h_raw(result['H'])
    H_shifts = parse_h_shifts(H_raw)
    C_shifts = parse_c_shifts(result['C'])
    print("H raw data:", H_raw)
    print("H splits:", parse_h_split(H_raw))
    print("H shifts:", H_shifts)
    print("C shifts:", C_shifts)


# test()