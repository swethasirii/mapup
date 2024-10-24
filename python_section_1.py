from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
   result = []
    length = len(lst)

    for i in range(0, length, n):
        
        end = min(i + n, length)
        group = lst[i:end]
        
        
        reversed_group = []
        for j in range(len(group)):
            reversed_group.append(group[len(group) - 1 - j])
        
        result.extend(reversed_group)
    
    lst = result 
    return lst


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {}

    for string in lst:
        length = len(string)
        if length not in result:
            result[length] = []
        result[length].append(string)

   
    sorted_result = dict(sorted(result.items()))
    
    return sorted_result 

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
     items = {}

        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.update(flatten(value, new_key))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        items.update(flatten(item, f"{new_key}[{index}]"))
                    else:
                        items[f"{new_key}[{index}]"] = item
            else:
                items[new_key] = value

        return items
    
    return flatten(nested_dict)
    

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])  
            return
        
        seen = set()  
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue 
            seen.add(nums[i])
            nums[start], nums[i] = nums[i], nums[start]  # Swap
            backtrack(start + 1)  # Recurse
            nums[start], nums[i] = nums[i], nums[start]  # Backtrack (swap back)

    nums.sort() 
    result = []
    backtrack(0)  
    return result
    
    pass


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    pass

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str)

    
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    
    distances = [0]  # First distance is always 0
    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i - 1]
        lat2, lon2 = df.iloc[i]
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)

    df['distance'] = distances
    return df
    


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    
    # Step 1: Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    
    # Step 2: Multiply each element by the sum of its original row and column indices
    transformed_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            original_row_index = i
            original_col_index = j
            index_sum = original_row_index + original_col_index
            transformed_matrix[i][j] = rotated_matrix[i][j] * index_sum

    return transformed_matrix
  


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    
    grouped = df.groupby(['id', 'id_2'])
    
    results = []
    
    for (id_val, id_2_val), group in grouped:
        
        if group.empty:
            results.append(((id_val, id_2_val), True))
            continue
        
        
        full_range = pd.date_range(start=group['start'].min().floor('D'),
                                    end=group['end'].max().ceil('D'),
                                    freq='D')

   
        has_complete_days = all(any((group['start'].dt.floor('D') <= day) & 
                                     (group['end'].dt.ceil('D') >= day) & 
                                     ((group['end'] - group['start']).dt.total_seconds() >= 86400))
                                 for day in full_range)
       
        days_present = group['start'].dt.dayofweek.unique()
        has_all_days = len(days_present) == 7

    
        results.append(((id_val, id_2_val), not (has_complete_days and has_all_days)))

    
    result_series = pd.Series(dict(results))
    return result_series

