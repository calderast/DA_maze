# Function definitions for turnaround analysis

import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt

def normalize(data):
    """ Given an array of numerical data, returns that data scaled between 0-1 """
    
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def scale_to_bounds(data, bounds):
    """ Scales the values of the 'data' list to match the min and max of the 'bounds' list

    Parameters:
    data (list): The data list to be scaled.
    bounds (list): The target bounds to which the data should be scaled.

    Returns:
    scaled_data (list): The scaled data list within the specified bounds.
    """

    return (data-np.min(data)) / (np.max(data)-np.min(data)) * (np.max(bounds)-np.min(bounds)) + np.min(bounds)


def get_bounds(df):
    """ Given a dataframe with x and y fields, returns min(x), min(y), max(x), max(y) """
    
    return np.nanmin(df.x), np.nanmin(df.y), np.nanmax(df.x), np.nanmax(df.y)


def expand_second_list(first_list, second_list):
    """ Expand the second list to match the size of the first list.

    Args:
    first_list (list): The list with counts or indices.
    second_list (list): The list to be expanded.

    Returns:
    list: The expanded second list.

    Example:
    >>> first_list = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3]
    >>> second_list = [4, 5, 6, 7]
    >>> expanded_second_list = expand_second_list(first_list, second_list)
    >>> print(expanded_second_list)
    [4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7]
    """
    
    expanded_second_list = [second_list[val] for val in first_list]
    return expanded_second_list


def moving_average(data, window_size=125):
    """ Smooth a jumpy signal using a simple moving average filter.
    
    Parameters:
    data (list): List of values to be smoothed.
    window_size (int): Size of the moving average window. Default of 125 = 0.5s (at 250 Hz sample frequency)
    """
    
    if window_size <= 0 or window_size >= len(data):
        raise ValueError("Invalid window size")
        
    smoothed_data = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        end = i + 1
        window = data[start:end]
        smoothed_value = sum(window) / len(window)
        smoothed_data.append(smoothed_value)

    return smoothed_data


def convert_indices_to_time(indices, start_point=0, sample_frequency=250):
    """ Given an array of indices and a sampling frequency (Hz), return corresponding time array (s)
    
    The returned time array will start from 0 seconds by default (better for plotting)
    start_point = -1 means the start time of the returned array will not be adjusted
    """
    
    seconds_per_sample = 1/sample_frequency
    if start_point == -1:
        return indices*seconds_per_sample
    else: # if not explicity set to -1, I choose to assume we want to start the time from 0 
        return (indices-np.nanmin(indices))*seconds_per_sample
    

def find_dead_ends(tmat):
    """ Given a transition matrix, returns a list of hexes that are dead ends """
    
    dead_ends = np.unique(np.where(tmat==1)[0])
    dead_ends = np.delete(dead_ends,np.isin(dead_ends,[1,2,3]))
    
    return dead_ends


def path_to_dead_end(tmat, next_hex, path=[]):
    """ Given a transition matrix and a dead end hex, recursively finds the path of hexes to that dead end 
    
    Note: Third argument [] or name of empty list is required to ensure you get a new list 
    instead of modifying the same list from the last time this function was run
    """

    path.append(next_hex)
    next_hexes = np.where(tmat[:, next_hex]==0.5)[0]
    
    for hex in next_hexes:
        if hex not in path and hex not in [1,2,3]:
            path_to_dead_end(tmat, hex, path)
            
    return list(path)


def get_all_dead_end_paths(tmat, min_length=1):
    """ Given a transition matrix, get all paths to dead end hexes.
    
    Parameters: 
    tmat: the transition matrix
    min_length (int): the minimum length of dead end path to include
    
    Returns: a list of lists, where each list starts with a dead end hex and 
    includes the path (in order) of hexes to that dead end.
    """

    all_dead_end_paths = []
    dead_ends = find_dead_ends(tmat)
    
    for hex in dead_ends:
        path = path_to_dead_end(tmat, hex, [])
        if len(path) >= min_length:
            all_dead_end_paths.append(path)
        
    return list(all_dead_end_paths)


def find_dead_end_path_for_hex(hexes, all_dead_end_paths):
    """ Returns the dead end path (or list of paths) a hex (or hexes) is in, or [] if not in a dead end.
    
    Parameters: 
    hexes: A list of hexes, or a single hex
    all_dead_end_paths: List of all dead end paths in the maze (in hexes)
    
    Returns: 
    A list of dead end paths the same length as hexes, where each entry is the dead end path that the hex is in
    
    (I use this to add a column of dead end paths to the dataframe so I can group by dead end)
    """
    
    # if we only want to check a single hex, that's fine
    if isinstance(hexes, int):
        return next((path for path in all_dead_end_paths if hexes in path), [])
    
    # otherwise, loop through all hexes and return a list of dead end paths for each hex
    result = []
    for hex in hexes:
        dead_end_path = next((path for path in all_dead_end_paths if hex in path), [])
        result.append(dead_end_path)
        
    return result


def divide_into_sections(arr):
    """ Divides a sorted list of numbers into sections of consecutive numbers increasing by 1.
    
    Parameters: 
    arr (list): A sorted list of numbers with potential breaks. 
    
    Returns: 
    result (list): A list where each section of increasing numbers is represented by consecutive integers.
    
    (I use this function to find each distinct time a rat enters a dead end path.)

    Example:
    >>> input_array = [7, 8, 9, 10, 11, 55, 56, 57, 58, 59, 60, 61, 990, 991, 992, 993]
    >>> result_array = divide_into_sections(input_array)
    >>> print(result_array)
    [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3]
    """

    result = [1]*len(arr)
    current_section = 1
    
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1] + 1:
            current_section += 1
        result[i] = current_section
        
    return result


def count_entries_for_each_dead_end(dead_end_paths, list_of_entries):
    """ Count the number of times the rat entered a specific dead end

    Args:
    dead_end_paths (list): A list representing the areas a person entered
    list_of_entries (list): A list representing distinct instances of area entry

    Returns:
    result: A list of counts, where each count represents how many times the rat entered a specific dead end
    dead_end_counts: A dictionary representing the number of times the rat entered the path to each dead end
    """
    
    result = []
    dead_end_counts = {}
    distinct_entries = set()
    
    # replace each dead end path with just the dead end hex to make things easier
    dead_ends = [sublist[0] for sublist in dead_end_paths]
    
    for de, entry in zip(dead_ends, list_of_entries):    
        # if this is a new entry into a dead end, start or add to the entry count for this dead end
        if entry not in distinct_entries:
            if de not in dead_end_counts: 
                dead_end_counts[de] = 1
            else:
                dead_end_counts[de] += 1
            # add this to our set of distinct entries so we don't double count it
            distinct_entries.add(entry)
            
        result.append(dead_end_counts[de])
        
    return result, dead_end_counts


def find_turnaround_index_hexes(rats_hex_path, dead_end_path):
    """ Find the index where the rat turned around when going down a dead end path, using hexes.
    
    Parameters: 
    rats_hex_path: The path the rat took through the dead end (in hexes)
    dead_end_path: The dead end path that the rat is in (hexes)
    
    Returns: 
    entered_index, exited_index: indices of the last time the rat entered and exited the furthest hex in the path
    """
    
    # find the furthest hex along the dead end path we get to before turning around
    for hex in dead_end_path:
        if hex in rats_hex_path.values:
            furthest_hex = hex
            break 
    
    # we didn't get past the first hex, entered/exited indices are just the first and last indices
    first_hex = rats_hex_path.values[0]
    if first_hex == furthest_hex:
        entered_index = 0
        exited_index = len(rats_hex_path)-1
        return entered_index, exited_index
    
    # otherwise, find the indices of the last time we entered and exited the furthest hex
    turnaround_indices = []
    for i in range(len(rats_hex_path) - 1):
        if rats_hex_path.values[i] != furthest_hex and rats_hex_path.values[i+1] == furthest_hex:
            entered_index = i+1
        elif rats_hex_path.values[i] == furthest_hex and rats_hex_path.values[i+1] != furthest_hex:
            exited_index = i
    
    return entered_index, exited_index


def find_furthest_point_from_endpoints(x_coords, y_coords):
    """ Finds the index of the (x,y) coordinate that is furthest from both the first and last coordinates.
    
    Parameters: 
    x_coords: List of x coordinates
    y_coords: List of y coordinates

    Returns:
    furthest_index (int): The index of the furthest coordinate
    furthest_coordinate (tuple): The (x,y) coordinate that is furthest from both endpoints
    """
    
    start_x, start_y = x_coords.iloc[0], y_coords.iloc[0]
    end_x, end_y = x_coords.iloc[-1], y_coords.iloc[-1]

    max_combined_distance = 0
    furthest_coordinate = []
    furthest_index = []

    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        distance_to_start = math.sqrt((x - start_x) ** 2 + (y - start_y) ** 2)
        distance_to_end = math.sqrt((x - end_x) ** 2 + (y - end_y) ** 2)

        combined_distance = distance_to_start + distance_to_end
        if combined_distance > max_combined_distance:
            max_combined_distance = combined_distance
            furthest_coordinate = (x, y)
            furthest_index = i

    furthest_index = furthest_index + min(x_coords.index) # adjust for index in the dataframe

    return furthest_index, furthest_coordinate


def find_dead_end_turnaround(x_coords, y_coords, rats_hex_path, dead_end_path):
    """ Finds the index where the rat turned around in a dead end path """
    
    # find where the rat entered and exited the furthest hex in the dead end path
    entered_index, exited_index = find_turnaround_index_hexes(rats_hex_path, dead_end_path)
    
    # find the furthest point in that hex from the entry and exit points
    x = x_coords[entered_index:exited_index]
    y = y_coords[entered_index:exited_index]
    turnaround_index, turnaround_coordinate = find_furthest_point_from_endpoints(x, y)
    
    return turnaround_index


def stats_for_dead_end_entry(rats_hex_path):
    """ Given the rat's path in a dead end (in hexes), return hexes traveled and time spent in this dead end """
    
    hexes_traveled = len(set(rats_hex_path))
    time_spent = len(rats_hex_path)*1/250 # time (s) = number of samples * seconds per sample
    
    return hexes_traveled, time_spent


def get_stats(df):
    """ Given a dataframe, iterates through each dead end entry and calculates the max hexes traveled and
    the time spent for each dead end entry. Returned lists have one entry for each entry into a dead end.
    
    Returns:
    hexes_traveled (list): Number of hexes traveled for each dead end entry
    time_spent (list): The amount of time (in seconds) spent in that dead end entry
    """
    
    num_dead_end_entries = max(df.dead_end_entry)
    hexes_traveled = [0]*(num_dead_end_entries)
    time_spent = [0]*(num_dead_end_entries)
    
    for entry in list(range(1, num_dead_end_entries+1)):
        hexes = df.hexlabels[df.dead_end_entry==entry]
    
        # get stats for this entry
        hexes_traveled[entry-1], time_spent[entry-1] = stats_for_dead_end_entry(hexes)
    
    return hexes_traveled, time_spent


def get_centered_indices(df):
    """ Given a dataframe, loops through each dead end entry and calculates new indices for that entry
    centered around the rat's turnaround point in that dead end. For example, the rat's turnaround point
    would now have index 0, indices leading up to that point are ..., -2, -1, and indices following the 
    turnaround point are 1, 2, ...
    
    Returns:
    centered_indices (list): list of the same length as the dataframe indicating the centered indices for
    each of the dead end entries """
    
    num_dead_end_entries = max(df.dead_end_entry)
    centered_indices = []
    for entry in list(range(1, num_dead_end_entries+1)):
        entry_df = df[df.dead_end_entry==entry]
        dead_end_path = entry_df.dead_end_path.iloc[0]
        turnaround_index = find_dead_end_turnaround(entry_df.x, entry_df.y, entry_df.hexlabels, dead_end_path)
        indices = (entry_df.index - turnaround_index)
        centered_indices += indices.tolist()
    return centered_indices


def get_mean_and_std(df, data_name, start_index, end_index):
    """ Messy function for now to take a dataframe, desired data (ex 'green_z_scored'), 
    and return the mean and standard deviation over all dead end entries, centered around
    the dead end turnaround point. 
    """
    
    filtered_df = df[(df['centered_indices'] >= start_index) & (df['centered_indices'] <= end_index)]
    num_dead_end_entries = max(df.dead_end_entry)+1
    
    # Create a common index range from start to end
    common_indices = np.arange(start_index, end_index + 1)
    extended_lists = []

    # Extend and interpolate values for each list to match the common index range
    for entry in list(range(1, num_dead_end_entries)):
        data = filtered_df[data_name][filtered_df.dead_end_entry==entry]
        valid_indices = filtered_df.centered_indices[filtered_df.dead_end_entry==entry]
    
        # Create an array of NaN values with the length of the common index range
        extended_values = np.full(end_index - start_index + 1, np.nan)

        # Assign the valid values to the corresponding positions in the extended array
        extended_values[valid_indices - start_index] = data
        extended_lists.append(extended_values)
    
    mean_array = np.nanmean(extended_lists, axis=0)
    std_array = np.nanstd(extended_lists, axis=0)
    
    return mean_array, std_array


def plot_mean_with_error(mean_array, error_array, start_index=-500, end_index=500, ylabel=None):
    """ Given a mean array and error array (SD, SEM, etc), plots the mean with the error as a shaded region
    
    Optionally specify the start index, end index, and y axis label. Indices for the x axis will
    automatically be converted to time assuming a sampling frequency of 250 Hz """
    
    upper_bound = mean_array + error_array
    lower_bound = mean_array - error_array
    common_indices = np.arange(start_index, end_index + 1)
    time = convert_indices_to_time(common_indices, -1)

    plt.figure()
    plt.plot(time, mean_array, linewidth=2)
    plt.fill_between(time, lower_bound, upper_bound, color='blue', alpha=0.2)
    plt.xlabel('Time (s)')
    if ylabel is not None:
        plt.ylabel(ylabel)