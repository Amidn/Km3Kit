import time 
import psutil

def report_time_interval(start, label="read data", verbose=False):
    if verbose:
        end_time = time.time()
        time_interval = end_time - start
        hours, remainder = divmod(time_interval, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f">>>>>>>>>>>>>>>>>>>> Time interval up to {label}: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")

def report_memory_usage(label="process", max_memory=0, verbose=False):
    if verbose:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_used = memory_info.rss / (1024 ** 2)  # Convert bytes to MB
        max_memory = max(max_memory, memory_used)  # Update maximum memory usage
        print(f">>>>>>>>>>>>>>>>>>>> Memory usage during {label}: {memory_used:.2f} MB")
    return max_memory

def diagnose_dataframe(df):
    """
    Diagnoses a DataFrame for potential issues when saving to HDF5.

    Args:
        df (pd.DataFrame): The DataFrame to diagnose.

    Returns:
        None
    """
    print("Diagnosing DataFrame for potential issues...")
    
    for col in df.columns:
        print(f"\nColumn: '{col}'")
        print(f"  Data type: {df[col].dtype}")
        unique_types = df[col].apply(type).unique()
        print(f"  Unique data types in column: {unique_types}")

        if pd.api.types.is_object_dtype(df[col]):
            print(f"  Issue: Column '{col}' is of type 'object' (likely mixed types).")
        elif pd.api.types.is_integer_dtype(df[col]):
            max_val = df[col].max()
            if max_val > np.iinfo(np.int32).max:
                print(f"  Issue: Column '{col}' contains large integers (max: {max_val}).")
        elif pd.api.types.is_float_dtype(df[col]):
            print(f"  Column '{col}' is a valid float column.")
        elif pd.api.types.is_string_dtype(df[col]):
            print(f"  Column '{col}' is a valid string column.")
        else:
            print(f"  Issue: Column '{col}' has an unsupported or unusual data type.")
    
    print("\nDiagnosis complete.")
