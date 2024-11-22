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