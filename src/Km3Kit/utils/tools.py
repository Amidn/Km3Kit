import time 


def report_time_interval(start, label="read data"):
    # Capture end time
    end_time = time.time()
    # Calculate interval in seconds
    time_interval = end_time - start
    # Convert to hours, minutes, seconds
    hours, remainder = divmod(time_interval, 3600)
    minutes, seconds = divmod(remainder, 60)
    # Print the result
    print(f">>>>>>>>>>>>>>>>>>>> Time interval up to {label}: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
