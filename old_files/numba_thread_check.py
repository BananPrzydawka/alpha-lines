import numba, os
def main():
    print("numba threads:", numba.config.NUMBA_NUM_THREADS)
    print("os.cpu_count():", os.cpu_count())