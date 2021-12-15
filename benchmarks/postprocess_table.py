#!/usr/bin/env python3

# Load the data
import pandas as pd
from postprocess_base import read_logs

runs = read_logs()

# Data output
print('Writing data to \'benchmark_data.csv\'...')
runs.to_csv('benchmark_data.csv', sep='\t', index=False)
print('Writing complete')
