import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import glob
import import_ipynb

import base_code1

df_1 = base_code1.read_in_files(file_path="/Users/student/Library/CloudStorage/OneDrive-DurhamUniversity/Team project/TeamRepo/TeamProject-1/50_hz/*.csv",
                        f=50, # Change this 
                        time_error=0.001, # This should be in the units which your original time column is in, the conversion is accounted for in the code
                        time_conversion=1e-3,
                        B_volts_conversion=1,
                        C_volts_conversion=1)

df_2 = base_code1.B_H_calculation(df=df_1,
                            time_column='Time (s)',
                            time_error_column='Time Error (s)',
                            voltage_column='Channel C Average (V)',
                            voltage_error_column='Channel C Average Error (V)',
                            voltage_current_column='Channel B Average (V)',
                            voltage_current_error_column='Channel B Average Error (V)',
                            V_0=0,
                            N_2=15,# Change this
                            N_1=5, # Change this
                            A=8.93e-4,
                            A_error=5e-5,
                            R=0.6,
                            R_error=0.001,
                            l=0.32,
                            l_error=0.01) # Change this

final_answer = base_code1.B_H_plotter(df=df_2,
                     show=True,
                     freq=50, # Change this 
                     freq_error=0.0001,
                     A=8.93e-4,
                     A_error=1e-4,
                     l=0.32,
                     l_error=0.011)

print(final_answer)



