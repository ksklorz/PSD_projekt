import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def read_data(file_path, cycle_name, timeConstraints):
    data = pd.read_csv(file_path)
    cycles = parse_cycles(cycle_name)
    
    tlmTime = pd.to_datetime(data['MeasureTime'])
    tlmTime = tlmTime.dt.hour * 3600 + tlmTime.dt.minute * 60 + tlmTime.dt.second
    tlmTime = tlmTime - tlmTime.iloc[0]

    isOK = (tlmTime >= timeConstraints[0]) & (tlmTime <= timeConstraints[1])
    data = data[isOK]
    tlmTime = tlmTime[isOK]

    return tlmTime, data, cycles

def parse_cycles(cycle_name):
    script_dir = os.path.dirname(__file__)
    file_path = 'DANE/cycles.txt'
    file_path = os.path.join(script_dir, file_path)
    with open(file_path, 'r', encoding = "utf-8") as file:
        lines = file.readlines()
    
    cycles = []
    parse = False
    for line in lines:
        if line.startswith(cycle_name):
            parse = True
            line = line.split('=')[1].strip()

        if line.strip() == '':
            parse = False
        
        if parse :
            line = line.replace("]]", ',')
            cycle_data = line.replace('[', '').replace(']', '').split(',')
            start = int(cycle_data[0].strip())
            end = int(cycle_data[1].strip())
            text = cycle_data[2].strip()
            text = text.replace('#', '')
            cycles.append((start, end, text))
        
    return cycles

def cycles_inPlot(cycles, timeZero):
    yLimits = plt.ylim()
    xLimits = plt.xlim()
    cyclesTS = [timeZero]*len(cycles)
    for i in range(len(cycles)):
        cyclesTS[i] = timeZero + timedelta(seconds=cycles[i][0])

    H = yLimits[1] - (yLimits[1] - yLimits[0]) *0.03
    for i in range(len(cycles)):
        # if (cyclesTS[i] > xLimits[0] and cyclesTS[i] < xLimits[1]):
            plt.axvline(x=cyclesTS[i], color='k', linestyle='--', linewidth=0.8)
            plt.text(cyclesTS[i], H, cycles[i][2], rotation=0, color='k', fontsize=6)