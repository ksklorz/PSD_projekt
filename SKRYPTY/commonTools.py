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

def cycles_getFaults(cycles,timeZero):
    faults = []
    for cycle in cycles:
        if 'awaria' in cycle[2].lower():
            faults.append(cycle)
    return faults

def detectCycle(dane):

    V101 = dane['V101_manual']
    V102 = dane['set_valve_y102_open_manual']
    V103 = dane['V103_manual']
    V104 = dane['V104_manual']
    V106 = dane['set_valve_v106_open_manual']
    pumpSpeed = dane['Set_PumpSpeed_P101']
    V108 = dane['V108_manual']
    V112 = dane['V112_manual']

    state = 1
    States = [0] * len(dane)
    for index, row in dane.iterrows():
        if state == 1:
            if V103[index] and V108[index]:
                state = 2
        elif state == 2:
            if V104[index]:
                state = 3
        elif state == 3:
            if V101[index]:
                state = 4
        elif state == 4:
            if 0 == pumpSpeed[index]:
                state = 5
        elif state == 5:
            if V102[index]:
                state = 6
        elif state == 6:
            if V106[index] and V108[index]:
                state = 7
        elif state == 7:
            if V101[index]:
                state = 8
        elif state == 8:
            if V102[index] and V112[index]:
                state = 9
        elif state == 9:
            if 0 == V102[index]:
                state = 1


        States[index] = state

    dane['State'] = States
    return States