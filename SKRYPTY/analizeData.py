import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from math import inf
from commonTools import *


def clogging_detection(timeConstraints = [-inf, inf]):
    """
    Detects clogging in a pipe based on flow and pump speed data within a specified time range.
    Parameters:
    interestinTime (list): A list containing two elements [start_time, end_time] in seconds. 
                           The function will analyze data within this time range. 
                           Default is [-inf, inf], which means the entire dataset.
    Returns:
    None: The function generates a plot showing the pump speed and flow rate over time, 
          highlights cycles, and saves the plot as a PDF file.
    The function performs the following steps:
    1. Reads data from 'DANE/F1_data.csv'.
    2. Parses cycle data from 'cycles_F1'.
    3. Converts 'MeasureTime' to seconds and adjusts it relative to the first measurement.
    4. Filters the data based on the specified time range.
    5. Plots the pump speed and flow rate over time.
    6. Highlights cycles on the plot.
    7. Saves the plot as 'clogging_detection_<start_time>_<end_time>.pdf'.
    8. Displays the plot.
    """
    fileName = 'DANE/F1_data.csv'
    cycleName = 'cycles_F1'

    tlmTime, data, cycles = read_data(fileName, cycleName, timeConstraints)

    flow_SETspeed_P101 = data['Set_PumpSpeed_P101']/100.0
    flow_B102 = data['Flow_FlowmeterB102']

    plt.figure(1, figsize=(10, 5))
    plt.plot(tlmTime, flow_SETspeed_P101, label='Speed P101, 1/1')
    plt.plot(tlmTime, flow_B102, label='Flow B102 L/min')
    cycles_inPlot(cycles)

    plt.xlabel('Czas, s')
    plt.ylabel('Zmienne procesowe')
    plt.title('Wykrycie przytkania rury')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=5)
    plt.grid(True)
    plt.savefig(f'FIG\\clogging_detection_{timeConstraints[0]}_{timeConstraints[1]}.pdf')
    # plt.show()
    plt.clf()

def leak_detection(timeConstraints = [-inf, inf]):
    
    fileName = 'DANE/F3_data.csv'
    cycleName = 'cycles_F3'
    tlmTime, data, cycles = read_data(fileName, cycleName, timeConstraints)

    level_b101 = data['KQ1001_Level_B101']
    level_b102 = data['KQ1001_Level_B102']
    level_total = level_b101 + level_b102

    plt.figure(figsize=(10, 5))
    plt.plot(tlmTime, level_b101, label='Poziom B101')
    plt.plot(tlmTime, level_b102, label='Poziom B102')
    plt.plot(tlmTime, level_total, label='Suma poziomów')
    cycles_inPlot(cycles)

    plt.xlabel('Czas, s')
    plt.ylabel('Poziomy, mm')
    plt.title('Wykrycie wycieku')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=5)
    plt.grid(True)
    plt.savefig(f'FIG\\leak_detection_{timeConstraints[0]}_{timeConstraints[1]}.pdf')
    # plt.show()
    plt.clf()



def humanError_detection(timeConstraints = [-inf, inf]):
    fileName = 'DANE/F4_data.csv'
    cycleName = 'cycles_F4'

    tlmTime, data, cycles = read_data(fileName, cycleName, timeConstraints)

    pressure_B103 = data['Pressure_Tank103']
    flow_SETspeed_P101 = data['Set_PumpSpeed_P101']

    plt.figure(3, figsize=(10, 5))
    plt.plot(tlmTime, pressure_B103, label='Ciśnienie B103, mbar')
    plt.plot(tlmTime, flow_SETspeed_P101, label='Prędkość zadana pompy P101, %')

    cycles_inPlot(cycles)

    plt.xlabel('Czas, s')
    plt.ylabel('Zmienne procesowe')
    plt.title('Wykrycie błędu operatora')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=5)
    plt.grid(True)
    plt.savefig(f'FIG\\error_detection_{timeConstraints[0]}_{timeConstraints[1]}.pdf')
    # plt.show()
    plt.clf()
    

def attack_detection(timeConstraints = [-inf, inf]):
    fileName = 'DANE/F2_data.csv'
    cycleName = 'cycles_F2'

    tlmTime, data, cycles = read_data(fileName, cycleName, timeConstraints)

    pressure_B103 = data['Pressure_Tank103']
    flow_SETspeed_P101 = data['Set_PumpSpeed_P101']

    plt.figure(3, figsize=(10, 5))
    plt.plot(tlmTime, pressure_B103, label='Ciśnienie B103, mbar')
    plt.plot(tlmTime, flow_SETspeed_P101, label='Prędkość zadana pompy P101, %')

    cycles_inPlot(cycles)

    plt.xlabel('Czas, s')
    plt.ylabel('Zmienne procesowe')
    plt.title('Wykrycie cyberataku')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=5)
    plt.grid(True)
    plt.savefig(f'FIG\\attack_detection_{timeConstraints[0]}_{timeConstraints[1]}.pdf')
    # plt.show()
    plt.clf()
    

    #####################################################################################################################

def cycles_inPlot(cycles):
    yLimits = plt.ylim()
    xLimits = plt.xlim()
    H = yLimits[1] - (yLimits[1] - yLimits[0]) *0.03
    for cycle in cycles:
        if (cycle[1] > xLimits[0] and cycle[1] < xLimits[1]):
            plt.axvline(x=cycle[1], color='k', linestyle='--', linewidth=0.8)
            plt.text(cycle[0], H, cycle[2], rotation=0, color='k', fontsize=6)

def main():
    clogging_detection([1700, 2280]) # rura zatkana 80%
    clogging_detection([550, 1130]) # zatkana rura w 40%

    leak_detection() #cały przekrój
    leak_detection([0, 850]) #jeden cykl
    
    attack_detection([1190, 1750]) #jeden cykl

    humanError_detection([550, 1100]) #jeden cykl
    

if __name__ == "__main__":
    main()