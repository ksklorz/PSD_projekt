from commonTools import *
from math import inf
import matplotlib.pyplot as plt
import numpy as np

from tools import myFilter

def leak_detection(dane):

    level_b101 = dane['KQ1001_Level_B101']
    level_b102 = dane['KQ1001_Level_B102']
    state = dane['State']
    level_total = level_b101 + level_b102
    tlmTime = pd.to_datetime(dane['MeasureTime'])

    fault = False
    last_state = 9
    faultTimes = []

    for i, row in dane.iterrows():
        if fault:
            if (3 == row['State']) and (2 == last_state):
                fault = False
        else:
            if level_total[i] < 190:
                fault = True
                faultTimes.append(tlmTime[i])
        last_state = row['State']


    plt.plot(tlmTime, level_total)
    for fault_time in faultTimes:
        plt.axvline(x=fault_time, color='r', linestyle='--')
    # plt.show()

    return faultTimes

def clogging_detection(dane):

    # level_b101 = dane['KQ1001_Level_B101']
    # level_b102 = dane['KQ1001_Level_B102']
    # state = dane['State']
    # level_total = level_b101 + level_b102
    tlmTime = pd.to_datetime(dane['MeasureTime'])

    fault = False
    last_state = 9
    faultTimes = []
    FlowMaxTime = 0

    for i, row in dane.iterrows():
        if fault:
            if (1 == row['State']) and (9 == last_state):
                fault = False
                FlowMaxTime = 0
        else:
            # if row['Set_PumpSpeed_P101'] > 99.0:
            if (3 == row['State']):
                error = row['Flow_FlowmeterB102'] - row['SetFlow_manual']
                if error < -0.1:
                    FlowMaxTime += 1
                    if FlowMaxTime > 5:
                        fault = True
                        faultTimes.append(tlmTime[i])
                else:
                    FlowMaxTime = 0
            else:
                FlowMaxTime = 0
        last_state = row['State']

    plt.clf
    plt.plot(tlmTime, dane['Set_PumpSpeed_P101'])
    plt.plot(tlmTime, dane['Flow_FlowmeterB102']*100)
    plt.plot(tlmTime, dane['SetFlow_manual']*100)
    for fault_time in faultTimes:
        plt.axvline(x=fault_time, color='r', linestyle='--')
    # plt.show()

    return faultTimes

def attack_detection(dane):
    tlmTime = pd.to_datetime(dane['MeasureTime'])

    fault = False
    last_state = 9
    faultTimes = []
    MaxTime = 0

    filterLow = myFilter(dt=1.0, tau = 3.0, state = 0.0)
    filterHigh = myFilter(dt=1.0, tau = 0.5, state = 0.0)

    filtered = 0.0
    FILTER = np.zeros(len(dane))
    # fLow = 0.0


    for i, row in dane.iterrows():
        filtered = filterHigh.high_pass_filter(row['Pressure_Tank103'])
        filtered = abs(filtered)
        filtered = filterLow.low_pass_filter(filtered)
        if fault:
            if (1 == row['State']) and (9 == last_state):
                fault = False
                filterLow.set_state(row['Pressure_Tank103'])
                filterHigh.set_state(row['Pressure_Tank103'])
                MaxTime = 0
        else:
            if filtered > 5:
                MaxTime += 1
                if MaxTime > 25:
                    fault = True
                    faultTimes.append(tlmTime[i])
            else:
                MaxTime = 0
        last_state = row['State']

        FILTER[i] = filtered


    plt.clf()
    plt.plot(tlmTime, dane['Pressure_Tank103'])
    plt.plot(tlmTime, FILTER*30)
    for fault_time in faultTimes:
        plt.axvline(x=fault_time, color='r', linestyle='--')
    # plt.show()

    return faultTimes

def error_detection(dane):
    tlmTime = pd.to_datetime(dane['MeasureTime'])

    fault = False
    last_state = 9
    faultTimes = []
    MaxTime = 0

    for i, row in dane.iterrows():
        if fault:
            if (1 == row['State']) and (9 == last_state):
                fault = False
                MaxTime = 0
        else:
            if (2 == row['State']) or (7 == row['State']):
                error = row['Pressure_Tank103'] - row['SetPressureTank103_manual']
                if error < -30:
                    MaxTime += 1
                    if MaxTime > 20:
                        fault = True
                        faultTimes.append(tlmTime[i])
                else:
                    MaxTime = 0
        last_state = row['State']

    plt.clf()
    plt.plot(tlmTime, dane['Pressure_Tank103'])
    plt.plot(tlmTime, dane['SetPressureTank103_manual'])
    for fault_time in faultTimes:
        plt.axvline(x=fault_time, color='r', linestyle='--')
    plt.show()

    return faultTimes



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

def main():
    fileNames = ['F1_data.csv', 'F2_data.csv', 'F3_data.csv', 'F4_data.csv']

    for fileName in fileNames:
        script_dir = os.path.dirname(__file__)
        fileName = 'DANE/' + fileName
        fileName = os.path.join(script_dir, fileName)
        cycleName = 'cycles_F1'

        tlmTime, data, cycles = read_data(fileName, cycleName, [-inf, inf])
        detectCycle(data)

        leakTimes = leak_detection(data)
        cloggingTimes = clogging_detection(data)
        attackTimes = attack_detection(data)
        errorTimes = error_detection(data)

        fault_counts = {
            'fileName': os.path.basename(fileName),
            'leakTimes': len(leakTimes),
            'cloggingTimes': len(cloggingTimes),
            'attackTimes': len(attackTimes),
            'errorTimes': len(errorTimes)
        }
        print(fault_counts)

        # leakNumber = len(leakTimes)

        # print(f'File: {fileName}')
        # print(f'Number of leaks: {leakNumber}')



if __name__ == '__main__':
    main()