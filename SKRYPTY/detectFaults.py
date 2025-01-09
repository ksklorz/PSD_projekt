from commonTools import *
from math import inf
import matplotlib.pyplot as plt

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
            if row['Set_PumpSpeed_P101'] > 95.0:
                FlowMaxTime += 1
                if FlowMaxTime > 5:
                    fault = True
                    faultTimes.append(tlmTime[i])
            else:
                FlowMaxTime = 0


    plt.plot(tlmTime, dane['Set_PumpSpeed_P101'])
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
    
        # leakTimes = leak_detection(data)
        cloggingTimes = clogging_detection(data)

        # leakNumber = len(leakTimes)

        # print(f'File: {fileName}')
        # print(f'Number of leaks: {leakNumber}')


if __name__ == '__main__':
    main()