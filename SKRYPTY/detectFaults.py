from commonTools import *
from math import inf
import matplotlib.pyplot as plt
import numpy as np

from tools import myFilter

def SetErrors(data, cycles):
    data['Error'] = 0
    for cycle in cycles:
        state = cycle[2]
        if state in ('zatkana rura w 80%', 'cyberatak', 'wyciek', 'uzupełnianie', 'błąd operatora'):
            data.loc[(data.index >= cycle[0]) & (data.index <= cycle[1]), 'Error'] = 1
    return data

def wskazniki_alarmow(data):
    true_positives = ((data['Error'] == 1) & (data['detectedFault'] == 1)).sum()
    false_positives = ((data['Error'] == 0) & (data['detectedFault'] == 1)).sum()
    false_negatives = ((data['Error'] == 1) & (data['detectedFault'] == 0)).sum()
    true_negatives = ((data['Error'] == 0) & (data['detectedFault'] == 0)).sum()

    if true_positives + false_negatives > 0:
        correct_alarm_rate = true_positives / (true_positives + false_negatives)
    else:
        correct_alarm_rate = 0

    if false_positives + true_negatives > 0:
        false_alarm_rate = false_positives / (false_positives + true_negatives)
    else:
        false_alarm_rate = 0

    return correct_alarm_rate, false_alarm_rate


def leak_detection(dane):

    level_b101 = dane['KQ1001_Level_B101']
    level_b102 = dane['KQ1001_Level_B102']
    state = dane['State']
    level_total = level_b101 + level_b102
    tlmTime = pd.to_datetime(dane['MeasureTime'])

    fault = False
    last_state = 9
    last_error = 0
    faultTimes = []

    for i, row in dane.iterrows():
        if fault:
            if (3 == row['State']) and (2 == last_state):
                fault = False

            if (1 == last_error) and (0 == row['Error']):
                fault = False

        else:
            if level_total[i] < 190:
                fault = True
                faultTimes.append(tlmTime[i])
        last_state = row['State']
        last_error = row['Error']
        dane.at[i, 'detectedFault'] = fault


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
    last_error = 0
    faultTimes = []
    FlowMaxTime = 0

    for i, row in dane.iterrows():
        if fault:
            if (1 == row['State']) and (9 == last_state):
                fault = False
                FlowMaxTime = 0

            if (1 == last_error) and (0 == row['Error']):
                fault = False
                FlowMaxTime = 0
        else:
            if row['Set_PumpSpeed_P101'] > 95.0:
            # if (3 == row['State']):
                error = row['Flow_FlowmeterB102'] - row['SetFlow_manual']
                if error < -0.05:
                    FlowMaxTime += 1
                    if FlowMaxTime > 5:
                        fault = True
                        faultTimes.append(tlmTime[i])
                else:
                    FlowMaxTime = 0
            else:
                FlowMaxTime = 0
        last_state = row['State']
        last_error = row['Error']
        dane.at[i, 'detectedFault'] = fault

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
    last_error = 0
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
            if ((1 == row['State']) and (9 == last_state)) or ((1 == last_error) and (0 == row['Error'])):
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
        last_error = row['Error']
        dane.at[i, 'detectedFault'] = fault

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
    last_error = 0
    faultTimes = []
    MaxTime = 0

    for i, row in dane.iterrows():
        if fault:
            if ((1 == row['State']) and (9 == last_state)) or ((1 == last_error) and (0 == row['Error'])):
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
        last_error = row['Error']
        dane.at[i, 'detectedFault'] = fault

    plt.clf()
    plt.plot(tlmTime, dane['Pressure_Tank103'])
    plt.plot(tlmTime, dane['SetPressureTank103_manual'])
    for fault_time in faultTimes:
        plt.axvline(x=fault_time, color='r', linestyle='--')
    # plt.show()

    return faultTimes


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

def leakExample():
    fileName = 'DANE/F3_data.csv'
    script_dir = os.path.dirname(__file__)
    fileName = os.path.join(script_dir, fileName)
    cycleName = 'cycles_F3'
    plt.figure(3, figsize=(10, 5))

    tlmTime, data, cycles = read_data(fileName, cycleName, [-inf, inf])
    detectCycle(data)
    SetErrors(data, cycles)
    leakTimes = leak_detection(data)
    tempTime = pd.to_datetime(data['MeasureTime'])
    # for i in range(len(leakTimes)):
    #     leakTimes[i] -= tempTime.iloc[0]

    suma = data['KQ1001_Level_B101'] + data['KQ1001_Level_B102']
    plt.clf()
    plt.plot(tempTime, suma, label='Suma poziomów')
    # for fault_time in leakTimes:
    #     plt.axvline(x=fault_time, color='r', linestyle='--', label='leak detection' if fault_time == leakTimes[0] else "")
    # cycles_inPlot(cycles, tempTime[0])
    offset = suma.min()
    skala = suma.max() - offset
    plt.plot(tempTime, data['Error']*skala+offset, color = 'k', linestyle = '--', label = 'Rzeczywisty błąd')
    plt.plot(tempTime, data['detectedFault']*skala+offset, color = 'r', linestyle = '--', label = 'Wykryty błąd')

    TP, FP = wskazniki_alarmow(data)
    print(f'TP: {TP}, FP: {FP}')

    plt.legend()
    plt.xlabel('Czas')
    plt.ylabel('Poziom z biornikach, cm')
    plt.title('Wykrycie wycieku')
    plt.grid(True)
    # plt.show()
    plt.savefig( os.path.join(script_dir, 'FIG/leak_detection_wskazniki.pdf'))

def attackExample():
    fileName = 'DANE/F2_data.csv'
    script_dir = os.path.dirname(__file__)
    fileName = os.path.join(script_dir, fileName)
    cycleName = 'cycles_F2'
    plt.figure(3, figsize=(10, 5))

    tlmTime, data, cycles = read_data(fileName, cycleName, [-inf, inf])
    detectCycle(data)
    SetErrors(data, cycles)
    leakTimes = attack_detection(data)
    tempTime = pd.to_datetime(data['MeasureTime'])
    # for i in range(len(leakTimes)):
    #     leakTimes[i] -= tempTime.iloc[0]

    plt.clf()
    plt.plot(tempTime, data['Pressure_Tank103'], label = 'Ciśnienie w zbiorniku')
    # for fault_time in leakTimes:
    #     plt.axvline(x=fault_time, color='r', linestyle='--', label='fault detection' if fault_time == leakTimes[0] else "")
    # cycles_inPlot(cycles, tempTime[0])
    suma = data['Pressure_Tank103']
    offset = suma.min()
    skala = suma.max() - offset
    plt.plot(tempTime, data['Error']*skala+offset, color = 'k', linestyle = '--', label = 'Rzeczywisty błąd')
    plt.plot(tempTime, data['detectedFault']*skala+offset, color = 'r', linestyle = '--', label = 'Wykryty błąd')

    TP, FP = wskazniki_alarmow(data)
    print(f'TP: {TP}, FP: {FP}')

    plt.legend()
    plt.xlabel('Czas')
    plt.ylabel('ciśnienie w zbiorniku, mbar')
    plt.title('Wykrycie cyberataku')
    plt.grid(True)
    # plt.show()
    plt.savefig( os.path.join(script_dir, 'FIG/attack_detection_wskazniki.pdf'))

def errorExample():
    fileName = 'DANE/F4_data.csv'
    script_dir = os.path.dirname(__file__)
    fileName = os.path.join(script_dir, fileName)
    cycleName = 'cycles_F4'
    plt.figure(3, figsize=(10, 5))

    tlmTime, data, cycles = read_data(fileName, cycleName, [-inf, inf])
    detectCycle(data)
    SetErrors(data, cycles)
    leakTimes = error_detection(data)
    tempTime = pd.to_datetime(data['MeasureTime'])


    plt.clf()
    plt.plot(tempTime, data['Pressure_Tank103'], label = 'Ciśnienie w zbiorniku')
    plt.plot(tempTime, data['SetPressureTank103_manual'], label = 'zadane ciśnienie')
    # for fault_time in leakTimes:
    #     plt.axvline(x=fault_time, color='r', linestyle='--', label='fault detection' if fault_time == leakTimes[0] else "")
    # cycles_inPlot(cycles, tempTime[0])

    suma = data['Pressure_Tank103']
    offset = suma.min()
    skala = suma.max() - offset
    plt.plot(tempTime, data['Error']*skala+offset, color = 'k', linestyle = '--', label = 'Rzeczywisty błąd')
    plt.plot(tempTime, data['detectedFault']*skala+offset, color = 'r', linestyle = '--', label = 'Wykryty błąd')

    TP, FP = wskazniki_alarmow(data)
    print(f'TP: {TP}, FP: {FP}')

    plt.legend()
    plt.xlabel('Czas')
    plt.ylabel('ciśnienie w zbiorniku, mbar')
    plt.title('Wykrycie błędu operatora')
    plt.grid(True)
    # plt.show()
    plt.savefig( os.path.join(script_dir, 'FIG/error_detection_wskazniki.pdf'))

def clogginExample():
    fileName = 'DANE/F1_data.csv'
    script_dir = os.path.dirname(__file__)
    fileName = os.path.join(script_dir, fileName)
    cycleName = 'cycles_F1'
    plt.figure(3, figsize=(10, 5))

    tlmTime, data, cycles = read_data(fileName, cycleName, [-inf, inf])
    detectCycle(data)
    SetErrors(data, cycles)
    leakTimes = clogging_detection(data)
    tempTime = pd.to_datetime(data['MeasureTime'])


    plt.clf()
    plt.plot(tempTime, data['Flow_FlowmeterB102'], label='Przepływ B102')
    plt.plot(tempTime, data['SetFlow_manual'], label='Przepływ zadany')
    plt.plot(tempTime, data['Set_PumpSpeed_P101']/100.0, label='Prędkość pompy P101')
    # for fault_time in leakTimes:
    #     plt.axvline(x=fault_time, color='r', linestyle='--', label='fault detection' if fault_time == leakTimes[0] else "")
    # cycles_inPlot(cycles, tempTime[0])

    suma = data['Flow_FlowmeterB102']
    offset = suma.min()
    skala = suma.max() - offset
    plt.plot(tempTime, data['Error']*skala+offset, color = 'k', linestyle = '--', label = 'Rzeczywisty błąd')
    plt.plot(tempTime, data['detectedFault']*skala+offset, color = 'r', linestyle = '--', label = 'Wykryty błąd')

    TP, FP = wskazniki_alarmow(data)
    print(f'TP: {TP}, FP: {FP}')

    plt.legend()
    plt.xlabel('Czas')
    plt.ylabel('Przepływ L/min')
    plt.title('Wykrycie przytkania')
    plt.grid(True)
    # plt.show()
    plt.savefig( os.path.join(script_dir, 'FIG/clogging_detection_wskazniki.pdf'))


if __name__ == '__main__':
    # main()
    leakExample()
    attackExample()
    errorExample()
    clogginExample()