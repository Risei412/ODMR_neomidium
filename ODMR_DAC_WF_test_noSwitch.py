# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 20:39:42 2025

@author: Bridgman
"""

# -*- coding: utf-8 -*-

from scipy.optimize import curve_fit  # （未使用でもとりあえず残し）
import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
import time
import matplotlib.pyplot as plt
from windfreak import SynthHD
import csv
import os

# ========= ユーザ設定 =========
COM_PORT      = 'COM9'      # Windfreak
AI_CH         = "Dev3/ai0"  # DAQアナログ入力
MW_CH         = 0           # SynthHDチャネル (A=0, B=1)

# MW スイープ
f_start = 2.82e9   # Hz
f_span  = 1e8      # Hz
f_end   = f_start + f_span
num     = 200      # 周波数ポイント数

# DAQ 取得設定（各周波数での積算）
SAMPLE_RATE = 2000000      # [Hz] サンプリングレート
N           = 100        # 各周波数でのサンプル数（＝積算点数）
READ_TIMEOUT = 10.0     # [s] DAQmx read のタイムアウト
MW_SETTLE_S  = 0.002    # [s] 周波数設定後の待ち（必要に応じて調整）

# 出力
out_dir = "./Data"
os.makedirs(out_dir, exist_ok=True)
_file_name = time.strftime('%Y%m%d_%H%M%S')  # csv file output
print(_file_name)

# ========= 準備 =========
# 周波数配列と結果格納
f_list = np.linspace(f_start, f_end, num)
fluo_mean = np.zeros(num)
fluo_std  = np.zeros(num)

# Windfreak 準備
synth = SynthHD(COM_PORT)
synth.init()
synth[MW_CH].power = -10  # dBm
synth[MW_CH].enable = True

# DAQ タスク準備（有限サンプル取得用）
task = nidaqmx.Task()
task.ai_channels.add_ai_voltage_chan(AI_CH)
# タイミングは都度組み直す必要はないが、明示しておく
task.timing.cfg_samp_clk_timing(
    rate=SAMPLE_RATE,
    sample_mode=AcquisitionType.FINITE,
    samps_per_chan=N
)

try:
    # ========= 本測定ループ（周波数ごとにN点一括取得して平均） =========
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], marker='o', linestyle='-')
    ax.set_xlabel("freq (Hz)")
    ax.set_ylabel("fluo mean (V)")
    ax.set_title("Per-frequency finite sampling & averaging")
    ax.grid(True)

    for i, f in enumerate(f_list):
        # 1) 周波数設定
        synth[MW_CH].frequency = f

        # 2) 安定化待ち
        if MW_SETTLE_S > 0:
            time.sleep(MW_SETTLE_S)

        # 3) 有限サンプル取得（N点）
        task.start()
        data = task.read(
            number_of_samples_per_channel=N,
            timeout=READ_TIMEOUT
        )
        task.stop()

        # 4) 統計量計算
        data = np.asarray(data, dtype=float)
        fluo_mean[i] = float(np.mean(data))
        fluo_std[i]  = float(np.std(data, ddof=1)) if N >= 2 else 0.0

        # プロット更新（軽量に）
        line.set_data(f_list[:i+1], fluo_mean[:i+1])
        ax.relim(); ax.autoscale_view()
        plt.pause(0.01)

    plt.ioff()
    plt.show()

finally:
    # ========= 終了処理 =========
    try:
        synth[MW_CH].enable = False
    except Exception:
        pass
    try:
        synth.close()
    except Exception:
        pass

    try:
        task.close()
    except Exception:
        pass

# ========= CSV保存 =========
# 周波数, mean[V], std[V]
out_path = os.path.join(out_dir, _file_name + ".csv")
with open(out_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["freq_Hz", "mean_V", "std_V"])
    for f, m, s in zip(f_list, fluo_mean, fluo_std):
        writer.writerow([f, m, s])

print(f"Saved: {out_path}")
