# -*- coding: utf-8 -*-
"""
ODMR 2ピーク（非対称Lorentz）フィッティング + 磁場解析 + 平滑化 + 2Dマップ表示
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ===================== 設定 =====================
filepath = r"C:\Users\Bridgman\Desktop\2025 mikaziri,koga\koga\Data\20251030_1411\odmr_data_averaged.npz"
GAMMA_E_OVER_2PI = 28.02495164e9  # [Hz/T]
SIGMA_SMOOTH = 1.0  # B_mapの平滑化パラメータ
# =================================================


# ---------- データロード ----------
def load_data(filepath):
    data = np.load(filepath)
    freq = data["freq"]
    rawdata = data["rawdata"].transpose(1, 0, 2)
    rawdata = np.fliplr(rawdata)
    return freq, rawdata


# ---------- 非対称2-Lorentzモデル ----------
def lorentzian_2_general(x, x1, x2, a1, a2, hwhm1, hwhm2, offset):
    return (offset
            - a1 * (hwhm1 ** 2 / ((x - x1) ** 2 + hwhm1 ** 2))
            - a2 * (hwhm2 ** 2 / ((x - x2) ** 2 + hwhm2 ** 2)))


# ---------- 全体平均スペクトルから初期ピーク位置を決定 ----------
def estimate_initial_split(freq, rawdata):
    avg_spectrum = np.mean(np.mean(rawdata, axis=0), axis=0)
    inverted = np.max(avg_spectrum) - avg_spectrum
    peaks, _ = find_peaks(inverted, distance=len(freq)//20, prominence=np.ptp(avg_spectrum)*0.05)
    if len(peaks) >= 2:
        sorted_idx = np.argsort(inverted[peaks])[-2:]
        x1, x2 = freq[peaks[sorted_idx[0]]], freq[peaks[sorted_idx[1]]]
        delta_est = abs(x2 - x1)
        x0_est = (x1 + x2) / 2
        return x0_est, delta_est
    else:
        return np.mean(freq), 40e6


# ---------- フィッティング ----------
def fit_odmr(freq, intensity, init_x0, init_delta, prev_delta=None):
    offset_est = np.max(intensity)
    a_est = np.ptp(intensity) / 2
    inverted = offset_est - intensity
    peaks, _ = find_peaks(inverted, distance=len(freq)//20, prominence=np.ptp(intensity)*0.05)

    # --- 2ピーク位置を推定 ---
    if len(peaks) >= 2:
        sorted_idx = np.argsort(inverted[peaks])[-2:]
        x1, x2 = freq[peaks[sorted_idx[0]]], freq[peaks[sorted_idx[1]]]
    elif prev_delta is not None:
        x1 = init_x0 - prev_delta / 2
        x2 = init_x0 + prev_delta / 2
    else:
        x1 = init_x0 - init_delta / 2
        x2 = init_x0 + init_delta / 2

    # --- 初期値・境界設定 ---
    p0 = [x1, x2, a_est, a_est, (freq.max()-freq.min())/200,
          (freq.max()-freq.min())/200, offset_est]
    bounds = (
        [freq.min(), freq.min(), 0, 0, 1e5, 1e5, 0],
        [freq.max(), freq.max(), np.inf, np.inf, 50e6, 50e6, np.inf]
    )

    popt, _ = curve_fit(
        lorentzian_2_general, freq, intensity,
        p0=p0, bounds=bounds, maxfev=20000
    )
    return popt


# ---------- Δfから磁場計算 ----------
def deltaf_to_B(deltaf_hz):
    return deltaf_hz / (2 * GAMMA_E_OVER_2PI)


# ---------- 全画素フィッティング ----------
def create_B_map(freq, rawdata):
    height, width, _ = rawdata.shape
    B_map = np.full((height, width), np.nan)
    print("Fitting all pixels...")

    init_x0, init_delta = estimate_initial_split(freq, rawdata)
    prev_delta = init_delta

    for y in tqdm(range(height)):
        for x in range(width):
            spectrum = rawdata[y, x, :]
            try:
                popt = fit_odmr(freq, spectrum, init_x0, init_delta, prev_delta)
                x1, x2 = popt[0], popt[1]
                delta = abs(x2 - x1)
                B_map[y, x] = deltaf_to_B(delta) * 1e3  # [mT]
                prev_delta = delta  # (A) 前回値を継承して安定化
            except Exception:
                continue

    # (C) ガウシアン平滑化
    B_map_smooth = gaussian_filter(B_map, sigma=SIGMA_SMOOTH)
    return B_map_smooth


# ---------- メイン ----------
freq, rawdata = load_data(filepath)

# 蛍光マップ
fluorescence_map = np.mean(rawdata, axis=2)

# 磁場マップ
B_map = create_B_map(freq, rawdata)
vmin = np.nanpercentile(B_map, 5)
vmax = np.nanpercentile(B_map, 95)

# ---------- 並列表示 ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10))

# --- 蛍光マップ ---
im1 = ax1.imshow(fluorescence_map, cmap="gray", origin="upper")
ax1.set_title("Fluorescence Map")
ax1.set_xlabel("Pixel X")
ax1.set_ylabel("Pixel Y")
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label("Fluorescence Intensity")

# --- 磁場マップ ---
im2 = ax2.imshow(B_map, cmap="jet", origin="upper", vmin=vmin, vmax=vmax)
ax2.set_title("Magnetic Field Map (B_parallel) [mT]")
ax2.set_xlabel("Pixel X")
ax2.set_ylabel("Pixel Y")
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label("B_parallel [mT]")

plt.tight_layout()
plt.show()

# 保存
np.savez("./B_map_result_refined.npz", B_map=B_map, fluorescence=fluorescence_map, freq=freq)
print("結果を保存しました → ./B_map_result_refined.npz")
