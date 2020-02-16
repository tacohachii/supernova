
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import subprocess
import sys
import os
pd.options.mode.chained_assignment = None

# -----------------------------------------------------------------------------------------------------------
#
# 引く数の設定
# 例）
# python3 main.py woosley s12.4
# args[1] = mueller
# args[2] = s12.4
#
# -----------------------------------------------------------------------------------------------------------
args = sys.argv
if len(args) <= 2:
    print('Error: Set 2 argument')
    sys.exit()
progenitor_type = args[1]
if progenitor_type != 'mueller' and progenitor_type != 'woosley':
    print('Error: Set progenitor_type')
    sys.exit()
filename = args[2]
input_file = f"./data/{progenitor_type}/{filename}"
if not os.path.exists(input_file):
    print(f"Error: Don't open {input_file}")
    sys.exit()
output_folder = f"./output/{progenitor_type}_{filename}/"
subprocess.call(["mkdir", "-p", output_folder])
output_prefix = f"{output_folder}{progenitor_type}_{filename}_"

# -----------------------------------------------------------------------------------------------------------
#
# 単位変換の設定
#
# -----------------------------------------------------------------------------------------------------------
_kg = 1e3  # kg -> g
_m = 1e2  # m -> cm
_km = 1e5  # km -> cm
_MeV = 1.60218e-6   # MeV -> erg
_J = 1e7  # J -> erg
_ms = 1e-3  # ms -> s


# -----------------------------------------------------------------------------------------------------------
#
# 定数の定義(CGS単位系)
#
# -----------------------------------------------------------------------------------------------------------
PI = 3.14159265359
G = 6.67259e-11*_m**3/_kg  # cm^3/(g.s^2)
r_0 = 12*_km  # cm
r_1 = 120*_km  # cm
M_sun = 1.989e+30*_kg  # g
tau_1p5 = 1.2  # s
zeta = 0.8
a = 0.084
c = 2.99792458e8*_m  # cm/s
alpha_turb = 1.18
mu = 1.66053906660e-27*_kg  # g
e_diss = 8.8*_MeV/mu  # erg/g
eta_out = 1
e_rec = 5*_MeV/mu  # erg/g
beta_expl = 4
radiation_constant = 7.566e-16*_J/_m**3  # erg/cm^3/K^4
T_o_16 = 2.5e9  # K
T_si_28 = 3.5e9  # K
T_ni_56 = 7e9  # K
alpha_out = 0.5
X_i_common_arr = [  # 原子の情報(delta(erg))
    {'name': 'neutrons', 'delta': 8.071*_MeV,    'atom': 1.0},
    {'name': 'H1',       'delta': 7.289*_MeV,    'atom': 1.0},
    {'name': 'He3',      'delta': 14.931*_MeV,   'atom': 3.0},
    {'name': 'He4',      'delta': 2.425*_MeV,    'atom': 4.0},
    {'name': 'C12',      'delta': 0.0*_MeV,      'atom': 12.0},
    {'name': 'N14',      'delta': 2.863*_MeV,    'atom': 14.0},
    {'name': 'O16',      'delta': -4.737*_MeV,   'atom': 16.0},
    {'name': 'Ne20',     'delta': -7.042*_MeV,   'atom': 20.0},
    {'name': 'Mg24',     'delta': -13.933*_MeV,  'atom': 24.0},
    {'name': 'Si28',     'delta': -21.492*_MeV,  'atom': 28.0},
    {'name': 'S32',      'delta': -26.016*_MeV,  'atom': 32.0},
    {'name': 'Ar36',     'delta': -30.230*_MeV,  'atom': 36.0},
    {'name': 'Ca40',     'delta': -34.846*_MeV,  'atom': 40.0},
    {'name': 'Ti44',     'delta': -37.548*_MeV,  'atom': 44.0},
    {'name': 'Cr48',     'delta': -42.819*_MeV,  'atom': 48.0},
    {'name': 'Fe52',     'delta': -48.331*_MeV,  'atom': 52.0},
    {'name': 'Fe54',     'delta': -56.249*_MeV,  'atom': 54.0},
    {'name': 'Ni56',     'delta': -53.904*_MeV,  'atom': 56.0},
    {'name': 'Fe56',     'delta': -60.601*_MeV,  'atom': 56.0}
]
if progenitor_type == 'mueller':
    X_i_common_arr.remove(
        {'name': 'He3', 'delta': 14.931*_MeV, 'atom': 3.0}
    )

# -----------------------------------------------------------------------------------------------------------
#
# データの読み込み
#
# -----------------------------------------------------------------------------------------------------------
header_name = [
    "grid",
    "cell_mass",  # woosley で 入れない
    "cell_outer_total_mass",
    "cell_outer_radius",
    "cell_outer_velocity",
    "cell_density",
    "cell_temperature",
    "cell_pressure",
    "cell_specific_energy",
    "cell_specific_entropy",
    "cell_angular_velocity",
    "cell_A_bar",
    "cell_Y_e",
    "stability",  # mueller で 入れない
    "NETWORK",  # mueller で 入れない
    "neutrons",
    "H1",
    "He3",  # mueller で 入れない
    "He4",
    "C12",
    "N14",
    "O16",
    "Ne20",
    "Mg24",
    "Si28",
    "S32",
    "Ar36",
    "Ca40",
    "Ti44",
    "Cr48",
    "Fe52",
    "Fe54",
    "Ni56",
    "Fe56",
    "'Fe'"
]

if progenitor_type == 'mueller':
    header_name.remove("stability")
    header_name.remove("NETWORK")
    header_name.remove("He3")
    data = pd.read_csv(
        input_file,
        skiprows=0,
        names=header_name,
        sep='\s{1,}',
        engine='python',
        na_values='-',
        skipfooter=1
    )
if progenitor_type == 'woosley':
    header_name.remove("cell_mass")
    data = pd.read_csv(
        input_file,
        skiprows=2,
        names=header_name,
        sep='\s{2,}',
        engine='python',
        na_values='---'
    )

data['average_densty'] = float(np.nan)
data['infall_time'] = float(np.nan)
data['mass_accretion_rate'] = float(np.nan)
data['gain_radius'] = float(np.nan)
data['pns_radius'] = float(np.nan)
data['cool_time_scale'] = float(np.nan)
data['bindin_energy'] = float(np.nan)
data['luminosity_of_accretion_component'] = float(np.nan)
data['luminosity_of_diffusive_component'] = float(np.nan)
data['neutrino_luminosity'] = float(np.nan)
data['redsift_factor'] = float(np.nan)
data['shock_radius'] = float(np.nan)
data['radius_max'] = float(np.nan)
data['radius_min'] = float(np.nan)
data['post_shock_binding_energy'] = float(np.nan)
data['advection_time_scale'] = float(np.nan)
data['heating_time_scale'] = float(np.nan)
data['critical_time_scale_ratio'] = float(np.nan)
data['effeciency_parameter'] = float(np.nan)
data['avarage_shock_velocity'] = float(np.nan)
data['post_shock_velocity'] = float(np.nan)
data['escape_velocity'] = float(np.nan)
data['post_shock_temperature'] = float(np.nan)
data['O_16_temperature'] = T_o_16
data['Si_28_temperature'] = T_si_28
data['Ni_56_temperature'] = T_ni_56
data['unshocked_material_binding_energy'] = float(np.nan)
data['nuclear_burning_energy'] = float(np.nan)
data['E_diag_at_a_given_mass_shell'] = float(np.nan)
data['diagnostic_explosion_energy'] = float(np.nan)
data['mass_accretion_rate_against_mass_out'] = float(np.nan)
data['outflow_rate'] = float(np.nan)
data['mass_of_baryonic_neutron_star'] = float(np.nan)
data['mass_of_proto_neutron_star'] = float(np.nan)
data['final_explosion_energy'] = float(np.nan)
data['which_phase'] = float(np.nan)


# -----------------------------------------------------------------------------------------------------------
#
# 変数の定義
#
# -----------------------------------------------------------------------------------------------------------
max_row = len(data.index)  # 列数
M = data['cell_outer_total_mass']  # 質量M(g)
r = data['cell_outer_radius']  # 半径r(cm)
rho = data['cell_density']  # 密度rho(g/cm^3)
rho_bar = data['average_densty']  # 平均密度ρ_bar(g/cm^3)
t = data['infall_time']  # 落下時間t(s)
M_dot = data['mass_accretion_rate']  # 質量降着率M_dot(g/s)
r_g = data['gain_radius']  # ゲイン半径r_g(cm)
r_PNS = data['pns_radius']  # 原始中性子星の半径r_PNS(cm)
tau_cool = data['cool_time_scale']  # 中性子星結合エネルギーが放射されるタイムスケールtau_cool(s)
E_bind = data['bindin_energy']  # 冷たい中性子星の結合エネルギーE_bind(erg)
L_acc = data['luminosity_of_accretion_component']  # 降着成分による光度L_acc(erg/s)
L_diff = data['luminosity_of_diffusive_component']  # 拡散による光度L_diff(erg/s)
L_nu = data['neutrino_luminosity']  # ニュートリノ光度L_nu(erg/s)
alpha = data['redsift_factor']  # 赤方偏移因子alpha
r_sh = data['shock_radius']  # 衝撃波の半径r_sh(cm)
r_max = data['radius_max']  # r_shとr_gの大きい方r_max(cm)
r_min = data['radius_min']  # r_shとr_gの小さい方r_min(cm)
e_g = data['post_shock_binding_energy']  # ショック後の結合エネルギーe_g(erg/g)
tau_adv = data['advection_time_scale']  # 降着物質がゲイン領域を移動するタイムスケールtau_adv(s)
tau_heat = data['heating_time_scale']  # 降着物質がゲイン領域を移動するタイムスケールtau_heat(s)
tau_ration = data['critical_time_scale_ratio']  # pre->exの基準tau_ration
eta_acc = data['effeciency_parameter']  # 降着のパラメーター?eta_acc(erg/g)
v_sh = data['avarage_shock_velocity']  # 衝撃波の速さv_sh(cm/s)
v_post = data['post_shock_velocity']  # 衝撃波通過後の速さv_post(cm/s)
v_esc = data['escape_velocity']  # 脱出速さv_esc(cm/s)
T_sh = data['post_shock_temperature']  # 衝撃波直後の温度T_sh(K)
e_bind = data['unshocked_material_binding_energy']  # 衝撃波前の結合エネルギーe_bind(erg/g)
e_burn = data['nuclear_burning_energy']  # 核融合によるエネルギーe_burn(erg/g)
E_imm = data['E_diag_at_a_given_mass_shell']  # 衝撃波が到達した時の診断エネルギーE_imm(erg)
E_diag = data['diagnostic_explosion_energy']  # 診断爆発エネルギーE_diag(erg)
M_dot_acc = data['mass_accretion_rate_against_mass_out']  # 単位時間降着質量M_dot_acc
M_dot_out = data['outflow_rate']  # 単位時間あたりに出ていく質量M_dot_out
M_by = data['mass_of_baryonic_neutron_star']  # 原始中性子星M_by
M_ns = data['mass_of_proto_neutron_star']  # 最終的な中性子星の質量M_ns
E_expl = data['final_explosion_energy']  # 爆発のエネルギーE_expl
phase = data['which_phase']
X_i_data = []  # 原子の質量分立の配列  例) X_i_data[1] = data['H1']
# 原子の名前の配列  例) X_i_name[1] = 'H1'
X_i_name = [d.get('name') for d in X_i_common_arr]
# 原子の静止質量の配列  例) X_i_delta[1] = 7.289*_MeV
X_i_delta = [d.get('delta') for d in X_i_common_arr]
# 原子の原子質量の配列  例) X_i_atom[1] = 1.0
X_i_atom = [d.get('atom') for d in X_i_common_arr]


# -----------------------------------------------------------------------------------------------------------
#
# 関数の定義
#
# -----------------------------------------------------------------------------------------------------------


def flashing_method(i):
    # [flashing method] 衝撃波直後の温度によってどこまで核融合が進むのか決まる
    if T_sh[i] < T_o_16:
        e_burn[i] = 0
    if T_o_16 <= T_sh[i] and T_sh[i] < T_si_28:
        # このときO_16より軽いものはO_16まで核融合する
        i_o_16 = X_i_name.index('O16')
        for j in range(i_o_16):
            if X_i_delta[j] / X_i_atom[j] < X_i_delta[i_o_16] / X_i_atom[i_o_16]:
                continue
            e_burn[i] = e_burn[i-1] + X_i_data[j][i] * 1e-3 * (X_i_delta[j] / X_i_atom[j] -
                                                               X_i_delta[i_o_16] / X_i_atom[i_o_16]) / mu
    elif T_si_28 <= T_sh[i] and T_sh[i] < T_ni_56:
        # このときSi_28より軽いものはSi_28まで核融合する
        i_si28 = X_i_name.index('Si28')
        for j in range(i_si28):
            if X_i_delta[j] / X_i_atom[j] < X_i_delta[i_si28] / X_i_atom[i_si28]:
                continue
            e_burn[i] = e_burn[i-1] + X_i_data[j][i] * 1e-3 * (X_i_delta[j] / X_i_atom[j] -
                                                               X_i_delta[i_si28] / X_i_atom[i_si28]) / mu
    elif T_ni_56 <= T_sh[i]:
        # ニュートン法でT_alphaを求める
        T_n = T_ni_56 / 10 ** 9
        while True:
            f_T_n = 11.62 + 1.5 * \
                np.log10(T_n) - 39.17 / T_n - np.log10(rho[i])
            f_T_n_dif = 1.5 / (T_n * np.log(10.0)) + 39.17 / T_n ** 2
            T_n_1 = T_n - f_T_n / f_T_n_dif
            if abs(T_n_1 - T_n) < 1.0e-4:
                break
            T_n = T_n_1
        T_alpha = T_n * 10 ** 9
        # T_alphaより小さい時は全てNi_56まで核融合する
        if T_sh[i] < T_alpha:
            i_ni56 = X_i_name.index('Si28')
            for j in range(i_ni56):
                if X_i_delta[j] / X_i_atom[j] < X_i_delta[i_ni56] / X_i_atom[i_ni56]:
                    continue
                e_burn[i] = e_burn[i-1] + X_i_data[j][i] * 1e-3 * \
                    (X_i_delta[j] / X_i_atom[j] -
                     X_i_delta[i_ni56] / X_i_atom[i_ni56]) / mu
        else:
            e_burn[i] = 0


# -----------------------------------------------------------------------------------------------------------
#
# tau_ratioまで計算
#
# -----------------------------------------------------------------------------------------------------------
# 平均密度rho_bar(g/cm^3)
rho_bar = M / (4 / 3 * PI * r ** 3)
# 落下時間t(s) -- 式(2)
t = np.sqrt(PI/(4 * G * rho_bar))
# 質量降着率M_dot(g/s) -- 式(3)
M_dot = 2 * M / t * rho / (rho_bar - rho)
# ゲイン半径r_g(cm) -- 式(9)
r_g = (r_1 ** 3 * (M_dot / M_sun) * (M / M_sun) ** (-3) + r_0 ** 3) ** (1 / 3)
# 原始中性子星の半径r_PNS(cm)
r_PNS = 5 / 7 * r_g
# 中性子星結合エネルギーが放射されるタイムスケールtau_cool(s) -- 式(12)
for i in range(max_row):
    tmp_tau = tau_1p5 * (M[i] / (1.5 * M_sun)) ** (5 / 3)
    if tmp_tau < 0.1:
        tau_cool[i] = 0.1
    else:
        tau_cool[i] = tmp_tau
# 冷たい中性子星の結合エネルギーE_bind(erg) -- 式(15')
E_bind = a * (M / M_sun) ** 2 * M_sun * c ** 2
# 降着成分による光度L_acc(erg/s) -- 式(10)
L_acc = zeta * G * M * M_dot / r_g
# 拡散による光度L_diff(erg/s) -- 式(14')
L_diff = 0.3 * E_bind / tau_cool * np.exp(-1 * t / tau_cool)
# ニュートリノ光度L_nu(erg/s) -- 式(16')
L_nu = L_acc + L_diff
# 赤方偏移因子alpha
for i in range(max_row):
    tmp_root = 1 - 2 * G * M[i] / (r_PNS[i] * c ** 2)
    if tmp_root < 0:
        alpha[i] = 1.0
    else:
        alpha[i] = np.sqrt(tmp_root)
# 衝撃波の半径r_sh(cm) -- 式(22')
r_sh = alpha_turb * 0.55*_km * (L_nu / 1e52) ** (4 / 9) * (alpha ** 3) ** (4 / 9) * (
    M / M_sun) ** (5 / 9) * (r_g / (10*_km)) ** (16 / 9) * (M_dot / M_sun) ** (-2 / 3)
# r_shとr_gの大きい方r_max(cm)
for i in range(max_row):
    r_max[i] = max(r_sh[i], r_g[i])
# r_shとr_gの小さい方r_min(cm)
for i in range(max_row):
    r_min[i] = min(r_sh[i], r_g[i])
# ショック後の結合エネルギーe_g(erg/g) -- 式(29')
e_g = 3 / 4 * e_diss + G * M / (4 * r_max)
# 降着物質がゲイン領域を移動するタイムスケールtau_adv(s) -- 式(23)
tau_adv = 18e-3 * \
    (r_sh / (100*_km)) ** (3 / 2) * (M / M_sun) ** (-1 / 2) * \
    np.log(r_sh / r_g)  # TODO: どっち使う？ np.log(r_max / r_min)
# 降着物質がゲイン領域を移動するタイムスケールtau_heat(s) -- 式(30')
tau_heat = 150*_ms * (e_g / 1e19) * (r_g / (100*_km)) ** 2 * \
    (L_nu / 1e52) ** (-1) * (alpha ** 3) ** (-1) * (M / M_sun) ** (-2)
# pre->exの基準tau_ration
tau_ration = tau_adv / tau_heat
# 降着のパラメーター?eta_acc(erg/g) -- 式(31)
eta_acc = tau_adv / tau_heat * e_g
# 原子のデータを配列にセット 例) X_i_data[1] = data['H1']
for i in range(len(X_i_name)):
    X_i_data.append(data[X_i_name[i]])


# -----------------------------------------------------------------------------------------------------------
#
# phase毎に計算
#
# -----------------------------------------------------------------------------------------------------------
condition_1 = False
condition_2 = False
for i in range(max_row):
    v_esc[i] = np.sqrt(2 * G * M[i] / r[i])
    if i == 0 or (tau_ration[i] < 1 and condition_1 == False):
        phase[i] = 0  # pre_phase
        # 爆発前段階
        # tau_ration[i] > 1の条件を満たさない
        E_imm[i] = 0
        E_diag[i] = 0
        v_sh[i] = 1e-10  # 0だと割り算できない
        v_post[i] = 0
        T_sh[i] = 0
        e_bind[i] = 0
        e_burn[i] = 0
        M_dot_acc[i] = M_dot[i]
        M_dot_out[i] = eta_out * eta_acc[i] * M_dot_acc[i] / e_g[i]
        M_by[i] = 0
        M_ns[i] = 0
        M_ini = M[i]
    else:
        e_bind[i] = -1 * G * M[i] / r[i]
        if v_post[i-1] < v_esc[i-1] and condition_2 == False:
            phase[i] = 1  # ex_phase_1
            condition_1 = True
            # 爆発段階I
            # tau_ration[i] > 1の条件を満たす(1度でも)
            # v_post[i-1] > v_esc[i-1]の条件を満たさない
            dm = M[i] - M[i-1]
            tmp_val = M_dot[i] / (4 * PI * r[i] ** 2 * v_sh[i-1] * rho[i])
            if 1 < tmp_val:
                tmp_min = 1  # (v_sh 小)
            else:
                tmp_min = tmp_val  # (v_sh 大)
            #
            # E_imm/E_diag(仮) -> v_sh -> v_post
            #
            E_imm[i] = E_imm[i-1] + e_rec * eta_acc[i] / e_g[i] * tmp_min * dm
            E_diag[i] = E_diag[i-1] + (1 - alpha_out) * \
                e_rec * eta_acc[i] / e_g[i] * dm
            tmp_v_sh = 0.794 * np.sqrt(E_imm[i] / (M[i] - M_ini)) * \
                ((M[i] - M_ini) / (rho[i] * r[i] ** 3)) ** 0.19
            #
            # T_sh -> Xi -> e_burn -> E_imm/E_diag
            #
            T_sh[i] = (3 * (beta_expl - 1) / (radiation_constant *
                                              beta_expl) * rho[i] * tmp_v_sh ** 2) ** (1 / 4)
            flashing_method(i)
            E_imm[i] += alpha_out * (e_bind[i] + e_burn[i]) * dm
            E_diag[i] += alpha_out * (e_bind[i] + e_burn[i]) * dm
            v_sh[i] = 0.794 * np.sqrt(E_imm[i] / (M[i] - M_ini)) * \
                ((M[i] - M_ini) / (rho[i] * r[i] ** 3)) ** 0.19
            v_post[i] = (beta_expl - 1) / beta_expl * v_sh[i]
            #
            # E_diagが負の時はブラックホールになる
            #
            if E_diag[i] < 0 or phase[i-1] == -1:
                # ブラックホール
                # E_diagが一度でも負になったり
                # ~~~~~~~~したらここに来る
                phase[i] = -1  # BH
                #
                # v_sh, v_post, E_diag
                #
                v_sh[i] = 1e-10
                v_post[i] = 0
                E_diag[i] = 0
                #
                # M_dot_acc, M_dot_out
                #
                M_dot_out[i] = 0
                M_dot_acc[i] = 0
                #
                # M_by, M_ns
                #
                M_by[i] = M_by[i-1]
                M_ns[i] = M_ns[i-1]
            else:
                #
                # M_dot_acc -> M_dot_out
                #
                tmp_val = M_dot[i] / (4 * PI * r[i] ** 2 * v_sh[i] * rho[i])
                if 1 < tmp_val:
                    M_dot_acc[i] = M_dot[i] * M_dot[i] / \
                        (4 * PI * r[i] ** 2 * v_sh[i] * rho[i])
                else:
                    M_dot_acc[i] = M_dot[i] * 1
                M_dot_out[i] = eta_out * eta_acc[i] * \
                    M_dot_acc[i] / e_g[i]
                #
                # M_by -> M_ns
                #
                if 1 - tau_adv[i] / tau_heat[i] > 0:
                    M_by[i] = M_ini + (1 - alpha_out) * \
                        (1 - tau_adv[i] / tau_heat[i]) * dm
                else:
                    M_by[i] = M_ini
                M_ns[i] = 0
        else:
            # 爆発段階II
            # tau_ration[i] > 1の条件を満たす(1度でも)
            # v_post[i-1] > v_esc[i-1]の条件を満たす(1度でも)
            phase[i] = 2  # ex_phase_2
            condition_2 = True
            dm = M[i] - M[i-1]
            tmp_val = M_dot[i] / (4 * PI * r[i] ** 2 * v_sh[i-1] * rho[i])
            if 1 < tmp_val:
                tmp_min = 1  # (v_sh 小)
            else:
                tmp_min = tmp_val  # (v_sh 大)
            #
            # E_imm/E_diag(仮) -> v_sh -> v_post
            #
            E_imm[i] = E_imm[i-1] + e_rec * eta_acc[i] / e_g[i] * tmp_min * dm
            E_diag[i] = E_diag[i-1]
            tmp_v_sh = 0.794 * np.sqrt(E_imm[i] / (M[i] - M_ini)) * \
                ((M[i] - M_ini) / (rho[i] * r[i] ** 3)) ** 0.19
            #
            # T_sh -> Xi -> e_burn -> E_imm/E_diag
            #
            T_sh[i] = (3 * (beta_expl - 1) / (radiation_constant *
                                              beta_expl) * rho[i] * tmp_v_sh ** 2) ** (1 / 4)
            flashing_method(i)
            E_imm[i] += alpha_out * (e_bind[i] + e_burn[i]) * dm
            # E_diag[i] += (e_bind[i] + e_burn[i]) * dm
            E_diag[i] += alpha_out * \
                (e_bind[i] + e_burn[i]) * dm  # TODO: こっちの方が合う
            v_sh[i] = 0.794 * np.sqrt(E_imm[i] / (M[i] - M_ini)) * \
                ((M[i] - M_ini) / (rho[i] * r[i] ** 3)) ** 0.19
            v_post[i] = (beta_expl - 1) / beta_expl * v_sh[i]
            #
            # E_diagが負の時はブラックホールになる
            #
            if phase[i-1] == -1 or E_diag[i] < 0:
                # ブラックホール
                # E_diagが一度でも負になったり
                # ~~~~~~~~したらここに来る
                phase[i] = -1  # BH
                v_sh[i] = 1e-10
                v_post[i] = 0
                E_imm[i] = E_imm[i-1]
                E_diag[i] = 0
                M_dot_out[i] = 0
                M_dot_acc[i] = 0
                M_by[i] = M_by[i-1]
                M_ns[i] = M_ns[i-1]
            else:
                #
                # M_dot_acc, M_dot_out
                #
                M_dot_out[i] = 0
                M_dot_acc[i] = 0
                #
                # M_by -> M_ns
                #
                M_by[i] = M_by[i-1]
                if E_diag[i] > 0:
                    M_ns[i] = (-1 + np.sqrt(1 + 4 * M_by[i] / M_sun * 0.084)
                               ) / 2 / 0.084 * M_sun
                else:
                    M_ns[i] = 0
E_expl = E_diag  # 爆発のエネルギーE_expl -- 式(50)

if phase[max_row-1] == 0:
    phase[max_row-1] = -1
    E_expl[max_row-1] = 0


# -----------------------------------------------------------------------------------------------------------
#
# 単位変換(出力用)
#
# -----------------------------------------------------------------------------------------------------------
for i in range(max_row):
    M[i] = M[i]/M_sun
    r_g[i] = r_g[i]/_km
    r_sh[i] = r_sh[i]/_km
    v_post[i] = v_post[i]/_km
    v_esc[i] = v_esc[i]/_km


# -----------------------------------------------------------------------------------------------------------
#
# データ格納
#
# -----------------------------------------------------------------------------------------------------------
data['average_densty'] = rho_bar
data['infall_time'] = t
data['mass_accretion_rate'] = M_dot
data['gain_radius'] = r_g
data['pns_radius'] = r_PNS
data['cool_time_scale'] = tau_cool
data['bindin_energy'] = E_bind
data['luminosity_of_accretion_component'] = L_acc
data['luminosity_of_diffusive_component'] = L_diff
data['neutrino_luminosity'] = L_nu
data['redsift_factor'] = alpha
data['shock_radius'] = r_sh
data['radius_max'] = r_max
data['radius_min'] = r_min
data['post_shock_binding_energy'] = e_g
data['advection_time_scale'] = tau_adv
data['heating_time_scale'] = tau_heat
data['critical_time_scale_ratio'] = tau_ration
data['effeciency_parameter'] = eta_acc
data['avarage_shock_velocity'] = v_sh
data['post_shock_velocity'] = v_post
data['escape_velocity'] = v_esc
data['post_shock_temperature'] = T_sh
data['unshocked_material_binding_energy'] = e_bind
data['nuclear_burning_energy'] = e_burn
data['E_diag_at_a_given_mass_shell'] = E_imm
data['diagnostic_explosion_energy'] = E_diag
data['mass_accretion_rate_against_mass_out'] = M_dot_acc
data['outflow_rate'] = M_dot_out
data['mass_of_baryonic_neutron_star'] = M_by
data['mass_of_proto_neutron_star'] = M_ns
data['final_explosion_energy'] = E_expl


# -----------------------------------------------------------------------------------------------------------
#
# CSV出力
#
# -----------------------------------------------------------------------------------------------------------
output_csv = f"{output_prefix}result.csv"
data.to_csv(output_csv)
# subprocess.call(["open", output_csv])


# -----------------------------------------------------------------------------------------------------------
#
# プロット画像出力
#
# -----------------------------------------------------------------------------------------------------------

# Phase
_phase_t = data.plot(
    title='t - Phase',
    xlim=[0, 5],
    grid=True,
    color='Black',
    yticks=[-1, 0, 1, 2],
    x='infall_time',
    y='which_phase'
)
plt.savefig(f'{output_prefix}phase_t.png')

# 半径
_r_t = data.plot(
    title='t - r',
    xlim=[0, 5],
    ylim=[0, 300],
    grid=True,
    x='infall_time',
    y='gain_radius'
)
data.plot(
    ax=_r_t,
    xlim=[0, 5],
    ylim=[0, 300],
    grid=True,
    x='infall_time',
    y='shock_radius'
)
data.plot(
    ax=_r_t,
    xlim=[0, 5],
    secondary_y=True,
    color='Black',
    x='infall_time',
    y='which_phase'
)
plt.savefig(f'{output_prefix}r_t.png')

# タイムスケール
_tau_t = data.plot(
    title='t - tau',
    xlim=[0, 5],
    ylim=[0, 1e-2],
    grid=True,
    x='infall_time',
    y='advection_time_scale'
)
data.plot(
    ax=_tau_t,
    xlim=[0, 5],
    ylim=[0, 1e-2],
    grid=True,
    x='infall_time',
    y='heating_time_scale'
)
data.plot(
    ax=_tau_t,
    xlim=[0, 5],
    secondary_y=True,
    color='Black',
    x='infall_time',
    y='which_phase'
)
plt.savefig(f'{output_prefix}tau_t.png')

# 速さ
_v_t = data.plot(
    title='t - v',
    xlim=[0, 5],
    ylim=[0, 3e4],
    grid=True,
    x='infall_time',
    y='post_shock_velocity'
)
data.plot(
    ax=_v_t,
    xlim=[0, 5],
    ylim=[0, 3e4],
    grid=True,
    x='infall_time',
    y='escape_velocity'
)
data.plot(
    ax=_v_t,
    xlim=[0, 5],
    secondary_y=True,
    color='Black',
    x='infall_time',
    y='which_phase'
)
plt.savefig(f'{output_prefix}v_t.png')

# 温度
_T_t = data.plot(
    title='t - T_sh',
    xlim=[0, 5],
    grid=True,
    x='infall_time',
    y='post_shock_temperature'
)
data.plot(
    ax=_T_t,
    xlim=[0, 5],
    grid=True,
    x='infall_time',
    y='O_16_temperature'
)
data.plot(
    ax=_T_t,
    xlim=[0, 5],
    grid=True,
    x='infall_time',
    y='Si_28_temperature'
)
data.plot(
    ax=_T_t,
    xlim=[0, 5],
    grid=True,
    x='infall_time',
    y='Ni_56_temperature'
)
data.plot(
    ax=_T_t,
    xlim=[0, 5],
    secondary_y=True,
    color='Black',
    x='infall_time',
    y='which_phase'
)
plt.savefig(f'{output_prefix}T_t.png')

# エネルギー
_E_t = data.plot(
    title='t - E',
    xlim=[0, 5],
    ylim=[0, 6e50],
    grid=True,
    x='infall_time',
    y='E_diag_at_a_given_mass_shell'
)
data.plot(
    ax=_E_t,
    xlim=[0, 5],
    ylim=[0, 6e50],
    grid=True,
    x='infall_time',
    y='diagnostic_explosion_energy'
)
data.plot(
    ax=_E_t,
    xlim=[0, 5],
    secondary_y=True,
    color='Black',
    x='infall_time',
    y='which_phase'
)
plt.savefig(f'{output_prefix}E_t.png')

# 質量分立
_Xi_M = data.plot(
    title='M - Xi',
    ylim=[1e-4, 1],
    grid=True,
    x='cell_outer_total_mass',
    y='neutrons'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell_outer_total_mass',
    y='H1'
)
if progenitor_type == 'woosley':
    data.plot(
        ax=_Xi_M,
        ylim=[1e-4, 1],
        grid=True,
        x='cell_outer_total_mass',
        y='He3'
    )
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell_outer_total_mass',
    y='He4'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell_outer_total_mass',
    y='C12'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell_outer_total_mass',
    y='N14'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell_outer_total_mass',
    y='O16'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell_outer_total_mass',
    y='Ne20'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell_outer_total_mass',
    y='Mg24'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell_outer_total_mass',
    y='Si28'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell_outer_total_mass',
    y='S32'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell_outer_total_mass',
    y='Ar36'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell_outer_total_mass',
    y='Ca40'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell_outer_total_mass',
    y='Ti44'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell_outer_total_mass',
    y='Cr48'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell_outer_total_mass',
    y='Fe52'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell_outer_total_mass',
    y='Fe54'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell_outer_total_mass',
    y='Ni56'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell_outer_total_mass',
    y='Fe56'
)
plt.savefig(f'{output_prefix}Xi_M.png')


# -----------------------------------------------------------------------------------------------------------
#
# pdf出力
#
# -----------------------------------------------------------------------------------------------------------
pdf = PdfPages(f'{output_prefix}plots.pdf')
fignums = plt.get_fignums()
for fignum in fignums:
    plt.figure(fignum)
    pdf.savefig()
pdf.close()
plt.close('all')


# -----------------------------------------------------------------------------------------------------------
#
# 比較データの出力
#
# -----------------------------------------------------------------------------------------------------------
if progenitor_type == 'mueller':
    subprocess.call(["mkdir", "-p", './data/compare/'])
    output_dat = f"./data/compare/{filename}_calc.dat"
    data[
        [
            'infall_time',
            'cell_outer_total_mass',
            'mass_accretion_rate',
            'gain_radius',
            'shock_radius',
            'neutrino_luminosity',
            'advection_time_scale',
            'heating_time_scale',
            'outflow_rate',
            'diagnostic_explosion_energy',
            'E_diag_at_a_given_mass_shell',
            'luminosity_of_diffusive_component',
            'cool_time_scale',
            'bindin_energy',
            'post_shock_binding_energy'
        ]
    ].to_csv(output_dat, sep=' ', index=False)


# -----------------------------------------------------------------------------------------------------------
#
# 比較のデータの読み込み
#
# -----------------------------------------------------------------------------------------------------------
if progenitor_type == 'mueller':
    test_file = f'./data/compare/{filename}_test.dat'
    if not os.path.exists(test_file):
        print(f"Error: Don't open {test_file}")
        sys.exit()

    test_header = [
        'test_infall_time',
        'test_cell_outer_total_mass',
        'test_mass_accretion_rate',
        'test_gain_radius',
        'test_shock_radius',
        'test_neutrino_luminosity',
        'test_rho0 [g/cm^3] in prog.',  # 除く
        'test_r0 [cm] in progenitor',  # 除く
        'test_advection_time_scale',
        'test_heating_time_scale',
        'test_outflow_rate',
        'test_diagnostic_explosion_energy',
        'test_E_diag_at_a_given_mass_shell',
        'test_luminosity_of_diffusive_component',
        'test_cool_time_scale',
        'test_bindin_energy',
        'test_post_shock_binding_energy'
    ]
    test = pd.read_csv(
        test_file,
        skiprows=1,
        names=test_header,
        sep=' '
    )
    test_max_row = len(test.index)  # 列数

    # 単位変換
    for i in range(test_max_row):
        test['test_cell_outer_total_mass'][i] /= M_sun
        test['test_gain_radius'][i] /= _km
        test['test_shock_radius'][i] /= _km

    calc_header = [
        'infall_time',
        'cell_outer_total_mass',
        'mass_accretion_rate',
        'gain_radius',
        'shock_radius',
        'neutrino_luminosity',
        'advection_time_scale',
        'heating_time_scale',
        'outflow_rate',
        'diagnostic_explosion_energy',
        'E_diag_at_a_given_mass_shell',
        'luminosity_of_diffusive_component',
        'cool_time_scale',
        'bindin_energy',
        'post_shock_binding_energy'
    ]
    calc_file = output_dat
    calc = pd.read_csv(
        calc_file,
        skiprows=1,
        names=calc_header,
        sep=' '
    )


# -----------------------------------------------------------------------------------------------------------
#
# 比較のデータのプロット
#
# -----------------------------------------------------------------------------------------------------------
if progenitor_type == 'mueller':
    output_compare = f'./output/compare_{filename}/'
    subprocess.call(["mkdir", "-p", output_compare])

    # 質量
    c_M_t = test.plot(
        title='t - M',
        xlim=[0, 5],
        ylim=[0, 2],
        grid=True,
        alpha=0.5,
        x='test_infall_time',
        y='test_cell_outer_total_mass'
    )
    calc.plot(
        ax=c_M_t,
        xlim=[0, 5],
        ylim=[0, 2],
        grid=True,
        alpha=0.5,
        x='infall_time',
        y='cell_outer_total_mass'
    )
    plt.savefig(f'{output_compare}{filename}_M_t.png')

    # dot
    c_M_dot_t = test.plot(
        title='t - M_dot',
        xlim=[0, 5],
        ylim=[0, 1e33],
        grid=True,
        alpha=0.5,
        x='test_infall_time',
        y='test_mass_accretion_rate'
    )
    calc.plot(
        ax=c_M_dot_t,
        xlim=[0, 5],
        ylim=[0, 1e33],
        grid=True,
        alpha=0.5,
        x='infall_time',
        y='mass_accretion_rate'
    )
    plt.savefig(f'{output_compare}{filename}_M_dot_t.png')

    # 半径
    c_r_g_t = test.plot(
        title='t - r_g',
        xlim=[0, 5],
        ylim=[0, 300],
        grid=True,
        alpha=0.5,
        x='test_infall_time',
        y='test_gain_radius'
    )
    calc.plot(
        ax=c_r_g_t,
        xlim=[0, 5],
        ylim=[0, 300],
        grid=True,
        alpha=0.5,
        x='infall_time',
        y='gain_radius'
    )
    plt.savefig(f'{output_compare}{filename}_r_g_t.png')

    # タイムスケール
    c_tau_cool_t = test.plot(
        title='t - tau_cool',
        xlim=[0, 5],
        ylim=[0, 2],
        grid=True,
        alpha=0.5,
        x='test_infall_time',
        y='test_cool_time_scale'
    )
    calc.plot(
        ax=c_tau_cool_t,
        xlim=[0, 5],
        ylim=[0, 2],
        grid=True,
        alpha=0.5,
        x='infall_time',
        y='cool_time_scale'
    )
    plt.savefig(f'{output_compare}{filename}_tau_cool_t.png')

    # エネルギー
    c_E_bind_t = test.plot(
        title='t - E_bind',
        xlim=[0, 5],
        ylim=[0, 1.0e54],
        grid=True,
        alpha=0.5,
        x='test_infall_time',
        y='test_bindin_energy'
    )
    calc.plot(
        ax=c_E_bind_t,
        xlim=[0, 5],
        ylim=[0, 1.0e54],
        grid=True,
        alpha=0.5,
        x='infall_time',
        y='bindin_energy'
    )
    plt.savefig(f'{output_compare}{filename}_E_bind_t.png')

    # 光度
    c_L_diff_t = test.plot(
        title='t - L_diff',
        xlim=[0, 5],
        ylim=[0, 1e53],
        grid=True,
        alpha=0.5,
        x='test_infall_time',
        y='test_luminosity_of_diffusive_component'
    )
    calc.plot(
        ax=c_L_diff_t,
        xlim=[0, 5],
        ylim=[0, 1e53],
        grid=True,
        alpha=0.5,
        x='infall_time',
        y='luminosity_of_diffusive_component'
    )
    plt.savefig(f'{output_compare}{filename}_L_diff_t.png')

    # 光度
    c_L_nu_t = test.plot(
        title='t - L_nu',
        xlim=[0, 5],
        ylim=[0, 2e53],
        grid=True,
        alpha=0.5,
        x='test_infall_time',
        y='test_neutrino_luminosity'
    )
    calc.plot(
        ax=c_L_nu_t,
        xlim=[0, 5],
        ylim=[0, 2e53],
        grid=True,
        alpha=0.5,
        x='infall_time',
        y='neutrino_luminosity'
    )
    plt.savefig(f'{output_compare}{filename}_L_nu_t.png')

    # 半径
    c_r_sh_t = test.plot(
        title='t - r_sh',
        xlim=[0, 1e4],
        ylim=[0, 300],
        grid=True,
        alpha=0.5,
        x='test_infall_time',
        y='test_shock_radius'
    )
    calc.plot(
        ax=c_r_sh_t,
        xlim=[0, 1e4],
        ylim=[0, 300],
        grid=True,
        alpha=0.5,
        x='infall_time',
        y='shock_radius'
    )
    plt.savefig(f'{output_compare}{filename}_r_sh_t.png')

    # エネルギー
    c_e_g_t = test.plot(
        title='t - e_g',
        xlim=[0, 5],
        ylim=[0, 4e19],
        grid=True,
        alpha=0.5,
        x='test_infall_time',
        y='test_post_shock_binding_energy'
    )
    calc.plot(
        ax=c_e_g_t,
        xlim=[0, 5],
        ylim=[0, 4e19],
        grid=True,
        alpha=0.5,
        x='infall_time',
        y='post_shock_binding_energy'
    )
    plt.savefig(f'{output_compare}{filename}_e_g_t.png')

    # タイムスケール
    c_tau_adv_t = test.plot(
        title='t - tau_adv',
        xlim=[0, 5],
        ylim=[0, 1e-2],
        grid=True,
        alpha=0.5,
        x='test_infall_time',
        y='test_advection_time_scale'
    )
    calc.plot(
        ax=c_tau_adv_t,
        xlim=[0, 5],
        ylim=[0, 1e-2],
        grid=True,
        alpha=0.5,
        x='infall_time',
        y='advection_time_scale'
    )
    plt.savefig(f'{output_compare}{filename}_tau_adv_t.png')

    # タイムスケール
    c_tau_heat_t = test.plot(
        title='t - tau_heat',
        xlim=[0, 5],
        ylim=[0, 1e-2],
        grid=True,
        alpha=0.5,
        x='test_infall_time',
        y='test_heating_time_scale'
    )
    calc.plot(
        ax=c_tau_heat_t,
        xlim=[0, 5],
        ylim=[0, 1e-2],
        grid=True,
        alpha=0.5,
        x='infall_time',
        y='heating_time_scale'
    )
    plt.savefig(f'{output_compare}{filename}_tau_heat_t.png')

    # エネルギー
    c_E_imm_t = test.plot(
        title='t - E_imm',
        xlim=[0, 5],
        ylim=[0, 1.0e51],
        grid=True,
        alpha=0.5,
        x='test_infall_time',
        y='test_E_diag_at_a_given_mass_shell'
    )
    calc.plot(
        ax=c_E_imm_t,
        xlim=[0, 5],
        ylim=[0, 1.0e51],
        grid=True,
        alpha=0.5,
        x='infall_time',
        y='E_diag_at_a_given_mass_shell'
    )
    plt.savefig(f'{output_compare}{filename}_E_imm_t.png')

    # エネルギー
    c_E_diag_t = test.plot(
        title='t - E_diag',
        xlim=[0, 5],
        ylim=[0, 1.0e51],
        grid=True,
        alpha=0.5,
        x='test_infall_time',
        y='test_diagnostic_explosion_energy'
    )
    calc.plot(
        ax=c_E_diag_t,
        xlim=[0, 5],
        ylim=[0, 1.0e51],
        grid=True,
        alpha=0.5,
        x='infall_time',
        y='diagnostic_explosion_energy'
    )
    plt.savefig(f'{output_compare}{filename}_E_diag_t.png')

    # dot
    c_M_dot_out_t = test.plot(
        title='t - M_dot_out',
        xlim=[0, 5],
        ylim=[0, 1e33],
        grid=True,
        alpha=0.5,
        x='test_infall_time',
        y='test_outflow_rate'
    )
    calc.plot(
        ax=c_M_dot_out_t,
        xlim=[0, 5],
        ylim=[0, 1e33],
        grid=True,
        alpha=0.5,
        x='infall_time',
        y='outflow_rate'
    )
    plt.savefig(f'{output_compare}{filename}_M_dot_out_t.png')


# -----------------------------------------------------------------------------------------------------------
#
# 比較データpdf出力
#
# -----------------------------------------------------------------------------------------------------------
if progenitor_type == 'mueller':
    pdf_compare = PdfPages(f'{output_compare}{filename}_plots.pdf')
    fignums = plt.get_fignums()
    for fignum in fignums:
        plt.figure(fignum)
        pdf_compare.savefig()
    pdf_compare.close()
    plt.close('all')

# -----------------------------------------------------------------------------------------------------------
#
# メモ
#
# -----------------------------------------------------------------------------------------------------------
# print((phase == 'pre_phase').sum())  # 323
# print((phase == 'ex phase 1').sum())  # 4 -> 1
# print((phase == 'ex phase 2').sum())  # 794 -> 797
# [計算&追加] 結合エネルギーを解除するために使われたエネルギー?E_diag_dot ------------------------------- 式(37)
# data['diagnostic_explosion_energy per time'] = e_rec * M_dot_out
# E_diag_dot = data['diagnostic_explosion_energy per time']
