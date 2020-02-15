
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

pd.options.mode.chained_assignment = None

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
beta_expl = 4.0
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


# -----------------------------------------------------------------------------------------------------------
#
# データの読み込み
#
# -----------------------------------------------------------------------------------------------------------
path = './data/s12.4'
data = pd.read_csv(
    path,
    header=1,
    sep='\s{2,}',
    engine='python',
    na_values='---'
)
data['average densty'] = float(np.nan)
data['infall time'] = float(np.nan)
data['mass accretion rate'] = float(np.nan)
data['gain radius'] = float(np.nan)
data['pns radius'] = float(np.nan)
data['cool time scale'] = float(np.nan)
data['bindin energy'] = float(np.nan)
data['luminosity of accretion component'] = float(np.nan)
data['luminosity of diffusive component'] = float(np.nan)
data['neutrino luminosity'] = float(np.nan)
data['redsift factor'] = float(np.nan)
data['shock radius'] = float(np.nan)
data['radius max'] = float(np.nan)
data['radius min'] = float(np.nan)
data['post shock binding energy'] = float(np.nan)
data['advection time scale'] = float(np.nan)
data['heating time scale'] = float(np.nan)
data['critical time scale ration'] = float(np.nan)
data['effeciency parameter'] = float(np.nan)
data['avarage shock velocity'] = float(np.nan)
data['post shock velocity'] = float(np.nan)
data['escape velocity'] = float(np.nan)
data['post shock temperature'] = float(np.nan)
data['unshocked material binding energy'] = float(np.nan)
data['nuclear burning energy'] = float(np.nan)
data['E_diag at a given mass shell'] = float(np.nan)
data['diagnostic explosion energy'] = float(np.nan)
data['mass accretion rate against mass out'] = float(np.nan)
data['outflow rate'] = float(np.nan)
data['mass of baryonic neutron star'] = float(np.nan)
data['mass of proto neutron star'] = float(np.nan)
data['final explosion energy'] = float(np.nan)
data['pre or phase1'] = float(np.nan)


# -----------------------------------------------------------------------------------------------------------
#
# 変数の定義
#
# -----------------------------------------------------------------------------------------------------------
max_row = len(data.index)  # 列数
M = data['cell outer total mass']  # 質量M(g)
r = data['cell outer radius']  # 半径r(cm)
rho = data['cell density']  # 密度rho(g/cm^3)
rho_bar = data['average densty']  # 平均密度ρ_bar(g/cm^3)
t = data['infall time']  # 落下時間t(s)
M_dot = data['mass accretion rate']  # 質量降着率M_dot(g/s)
r_g = data['gain radius']  # ゲイン半径r_g(cm)
r_PNS = data['pns radius']  # 原始中性子星の半径r_PNS(cm)
tau_cool = data['cool time scale']  # 中性子星結合エネルギーが放射されるタイムスケールtau_cool(s)
E_bind = data['bindin energy']  # 冷たい中性子星の結合エネルギーE_bind(erg)
L_acc = data['luminosity of accretion component']  # 降着成分による光度L_acc(erg/s)
L_diff = data['luminosity of diffusive component']  # 拡散による光度L_diff(erg/s)
L_nu = data['neutrino luminosity']  # ニュートリノ光度L_nu(erg/s)
alpha = data['redsift factor']  # 赤方偏移因子alpha
r_sh = data['shock radius']  # 衝撃波の半径r_sh(cm)
r_max = data['radius max']  # r_shとr_gの大きい方r_max(cm)
r_min = data['radius min']  # r_shとr_gの小さい方r_min(cm)
e_g = data['post shock binding energy']  # ショック後の結合エネルギーe_g(erg/g)
tau_adv = data['advection time scale']  # 降着物質がゲイン領域を移動するタイムスケールtau_adv(s)
tau_heat = data['heating time scale']  # 降着物質がゲイン領域を移動するタイムスケールtau_heat(s)
tau_ration = data['critical time scale ration']  # pre->exの基準tau_ration
eta_acc = data['effeciency parameter']  # 降着のパラメーター?eta_acc(erg/g)
v_sh = data['avarage shock velocity']  # 衝撃波の速さv_sh(cm/s)
v_post = data['post shock velocity']  # 衝撃波通過後の速さv_post(cm/s)
v_esc = data['escape velocity']  # 脱出速さv_esc(cm/s)
T_sh = data['post shock temperature']  # 衝撃波直後の温度T_sh(K)
e_bind = data['unshocked material binding energy']  # 衝撃波前の結合エネルギーe_bind(erg/g)
e_burn = data['nuclear burning energy']  # 核融合によるエネルギーe_burn(erg/g)
E_imm = data['E_diag at a given mass shell']  # 衝撃波が到達した時の診断エネルギーE_imm(erg)
E_diag = data['diagnostic explosion energy']  # 診断爆発エネルギーE_diag(erg)
M_dot_acc = data['mass accretion rate against mass out']  # 単位時間降着質量M_dot_acc
M_dot_out = data['outflow rate']  # 単位時間あたりに出ていく質量M_dot_out
M_by = data['mass of baryonic neutron star']  # 原始中性子星M_by
M_ns = data['mass of proto neutron star']  # 最終的な中性子星の質量M_ns
E_expl = data['final explosion energy']  # 爆発のエネルギーE_expl
phase = data['pre or phase1']
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
r_sh = alpha_turb * 0.55*_km * (L_nu / 1e52) ** (4 / 9) * alpha ** (4 / 9) * (
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
tau_heat = 150*_ms * (np.abs(e_g) / 1e19) * (r_g / (100*_km)) ** 2 * \
    (L_nu / 1e52) ** (-1) * (alpha ** 3) ** (-1) * (M / M_sun) ** (-2)
# pre->exの基準tau_ration
tau_ration = tau_adv / tau_heat
# 降着のパラメーター?eta_acc(erg/g) -- 式(31)
eta_acc = tau_adv / tau_heat * np.abs(e_g)
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
        phase[i] = 'pre phase'
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
        M_dot_out[i] = eta_out * eta_acc[i] * M_dot_acc[i] / np.abs(e_g[i])
        M_by[i] = 0
        M_ns[i] = 0
    else:
        e_bind[i] = - 1 * G * M[i] / r[i]
        if v_post[i-1] < v_esc[i-1] and condition_2 == False:
            phase[i] = 'ex phase 1'
            condition_1 = True
            # 爆発段階I
            # tau_ration[i] > 1の条件を満たす(1度でも)
            # v_post[i-1] > v_esc[i-1]の条件を満たさない
            dm = M[i] - M[i-1]
            M_ini = M[i-1]
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
                e_rec * eta_acc[i] / np.abs(e_g[i]) * dm
            v_sh[i] = 0.794 * np.sqrt(E_imm[i] / (M[i] - M_ini)) * \
                ((M[i] - M_ini) / (rho[i] * r[i] ** 3)) ** 0.19
            v_post[i] = (beta_expl - 1) / beta_expl * v_sh[i]
            #
            # T_sh -> Xi -> e_burn -> E_imm/E_diag
            #
            T_sh[i] = (3 * (beta_expl - 1) / (radiation_constant *
                                              beta_expl) * rho[i] * v_sh[i] ** 2) ** (1 / 4)
            flashing_method(i)
            E_imm[i] += alpha_out * (e_bind[i] + e_burn[i]) * dm
            E_diag[i] += alpha_out * (e_bind[i] + e_burn[i]) * dm
            #
            # E_diagが負の時はブラックホールになる
            #
            if E_diag[i] < 0 or phase[i-1] == 'BH':
                # ブラックホール
                # E_diagが一度でも負になったり
                # ~~~~~~~~したらここに来る
                phase[i] = 'BH'
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
                    M_dot_acc[i] / np.abs(e_g[i])
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
            phase[i] = 'ex phase 2'
            condition_2 = True
            dm = M[i] - M[i-1]
            M_ini = M[i-1]
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
            v_sh[i] = 0.794 * np.sqrt(E_imm[i] / (M[i] - M_ini)) * \
                ((M[i] - M_ini) / (rho[i] * r[i] ** 3)) ** 0.19
            v_post[i] = (beta_expl - 1) / beta_expl * v_sh[i]
            #
            # T_sh -> Xi -> e_burn -> E_imm/E_diag
            #
            T_sh[i] = (3 * (beta_expl - 1) / (radiation_constant *
                                              beta_expl) * rho[i] * v_sh[i] ** 2) ** (1 / 4)
            flashing_method(i)
            E_imm[i] += alpha_out * (e_bind[i] + e_burn[i]) * dm
            E_diag[i] += alpha_out * (e_bind[i] + e_burn[i]) * dm
            #
            # E_diagが負の時はブラックホールになる
            #
            if phase[i-1] == 'BH' or E_diag[i] < 0:
                # ブラックホール
                # E_diagが一度でも負になったり
                # ~~~~~~~~したらここに来る
                phase[i] = 'BH'
                v_sh[i] = 1e-10
                v_post[i] = 0
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

if phase[max_row-1] == 'pre phase':
    phase[max_row-1] = 'BH'
    E_expl[max_row-1] = 0


# -----------------------------------------------------------------------------------------------------------
#
# 単位変換(出力用)
#
# -----------------------------------------------------------------------------------------------------------
for i in range(max_row):
    r_g[i] = r_g[i]/_km
    r_sh[i] = r_sh[i]/_km
    v_post[i] = v_post[i]/_km
    v_esc[i] = v_esc[i]/_km


# -----------------------------------------------------------------------------------------------------------
#
# データ格納
#
# -----------------------------------------------------------------------------------------------------------
data['average densty'] = rho_bar
data['infall time'] = t
data['mass accretion rate'] = M_dot
data['gain radius'] = r_g
data['pns radius'] = r_PNS
data['cool time scale'] = tau_cool
data['bindin energy'] = E_bind
data['luminosity of accretion component'] = L_acc
data['luminosity of diffusive component'] = L_diff
data['neutrino luminosity'] = L_nu
data['redsift factor'] = alpha
data['shock radius'] = r_sh
data['radius max'] = r_max
data['radius min'] = r_min
data['post shock binding energy'] = e_g
data['advection time scale'] = tau_adv
data['heating time scale'] = tau_heat
data['critical time scale ration'] = tau_ration
data['effeciency parameter'] = eta_acc
data['avarage shock velocity'] = v_sh
data['post shock velocity'] = v_post
data['escape velocity'] = v_esc
data['post shock temperature'] = T_sh
data['unshocked material binding energy'] = e_bind
data['nuclear burning energy'] = e_burn
data['E_diag at a given mass shell'] = E_imm
data['diagnostic explosion energy'] = E_diag
data['mass accretion rate against mass out'] = M_dot_acc
data['outflow rate'] = M_dot_out
data['mass of baryonic neutron star'] = M_by
data['mass of proto neutron star'] = M_ns
data['final explosion energy'] = E_expl


# -----------------------------------------------------------------------------------------------------------
#
# プロット処理
#
# -----------------------------------------------------------------------------------------------------------
_r_t = data.plot(
    title='t - r',
    xlim=[0, 5],
    ylim=[0, 300],
    grid=True,
    x='infall time',
    y='gain radius'
)
data.plot(
    ax=_r_t,
    xlim=[0, 5],
    ylim=[0, 300],
    grid=True,
    x='infall time',
    y='shock radius'
)

_tau_t = data.plot(
    title='t - tau',
    xlim=[0, 5],
    ylim=[0, 1e-2],
    grid=True,
    x='infall time',
    y='advection time scale'
)
data.plot(
    ax=_tau_t,
    xlim=[0, 5],
    ylim=[0, 1e-2],
    grid=True,
    x='infall time',
    y='heating time scale'
)

_v_t = data.plot(
    title='t - v',
    xlim=[0, 5],
    ylim=[0, 3e4],
    grid=True,
    x='infall time',
    y='post shock velocity'
)
data.plot(
    ax=_v_t,
    xlim=[0, 5],
    ylim=[0, 3e4],
    grid=True,
    x='infall time',
    y='escape velocity'
)

_T_t = data.plot(
    title='t - T_sh',
    xlim=[0, 5],
    grid=True,
    x='infall time',
    y='post shock temperature'
)


_Xi_M = data.plot(
    title='M - Xi',
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='neutrons'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='H1'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='He3'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='He4'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='C12'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='N14'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='O16'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='Ne20'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='Mg24'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='Si28'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='S32'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='Ar36'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='Ca40'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='Ti44'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='Cr48'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='Fe52'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='Fe54'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='Ni56'
)
data.plot(
    ax=_Xi_M,
    ylim=[1e-4, 1],
    grid=True,
    x='cell outer total mass',
    y='Fe56'
)

plt.show()


# -----------------------------------------------------------------------------------------------------------
#
# CSV出力
#
# -----------------------------------------------------------------------------------------------------------
data.to_csv("result.csv")
subprocess.call(["open", "result.csv"])


# -----------------------------------------------------------------------------------------------------------
#
# メモ
#
# -----------------------------------------------------------------------------------------------------------
# print((phase == 'pre phase').sum())  # 323
# print((phase == 'ex phase 1').sum())  # 4 -> 1
# print((phase == 'ex phase 2').sum())  # 794 -> 797
# [計算&追加] 結合エネルギーを解除するために使われたエネルギー?E_diag_dot ------------------------------- 式(37)
# data['diagnostic explosion energy per time'] = e_rec * M_dot_out
# E_diag_dot = data['diagnostic explosion energy per time']
