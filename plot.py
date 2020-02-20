import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
pd.options.mode.chained_assignment = None

input_file_1 = './output/calc/main_data_s.dat'
input_file_2 = './output/calc/main_data_u.dat'
input_file_3 = './output/calc/main_data_m.dat'
if not os.path.exists(input_file_1):
    print(f"Error: Don't open {input_file_1}")
if not os.path.exists(input_file_2):
    print(f"Error: Don't open {input_file_2}")
if not os.path.exists(input_file_3):
    print(f"Error: Don't open {input_file_3}")

data_1 = pd.read_csv(
    input_file_1,
    skiprows=1,
    names=['M', 'metal', 'Type', 'E_expl', 'M_ns', 'M_BH'],
    sep='\s{2,}',
    engine='python',
)

data_2 = pd.read_csv(
    input_file_2,
    skiprows=1,
    names=['M', 'metal', 'Type', 'E_expl', 'M_ns', 'M_BH'],
    sep='\s{2,}',
    engine='python',
)

data_3 = pd.read_csv(
    input_file_3,
    skiprows=1,
    names=['M', 'metal', 'Type', 'E_expl', 'M_ns', 'M_BH'],
    sep='\s{2,}',
    engine='python',
)

output_prefix = './output/calc/'

# 爆発のエネルギー
_E_expl = data_1.plot(
    title='M - E_expl',
    xlim=[10, 78],
    ylim=[0, 3e51],
    grid=True,
    kind='scatter',
    alpha=0.5,
    x='M',
    y='E_expl'
)
data_2.plot(
    ax=_E_expl,
    xlim=[10, 78],
    ylim=[0, 3e51],
    grid=True,
    kind='scatter',
    color='Pink',
    alpha=0.5,
    x='M',
    y='E_expl'
)
data_3.plot(
    ax=_E_expl,
    xlim=[10, 78],
    ylim=[0, 3e51],
    grid=True,
    kind='scatter',
    color='Green',
    alpha=0.5,
    x='M',
    y='E_expl'
)
_E_expl.set_xlabel("M_zams[Msun]")
_E_expl.set_ylabel("E_expl[erg]")
plt.savefig(f'{output_prefix}E_expl_all.png')

# ニュートリノ質量
_M_ns = data_1.plot(
    title='M - M_ns',
    xlim=[10, 30],
    ylim=[1.2, 2.3],
    grid=True,
    kind='scatter',
    alpha=0.5,
    x='M',
    y='M_ns'
)
data_2.plot(
    ax=_M_ns,
    xlim=[10, 30],
    ylim=[1.2, 2.3],
    grid=True,
    kind='scatter',
    color='Pink',
    alpha=0.5,
    x='M',
    y='M_ns'
)
data_3.plot(
    ax=_M_ns,
    xlim=[10, 30],
    ylim=[1.2, 2.3],
    grid=True,
    kind='scatter',
    color='Green',
    alpha=0.5,
    x='M',
    y='M_ns'
)
_M_ns.set_xlabel("M_zams[Msun]")
_M_ns.set_ylabel("M_ns[Msun]")
plt.savefig(f'{output_prefix}M_ns_narrow_all.png')

# ブラックホール質量
_M_BH = data_1.plot(
    title='M - M_BH',
    xlim=[10, 78],
    ylim=[0, 35],
    grid=True,
    kind='scatter',
    legend=True,
    alpha=0.5,
    x='M',
    y='M_BH'
)
data_2.plot(
    ax=_M_BH,
    xlim=[10, 78],
    ylim=[0, 35],
    grid=True,
    kind='scatter',
    legend=True,
    color='Pink',
    alpha=0.5,
    x='M',
    y='M_BH'
)
data_3.plot(
    ax=_M_BH,
    xlim=[10, 78],
    ylim=[0, 35],
    grid=True,
    kind='scatter',
    legend=True,
    color='Green',
    alpha=0.5,
    x='M',
    y='M_BH'
)
_M_BH.set_xlabel("M_zams[Msun]")
_M_BH.set_ylabel("M_BH[Msun]")
plt.savefig(f'{output_prefix}M_BH_all.png')

pdf = PdfPages(f'{output_prefix}plots_all.pdf')
fignums = plt.get_fignums()
for fignum in fignums:
    plt.figure(fignum)
    pdf.savefig()
pdf.close()
plt.close('all')
