
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Define a list of color codes - as you wish
clr_codes = ["#194366","#088da5","#84B853","#FFE300","#4F5324","#C4C522","#142330","#0c3c60"]

#Global settings for matplotlib.pyplot
plt.rcParams['figure.figsize'] = (12,5)
plt.rcParams['grid.linestyle'] = '-.'
plt.rcParams["font.size"] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['figure.dpi'] = 500

#Theme settings for Seaborn
sns.set_theme(style="whitegrid", font="serif", palette = clr_codes,
              rc={"figure.figsize":(12, 5),
                  'axes.edgecolor': 'black',
                  'axes.spines.left': True,
                  'axes.spines.bottom': True,
                  'axes.spines.right': False,
                  'axes.spines.top': False,
                  'axes.grid': True,
                  'grid.color': "black",
                  'grid.linestyle': ':'})

#Set precision of printouts of pandas objects to reduce visual clutter
pd.set_option("display.precision", 3)