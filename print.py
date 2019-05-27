import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd


results = pickle.load( open("results_second.pkl", "rb") )
data = pd.DataFrame.from_dict(results).applymap(lambda element: element["val_loss"][-1])
#data = data.applymap(lambda val: np.log(val))

fig = plt.figure(figsize=(20,15))
ax1 = plt.subplot2grid((20,20), (0,1), colspan=19, rowspan=19) # main
ax2 = plt.subplot2grid((20,20), (19,1), colspan=19, rowspan=1) # bottom
ax3 = plt.subplot2grid((20,20), (0,0), colspan=1, rowspan=19) # left
ax3.set(ylabel="data")
ax2.set(xlabel="network")
sb.heatmap(data,                                        ax=ax1,  annot=True, cbar=False, xticklabels=False, yticklabels=False,  cmap="RdYlGn_r") # main
sb.heatmap(pd.DataFrame(data.sum(axis=0)).transpose(),  ax=ax2,  annot=True, cbar=False, xticklabels=True, yticklabels=False,   cmap="RdYlGn_r") # bottom
sb.heatmap(pd.DataFrame(data.sum(axis=1)),              ax=ax3,  annot=True, cbar=False, xticklabels=False, yticklabels=True,   cmap="RdYlGn_r") # left
plt.show()

