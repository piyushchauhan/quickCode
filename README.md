quickCode
=========

Code that are required for quick development and reference

Table of contents
=================

<!--ts-->
   * [quickCode](#quickCode)
   * [Table of contents](#table-of-contents)
   * [Logging](#Logging)
   * [Plots](#Plots)
   * [Pytorch](#Pytorch)
      * [GPU verification](#)
<!--te-->

Logging
=======

```
import logging
logging.basicConfig(level=logging.DEBUG,
                    format=' %(asctime)s - %(levelname)s- %(message)s')
# logging.disable(sys.maxsize)
```

Plots
========
## Plotting a diagonal correlation matrix
![Diagonal correlation matrix](https://seaborn.pydata.org/_images/many_pairwise_correlations.png)
```
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

# Generate a large random dataset
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                 columns=list(ascii_letters[26:]))

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```

Pytorch
=======

 - GPU verification
 