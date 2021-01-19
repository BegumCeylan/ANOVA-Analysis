# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:50:38 2021

@author: begum
"""

import random as rnd
import pandas as pd 
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.metrics import matthews_corrcoef

pw = pd.read_csv("packageWeights.csv")
pw.head()

#Draw boxplots to understand deviation distributions for each machie type

print("Machine C and D seem to have significantly different mean from others\n")

plt.figure(figsize=(20,10))

plt.subplot(121)
sns.set(font_scale=1.5)
sns.boxplot(x=pw.machine, y=pw.deviation, showmeans=True)
plt.axhline(pw.deviation.mean(), color='r', linestyle='dashed', linewidth=2)

plt.subplot(122)
machines  = ['A', 'B', 'C', 'D', 'E','F']
for mc in machines:
    subset = pw[pw['machine'] == mc]
    sns.distplot(subset['deviation'], hist = False, kde = True,
                 kde_kws = {'linewidth': 2}, label = mc)
plt.legend(prop={'size': 16}, title = 'Machines')
plt.xlabel('deviation') ; plt.ylabel('machine')
plt.show()

#State null and alternative hypothesis for ANOVA

print("H0: All samples have the same mean")
print("H1: H0 is not correct\n")

#Conduct a full ANOVA analysis using Python libraries and print the ANOVA table.

values = np.unique(np.array(pw.machine))
machines = []
for idx in values:
    add = list(pw[pw.machine == idx].deviation)
    machines.append(add)
    
F, p = stats.f_oneway(machines[0],machines[1],machines[2],machines[3],machines[4],machines[5])

#Draw a conclusion of the hypothesis test (for α= 0.05)

if p > 0.05:
    print("Fail to reject HO")
    print("There is no significant difference in machines\n")
else:
    print("Reject HO")
    print("At least one of the machines has significantly different sample\n")

#There is at least one mean not equal
#To determine which ones are different using the Tukey's test
    
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

deviation = np.array(pw.deviation)
machine = np.array(pw.machine)

mc = MultiComparison(deviation,machine)
result = mc.tukeyhsd()
print(result) 

#Plot it
result.plot_simultaneous()

print("According to Tukey's test C and D are different from others")