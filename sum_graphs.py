import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('tkagg')

# Data
# models = ['RF', 'SVR', 'XGBoost', 'L-REG']
# rmsle = [0.208024333, 0.3252044, 0.202621062, 0.2362209846]
# cross_val_rmsle = [0.147354258, 0.203888986, 0.151467661, 0.1857256398]
# mse = [55942329539, 152652972442, 53103831358, 82421283892]
# me = [236521.3088, 390708.2958, 230442.6856, 287091.0725]

models = ['without feature', 'with feature', 'combine']
rmsle= [0.118928352, 0.118753771, 0.118794342]
cross_val_rmsle =[0.129180384, 0.129242066,0.128791199 ]
me = [133746.3258, 135094.4804, 133240.7948]

# Colors
colors = ['#778da9', '#415a77', '#1b263b', '#0d1b2a']

# Subplot titles
titles = ['RMSLE', 'Cross-validated RMSLE', 'Mean Squared Error (MSE)', 'Mean Error (ME)']

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# RMSLE
bars_rmsle = axs[0].bar(models, rmsle, color=colors)
axs[0].set_title(titles[0])
axs[0].set_ylabel('RMSLE')
axs[0].set_ylim([0, max(rmsle) + 0.05])

# Cross-validated RMSLE
bars_cross_val_rmsle = axs[1].bar(models, cross_val_rmsle, color=colors)
axs[1].set_title(titles[1])
axs[1].set_ylabel('Cross-validated RMSLE')
axs[1].set_ylim([0, max(cross_val_rmsle) + 0.05])

# Mean Error (ME)
bars_me = axs[2].bar(models, me, color=colors)
axs[2].set_title(titles[3])
axs[2].set_ylabel('ME')
axs[2].set_ylim([0, max(me) + 50000])

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
