import numpy as np
import os

import matplotlib.pyplot as plt


for i in range(3):
    filters = np.load('./saved_params/Conv_' + str(i+1) + '_weights')

    count = 0
    for filter in filters:
        count += 1
        plt.matshow(filter, cmap=plt.cm.gray)
        plt.axis('off')

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        dir_path = './filters/6genres/Conv_' + str(i+1)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(dir_path + '/filter' + str(count) + '.png',
                    bbox_inches="tight", pad_inches=0)
