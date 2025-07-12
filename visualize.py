import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def vis_ID_OOD_Energy(vis_ID, vis_OOD, numOfai, save_folder):
    kde_ID = gaussian_kde(vis_ID)
    x_ID = np.linspace(min(vis_ID), max(vis_ID), 1000)
    y_ID = kde_ID(x_ID)

    kde_OOD = gaussian_kde(vis_OOD)
    x_OOD = np.linspace(min(vis_OOD), max(vis_OOD), 1000)
    y_OOD = kde_OOD(x_OOD)

    file_name = f"energy_{numOfai+1}_hist.png"
    plt.plot(x_ID, y_ID, color= 'blue', label= 'ID', alpha= 0.35)
    plt.fill_between(x_ID, y_ID, color='blue', alpha=0.1)
    plt.plot(x_OOD, y_OOD, color= 'red', label= 'OOD', alpha= 0.35)
    plt.fill_between(x_OOD, y_OOD, color='red', alpha=0.1)
    plt.xlabel('Energy Score')
    plt.ylabel('Frequency')
    plt.ylim([0.0, 0.47])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_folder}/{file_name}')
    plt.show()
    plt.clf()

    return x_ID, y_ID, x_OOD, y_OOD

def vis_ALL_Energy(x_ID, y_ID, x_OOD, y_OOD, arrV, number, save_folder):
    kde_Vk = gaussian_kde(arrV)
    x_Vk = np.linspace(min(arrV), max(arrV), 1000)
    y_Vk = kde_Vk(x_Vk)
    
    file_name = f"energy_epoch_{number}_hist.png"
    plt.plot(x_ID, y_ID, color= 'blue', label= 'ID', alpha= 0.35)
    plt.fill_between(x_ID, y_ID, color='blue', alpha=0.1)
    plt.plot(x_OOD, y_OOD, color= 'red', label= 'OOD', alpha= 0.35)
    plt.fill_between(x_OOD, y_OOD, color='red', alpha=0.1)
    plt.plot(x_Vk, y_Vk, color= 'green', label= 'Vk', alpha= 0.35)
    plt.fill_between(x_Vk, y_Vk, color='green', alpha=0.1)

    plt.xlabel('Energy Score')
    plt.ylabel('Frequency')
    plt.ylim([0.0, 0.47])
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_folder}/{file_name}')
    plt.show()
    plt.clf()

    return x_Vk, y_Vk
