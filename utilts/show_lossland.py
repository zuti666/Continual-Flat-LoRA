import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # 添加三维绘图库

def plot_loss_landscape(h5_file, title="Loss Landscape",savepth = None):
    """
    加载 HDF5 文件中的 Loss Landscape 数据，并绘制等高线图。
    文件中必须包含：
      - xcoordinates: 横轴扰动坐标
      - ycoordinates: 纵轴扰动坐标
      - train_loss: 对应的平均损失值矩阵
    """
    with h5py.File(h5_file, 'r') as f:
        x_coords = f['xcoordinates'][:]
        y_coords = f['ycoordinates'][:]
        loss_grid = f['train_loss'][:]
    
    # 构造网格
    X, Y = np.meshgrid(x_coords, y_coords)
    
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, Y, loss_grid.T, levels=20, cmap='viridis')
    plt.colorbar(cp)
    plt.title(title)
    plt.xlabel("Perturbation X")
    plt.ylabel("Perturbation Y")
    plt.savefig(f'{savepth}_2d.png')
    plt.show()
    print(f"save to {savepth}_2d.png")


    # 绘制三维图
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')  # 添加三维子图
    ax.plot_surface(X, Y, loss_grid.T, cmap='viridis')  # 绘制三维曲面图
    ax.set_title(title)
    ax.set_xlabel("Perturbation X")
    ax.set_ylabel("Perturbation Y")
    ax.set_zlabel("Loss")
    plt.savefig(f'{savepth}_3d.png')
    plt.show()

def plot_hessian_stats(h5_file, title="Hessian Statistics"):
    """
    加载 HDF5 文件中的 Hessian 统计数据，并绘制特征值分布直方图。
    文件中必须包含：
      - dominant_eigs_list: 各 batch 计算得到的 dominant eigenvalue 列表
      - median_eig, min_eig, max_eig: 统计值
    """
    with h5py.File(h5_file, 'r') as f:
        dominant_eigs = f['dominant_eigs_list'][:]
        median_eig = f['median_eig'][()]
        min_eig = f['min_eig'][()]
        max_eig = f['max_eig'][()]
    
    print("Hessian Statistics:")
    print(f"Min Eigenvalue: {min_eig:.4f}")
    print(f"Max Eigenvalue: {max_eig:.4f}")
    print(f"Median Eigenvalue: {median_eig:.4f}")
    
    plt.figure(figsize=(8, 6))
    sns.histplot(dominant_eigs, bins=30, kde=True)
    plt.title(title)
    plt.xlabel("Dominant Eigenvalue")
    plt.ylabel("Frequency")
    plt.axvline(median_eig, color='r', linestyle='--', label=f"Median: {median_eig:.4f}")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    """ --------- lora  method -10-----------------------"""

    """ ------------------Eval 1-1 lora  T5-small  loss landscape  order1------------------------ """
    #  order1 - 1-dbpedia
    # loss_landscape_lora_file = "logs_and_outputs/order_1/outputs/1-dbpedia/1_original_LoRA/lossShape_lora_only-evalDataset-orial.h5"
    # save_pth_lora =            'logs_and_outputs/order_1/outputs/1-dbpedia/1_original_LoRA/lossShape_T5small-10_lora'
    # plot_loss_landscape(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 2-amazon
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_1/outputs/2-amazon/adapter_T5small_lora-10/lossShape_lora_only-predictDataset-orial.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_1/outputs/2-amazon/adapter_T5small_lora-10/lossShape_T5small-10_lora'
    # plot_loss_landscape(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 3-yahoo
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_1/outputs/3-yahoo/adapter_T5small_lora-10/lossShape_lora_only-predictDataset-orial.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_1/outputs/3-yahoo/adapter_T5small_lora-10/lossShape_T5small-10_lora'
    # plot_loss_landscape(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 4-agnews
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_1/outputs/4-agnews/adapter_T5small_lora-10/lossShape_lora_only-predictDataset-orial.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_1/outputs/4-agnews/adapter_T5small_lora-10/lossShape_T5small-10_lora'
    # plot_loss_landscape(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    

    """------------------Eval 1-1 lora  T5-small  loss landscape  order2------------------------ """
    #  order1 - 1-yahoo
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_2/outputs/1-dbpedia/adapter_T5small_lora-10/lossShape_lora_only-evalDataset-orial.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_2/outputs/1-dbpedia/adapter_T5small_lora-10/lossShape_T5small-10_lora'
    # plot_loss_landscape(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 2-amazon
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_2/outputs/2-amazon/adapter_T5small_lora-10/lossShape_lora_only-evalDataset-orial.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_2/outputs/2-amazon/adapter_T5small_lora-10/lossShape_T5small-10_lora'
    # plot_loss_landscape(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 3-agnews
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_2/outputs/3-agnews/adapter_T5small_lora-10/lossShape_lora_only-evalDataset-orial.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_2/outputs/3-agnews/adapter_T5small_lora-10/lossShape_T5small-10_lora'
    # plot_loss_landscape(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 4-dbpedia
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_2/outputs/4-yahoo/adapter_T5small_lora-10/lossShape_lora_only-evalDataset-orial.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_2/outputs/4-yahoo/adapter_T5small_lora-10/lossShape_T5small-10_lora'
    # plot_loss_landscape(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    








    """------------------Eval 1-1 lora  T5-small  loss landscape  order3------------------------ """
    #  order1 - 1-yahoo
    # loss_landscape_lora_file = "logs_and_outputs/order_1/outputs/1-dbpedia/1_original_LoRA/lossShape_lora_only-evalDataset-orial.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_3/outputs/1-yahoo/adapter_T5small_lora-10/lossShape_T5small-10_lora'
    # plot_loss_landscape(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 2-amazon
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_3/outputs/2-amazon/adapter_T5small_lora-10/lossShape_lora_only-evalDataset-orial.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_3/outputs/2-amazon/adapter_T5small_lora-10/lossShape_T5small-10_lora'
    # plot_loss_landscape(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 3-agnews
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_3/outputs/3-agnews/adapter_T5small_lora-10/lossShape_lora_only-evalDataset-orial.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_3/outputs/3-agnews/adapter_T5small_lora-10/lossShape_T5small-10_lora'
    # plot_loss_landscape(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 4-dbpedia
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_3/outputs/4-dbpedia/adapter_T5small_lora-10/lossShape_lora_only-evalDataset-orial.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_3/outputs/4-dbpedia/adapter_T5small_lora-10/lossShape_T5small-10_lora'
    # plot_loss_landscape(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    

    """------------------Eval 1-1 Nlora   only taskT5-small  loss landscape  order1------------------------ """
    # order1 - 1-dbpedia
    # loss_landscape_lora_file = "logs_and_outputs_T5small_Nlora/order_1/outputs/1-dbpedia/adapter_T5small_Nlora_onlytasknew-10/lossShape_Nlora_onlytask-predictDataset-2.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_Nlora/order_1/outputs/1-dbpedia/adapter_T5small_Nlora_onlytasknew-10/lossShape_Nlora_onlytask'
    # plot_loss_landscape(loss_landscape_lora_file, title="NLoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # # order1 - 2-amazon
    # loss_landscape_lora_file = "logs_and_outputs_T5small_Nlora/order_1/outputs/2-amazon/adapter_T5small_Nlora_onlytasknew-10/lossShape_Nlora_onlytask-predictDataset-2.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_Nlora/order_1/outputs/2-amazon/adapter_T5small_Nlora_onlytasknew-10/lossShape_Nlora_onlytask'
    # plot_loss_landscape(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # # #  order1 - 3-yahoo
    # loss_landscape_lora_file = "logs_and_outputs_T5small_Nlora/order_1/outputs/3-yahoo/adapter_T5small_Nlora_onlytasknew-10/lossShape_Nlora_onlytask-predictDataset-2.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_Nlora/order_1/outputs/3-yahoo/adapter_T5small_Nlora_onlytasknew-10/lossShape_T5small-onlytask'
    # plot_loss_landscape(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 4-agnews
    # loss_landscape_lora_file = "logs_and_outputs_T5small_Nlora/order_1/outputs/4-agnews/adapter_T5small_Nlora_onlytasknew-10/lossShape_Nlora_onlytask-predictDataset-2.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_Nlora/order_1/outputs/4-agnews/adapter_T5small_Nlora_onlytasknew-10/lossShape_T5small-onlytask'
    # plot_loss_landscape(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    


    """------------------Eval 1-1 Nlora  T5-small full task loss landscape  order1------------------------"""
    #order1 - 1-dbpedia
    # loss_landscape_lora_file = "logs_and_outputs_T5small_Nlora/order_1/outputs/1-dbpedia/adapter_T5small_Nlora_onlytasknew-10/lossShape_Nlora_full-predictDataset.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_Nlora/order_1/outputs/1-dbpedia/adapter_T5small_Nlora_onlytasknew-10/lossShape_Nlora_full'
    # plot_loss_landscape(loss_landscape_lora_file, title="NLoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 2-amazon
    # loss_landscape_lora_file = "logs_and_outputs_T5small_Nlora/order_1/outputs/2-amazon/adapter_T5small_Nlora_onlytasknew-10/lossShape_Nlora_onlytask-predictDataset-2.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_Nlora/order_1/outputs/2-amazon/adapter_T5small_Nlora_onlytasknew-10/lossShape_Nlora_onlytask'
    # plot_loss_landscape(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 3-yahoo
    # loss_landscape_lora_file = "logs_and_outputs_T5small_Nlora/order_1/outputs/3-yahoo/adapter_T5small_Nlora_onlytasknew-10/lossShape_Nlora_onlytask-predictDataset-2.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_Nlora/order_1/outputs/3-yahoo/adapter_T5small_Nlora_onlytasknew-10/lossShape_T5small-10_Nlora'
    # plot_loss_landscape(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 4-agnews
    # loss_landscape_lora_file = "logs_and_outputs_T5small_Nlora/order_1/outputs/4-agnews/adapter_T5small_Nlora_onlytasknew-10/lossShape_Nlora_onlytask-predictDataset-2.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_Nlora/order_1/outputs/4-agnews/adapter_T5small_Nlora_onlytasknew-10/lossShape_T5small-10_Nlora'
    # plot_loss_landscape(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    

    """------------------Eval T5 initial model------------------------"""
    # order1 - 1-dbpedia  --1 --origin model full perturb
    loss_landscape_lora_file = "initial_model/t5-small/lossLandscape_1.h5"
    save_pth_lora =            'initial_model/t5-small/lossLandscape_1'
    plot_loss_landscape(loss_landscape_lora_file, title="Dispeturb Loss Landscape",savepth = save_pth_lora)

    # order1 - 1-dbpedia --2 --origin model with LoRA no train -- only W
    loss_landscape_lora_file = "initial_model/t5-small/lossLandscape_2-1.h5"
    save_pth_lora =            'initial_model/t5-small/lossLandscape_2-1'
    plot_loss_landscape(loss_landscape_lora_file, title="Dispeturb Loss Landscape",savepth = save_pth_lora)

    # order1 - 1-dbpedia --2 --origin model with LoRA no train -- only AB
    loss_landscape_lora_file = "initial_model/t5-small/lossLandscape_2-2.h5"
    save_pth_lora =            'initial_model/t5-small/lossLandscape_2-2'
    plot_loss_landscape(loss_landscape_lora_file, title="Dispeturb Loss Landscape",savepth = save_pth_lora)


    # order1 - 1-dbpedia --2 --origin model with LoRA no train --  W -AB
    loss_landscape_lora_file = "initial_model/t5-small/lossLandscape_2-3.h5"
    save_pth_lora =            'initial_model/t5-small/lossLandscape_2-3'
    plot_loss_landscape(loss_landscape_lora_file, title="Dispeturb Loss Landscape",savepth = save_pth_lora)




    # order1 - 1-dbpedia --3 --origin model with LoRA after train -- only W
    loss_landscape_lora_file = "logs-outputs_T5/order_1/outputs/1-dbpedia/adapter_lora/lossLandscape_3-1.h5"
    save_pth_lora =            'logs-outputs_T5/order_1/outputs/1-dbpedia/adapter_lora/lossLandscape_3-1'
    plot_loss_landscape(loss_landscape_lora_file, title="Dispeturb Loss Landscape",savepth = save_pth_lora)

    # order1 - 1-dbpedia --3 --origin model with LoRA after train -- only AB
                                
    loss_landscape_lora_file = "logs-outputs_T5/order_1/outputs/1-dbpedia/adapter_lora/lossLandscape_3-2.h5"
    save_pth_lora =            'logs-outputs_T5/order_1/outputs/1-dbpedia/adapter_lora/lossLandscape_3-2'
    plot_loss_landscape(loss_landscape_lora_file, title="Dispeturb Loss Landscape",savepth = save_pth_lora)


    # order1 - 1-dbpedia --3 --origin model with LoRA after train --  W -AB
    loss_landscape_lora_file = "logs-outputs_T5/order_1/outputs/1-dbpedia/adapter_lora/lossLandscape_3-3.h5"
    save_pth_lora =            'logs-outputs_T5/order_1/outputs/1-dbpedia/adapter_lora/lossLandscape_3-3'
    plot_loss_landscape(loss_landscape_lora_file, title="Dispeturb Loss Landscape",savepth = save_pth_lora)
    
    
    