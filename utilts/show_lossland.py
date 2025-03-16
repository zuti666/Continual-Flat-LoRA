import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # 添加三维绘图库

import pandas as pd
from scipy.stats import skew, kurtosis
import os

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


    # 计算统计特征
    loss_min = np.min(loss_grid)
    loss_max = np.max(loss_grid)
    loss_mean = np.mean(loss_grid)
    loss_median = np.median(loss_grid)
    loss_std = np.std(loss_grid)
    loss_skew = skew(loss_grid.flatten())
    loss_kurt = kurtosis(loss_grid.flatten())

    # 计算局部梯度变化率 (用作 flatness 指标)
    grad_x = np.gradient(loss_grid, axis=0)
    grad_y = np.gradient(loss_grid, axis=1)
    grad_norm = np.sqrt(grad_x**2 + grad_y**2)
    mean_flatness = np.mean(grad_norm)
    min_flatness = np.min(grad_norm)
    max_flatness = np.max(grad_norm)

    # 统计分析结果
    analysis_results = {
        "Min Loss": loss_min,
        "Max Loss": loss_max,
        "Mean Loss": loss_mean,
        "Median Loss": loss_median,
        "Loss Std Dev": loss_std,
        "Skewness": loss_skew,
        "Kurtosis": loss_kurt,
        "Mean Flatness": mean_flatness,
        "Min Flatness": min_flatness,
        "Max Flatness": max_flatness
    }

    # 保存统计数据
    df = pd.DataFrame(analysis_results, index=[0])
    csv_path = f"{savepth}_stats.csv"
    df.to_csv(csv_path, index=False)

    # 打印统计结果
    print("=== Loss Landscape Analysis ===")
    # print(df)
    print(f"Results saved to {csv_path}")
    
    # 2D 等高线图
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

def read_sum_table(output_file="merged_results.csv"):
    # 创建用于存储结果的 DataFrame
    results_df = pd.DataFrame()

    csv_files = {
        '1'  : 'initial_model/t5-small/lossLandscape_1_stats.csv',
        '2-1': 'initial_model/t5-small/lossLandscape_2-1_stats.csv',
        '2-2': 'initial_model/t5-small/lossLandscape_2-2_stats.csv',
        '2-3': 'initial_model/t5-small/lossLandscape_2-3_stats.csv',
        '3-1': 'logs-outputs_T5/order_1/outputs/1-dbpedia/adapter_lora/lossLandscape_3-1_stats.csv',
        '3-2': 'logs-outputs_T5/order_1/outputs/1-dbpedia/adapter_lora/lossLandscape_3-2_stats.csv',
        '3-3': 'logs-outputs_T5/order_1/outputs/1-dbpedia/adapter_lora/lossLandscape_3-3_stats.csv'
    }

    # 读取并合并所有 CSV 文件
    for key, file in csv_files.items():
        if os.path.exists(file):  # 确保文件存在
            df = pd.read_csv(file)
            df.insert(0, "Experiment", key)  # 在第一列插入实验编号
            results_df = pd.concat([results_df, df], ignore_index=True)

    # 只保留小数点后三位
    results_df = results_df.round(3)

    # 保存合并后的数据到 CSV 文件
    # results_df.to_csv(output_file, index=False)
    # print(f"合并结果已保存到 {output_file}")
    results_df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"合并结果已保存到 {output_file}")

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
    # read_sum_table(output_file="logs-outputs_T5/order_1/outputs/1-dbpedia/adapter_lora/merged_results.xlsx")
    # # order1 - 1-dbpedia  --1 --origin model full perturb
    # loss_landscape_lora_file = "initial_model/t5-small/lossLandscape_1.h5"
    # save_pth_lora =            'initial_model/t5-small/lossLandscape_1_t2'
    # plot_loss_landscape(loss_landscape_lora_file, title="Dispeturb Loss Landscape",savepth = save_pth_lora)

    # order1 - 1-dbpedia --2 --origin model with LoRA no train -- only W
    # loss_landscape_lora_file = "initial_model/t5-small/lossLandscape_2-1.h5"
    # save_pth_lora =            'initial_model/t5-small/lossLandscape_2-1_t3'
    # plot_loss_landscape(loss_landscape_lora_file, title="Dispeturb Loss Landscape",savepth = save_pth_lora)

    # # order1 - 1-dbpedia --2 --origin model with LoRA no train -- only AB
    # loss_landscape_lora_file = "initial_model/t5-small/lossLandscape_2-2.h5"
    # save_pth_lora =            'initial_model/t5-small/lossLandscape_2-2'
    # plot_loss_landscape(loss_landscape_lora_file, title="Dispeturb Loss Landscape",savepth = save_pth_lora)


    # # order1 - 1-dbpedia --2 --origin model with LoRA no train --  W -AB
    # loss_landscape_lora_file = "initial_model/t5-small/lossLandscape_2-3.h5"
    # save_pth_lora =            'initial_model/t5-small/lossLandscape_2-3'
    # plot_loss_landscape(loss_landscape_lora_file, title="Dispeturb Loss Landscape",savepth = save_pth_lora)




    # # order1 - 1-dbpedia --3 --origin model with LoRA after train -- only W
    loss_landscape_lora_file = "logs-outputs_T5/order_1/outputs/1-dbpedia/adapter_lora/lossLandscape_3-1_t1.h5"
    save_pth_lora =            'logs-outputs_T5/order_1/outputs/1-dbpedia/adapter_lora/lossLandscape_3-1_t1'
    plot_loss_landscape(loss_landscape_lora_file, title="Dispeturb Loss Landscape",savepth = save_pth_lora)

    # # order1 - 1-dbpedia --3 --origin model with LoRA after train -- only AB
                                
    # loss_landscape_lora_file = "logs-outputs_T5/order_1/outputs/1-dbpedia/adapter_lora/lossLandscape_3-2.h5"
    # save_pth_lora =            'logs-outputs_T5/order_1/outputs/1-dbpedia/adapter_lora/lossLandscape_3-2'
    # plot_loss_landscape(loss_landscape_lora_file, title="Dispeturb Loss Landscape",savepth = save_pth_lora)


    # # order1 - 1-dbpedia --3 --origin model with LoRA after train --  W -AB
    # loss_landscape_lora_file = "logs-outputs_T5/order_1/outputs/1-dbpedia/adapter_lora/lossLandscape_3-3.h5"
    # save_pth_lora =            'logs-outputs_T5/order_1/outputs/1-dbpedia/adapter_lora/lossLandscape_3-3'
    # plot_loss_landscape(loss_landscape_lora_file, title="Dispeturb Loss Landscape",savepth = save_pth_lora)
    
    
    