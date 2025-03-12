import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # 添加三维绘图库

def plot_hessian_stats(h5_file, title="Loss Landscape",savepth = None):
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

def plot_hessian_stats(h5_file, title="Hessian Statistics", savepth=None):
    """
    加载 HDF5 文件中的 Hessian 统计数据，并绘制特征值分布直方图，并在图上标注统计信息。
    """
    # 读取数据
    with h5py.File(h5_file, 'r') as f:
        if "dominant_eigs" not in f or "min" not in f or "max" not in f or "median" not in f or "mean" not in f or "std" not in f:
            raise KeyError("HDF5 文件缺少必要数据集，请检查写入逻辑。")
        
        dominant_eigs = f['dominant_eigs'][:]
        min_eig = f['min'][()]
        max_eig = f['max'][()]
        median_eig = f['median'][()]
        mean_eig = f['mean'][()]
        std_eig = f['std'][()]

    # 打印统计信息
    print("Hessian Statistics:")
    print(f"Min Eigenvalue: {min_eig:.4f}")
    print(f"Max Eigenvalue: {max_eig:.4f}")
    print(f"Median Eigenvalue: {median_eig:.4f}")
    print(f"Mean Eigenvalue: {mean_eig:.4f}")
    print(f"Standard Deviation: {std_eig:.4f}")

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(dominant_eigs, bins=30, alpha=0.7, edgecolor="black", density=True, label="Eigenvalue Distribution")

    # 竖线标记
    plt.axvline(min_eig, color='blue', linestyle='--', linewidth=1.5, label=f"Min: {min_eig:.4f}")
    plt.axvline(max_eig, color='purple', linestyle='--', linewidth=1.5, label=f"Max: {max_eig:.4f}")
    plt.axvline(median_eig, color='red', linestyle='-', linewidth=2, label=f"Median: {median_eig:.4f}")
    plt.axvline(mean_eig, color='green', linestyle='-.', linewidth=2, label=f"Mean: {mean_eig:.4f}")

    # 标记标准差区域
    plt.fill_betweenx(
        y=[0, plt.ylim()[1]], 
        x1=mean_eig - std_eig, 
        x2=mean_eig + std_eig, 
        color='gray', alpha=0.2, label=f"1 Std Range: [{mean_eig - std_eig:.4f}, {mean_eig + std_eig:.4f}]"
    )

    # 添加标题和标签
    plt.title(title, fontsize=14)
    plt.xlabel("Dominant Eigenvalue", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    
    # 保存图片
    if savepth:
        plt.savefig(f"{savepth}_hessian_3.png", dpi=300)
    
    plt.show()

def plot_hessian_power_iteration(h5_file, title="Hessian Power Iteration Statistics", savepth=None):
    """
    加载 HDF5 文件中的 Hessian 最大特征值统计数据，并绘制直方图，标注统计信息。
    """
    # 读取数据
    with h5py.File(h5_file, 'r') as f:
        if "dominant_eigs" not in f or "min" not in f or "max" not in f or "mean" not in f or "std" not in f:
            raise KeyError("HDF5 文件缺少必要数据集，请检查写入逻辑。")
        
        dominant_eigs = f['dominant_eigs'][:]
        min_eig = f['min'][()]
        max_eig = f['max'][()]
        mean_eig = f['mean'][()]
        std_eig = f['std'][()]

    # 打印统计信息
    print("\nHessian Power Iteration Statistics:")
    print(f"Min Eigenvalue: {min_eig:.4f}")
    print(f"Max Eigenvalue: {max_eig:.4f}")
    print(f"Mean Eigenvalue: {mean_eig:.4f}")
    print(f"Standard Deviation: {std_eig:.4f}\n")

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(dominant_eigs, bins=30, alpha=0.7, edgecolor="black", density=True, label="Eigenvalue Distribution")

    # 竖线标记
    plt.axvline(min_eig, color='blue', linestyle='--', linewidth=1.5, label=f"Min: {min_eig:.4f}")
    plt.axvline(max_eig, color='purple', linestyle='--', linewidth=1.5, label=f"Max: {max_eig:.4f}")
    plt.axvline(mean_eig, color='green', linestyle='-.', linewidth=2, label=f"Mean: {mean_eig:.4f}")

    # 标记标准差区域
    plt.fill_betweenx(
        y=[0, plt.ylim()[1]], 
        x1=mean_eig - std_eig, 
        x2=mean_eig + std_eig, 
        color='gray', alpha=0.2, label=f"1 Std Range: [{mean_eig - std_eig:.4f}, {mean_eig + std_eig:.4f}]"
    )

    # 添加标题和标签
    plt.title(title, fontsize=14)
    plt.xlabel("Dominant Eigenvalue", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    
    # 保存图片
    if savepth:
        plt.savefig(f"{savepth}_hessian_power_iteration.png", dpi=300)
    
    plt.show()



if __name__ == "__main__":
    # hessian_full_file = "path/to/hessian_full_model_matrix.h5"
    # hessian_lora_file = "path/to/hessian_lora_only_matrix.h5"

    # 修改以下文件路径为实际保存的文件路径
    # loss_landscape_full_file = "logs_and_outputs/order_1/outputs/1-dbpedia/full_loss_full_model_landscape.h5"
    # loss_landscape_lora_file = "logs_and_outputs/order_1/outputs/1-dbpedia/full_loss_lora_only_landscape.h5"
    # # 绘制 Loss Landscape
    # save_pth_full = 'logs_and_outputs/order_1/outputs/1-dbpedia/full_loss_full_model_landscape'
    # save_pth_lora = 'logs_and_outputs/order_1/outputs/1-dbpedia/full_loss_lora_model_landscape'


    # # 修改以下文件路径为实际保存的文件路径
    # loss_landscape_full_file = "logs_and_outputs_lora/order_1/outputs/2-amazon/full_loss_full_model_landscape.h5"
    # loss_landscape_lora_file = "logs_and_outputs_lora/order_1/outputs/2-amazon/full_loss_lora_only_landscape.h5"
    # 绘制 Loss Landscape
    # save_p+th_full = 'logs_and_outputs_lora/order_1/outputs/2-amazon/full_loss_full_model_landscape'
    # save_pth_lora = 'logs_and_outputs_lora/order_1/outputs/2-amazon/full_loss_lora_model_landscape'


    # # 修改以下文件路径为实际保存的文件路径
    # loss_landscape_full_file = "logs_and_outputs_lora/order_1/outputs/3-yahoo/full_loss_full_model_landscape.h5"
    # loss_landscape_lora_file = "logs_and_outputs_lora/order_1/outputs/3-yahoo/full_loss_lora_only_landscape.h5"
    # # 绘制 Loss Landscape
    # save_pth_full = 'logs_and_outputs_lora/order_1/outputs/3-yahoo/full_loss_full_model_landscape'
    # save_pth_lora = 'logs_and_outputs_lora/order_1/outputs/3-yahoo/full_loss_lora_model_landscape'


    # 修改以下文件路径为实际保存的文件路径
    # loss_landscape_full_file = "logs_and_outputs_lora/order_1/outputs/4-agnews/full_loss_full_model_landscape.h5"
    # loss_landscape_lora_file = "logs_and_outputs_lora/order_1/outputs/4-agnews/full_loss_lora_only_landscape.h5"
    # # 绘制 Loss Landscape
    # save_pth_full = 'logs_and_outputs_lora/order_1/outputs/4-agnews/full_loss_full_model_landscape'
    # save_pth_lora = 'logs_and_outputs_lora/order_1/outputs/4-agnews/full_loss_lora_model_landscape'


    # plot_hessian_stats(loss_landscape_full_file, title="Model Loss Landscape",savepth = save_pth_full)
    # plot_hessian_stats(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    
    # 绘制 Hessian 统计数据
    # plot_hessian_stats(hessian_full_file, title="Full Model Hessian Statistics")
    # plot_hessian_stats(hessian_lora_file, title="LoRA Only Hessian Statistics")



    # --------- lora  method -10

    # Eval 1-1 lora  T5-small  loss landscape  order1
    #  order1 - 1-dbpedia
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_1/outputs/1-dbpedia/adapter_T5small_lora-10/hessian_lora_only-predictDataset.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_1/outputs/1-dbpedia/adapter_T5small_lora-10/Hessian_T5small-10_lora'
    # plot_hessian_stats(loss_landscape_lora_file, title="Hessian matrix",savepth = save_pth_lora)
    # #  order1 - 2-amazon
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_1/outputs/2-amazon/adapter_T5small_lora-10/hessian_lora_only-predictDataset.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_1/outputs/2-amazon/adapter_T5small_lora-10/Hessian_T5small-10_lora'
    # plot_hessian_stats(loss_landscape_lora_file, title="Hessian matrix",savepth = save_pth_lora)
    # #  order1 - 3-yahoo
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_1/outputs/3-yahoo/adapter_T5small_lora-10/hessian_lora_only-predictDataset.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_1/outputs/3-yahoo/adapter_T5small_lora-10/Hessian_T5small-10_lora'
    # plot_hessian_stats(loss_landscape_lora_file, title="Hessian matrix",savepth = save_pth_lora)
    # #  order1 - 4-agnews
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_1/outputs/4-agnews/adapter_T5small_lora-10/hessian_lora_only-predictDataset.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_1/outputs/4-agnews/adapter_T5small_lora-10/Hessian_T5small-10_lora'
    # plot_hessian_stats(loss_landscape_lora_file, title="Hessian matrix",savepth = save_pth_lora)

    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_1/outputs/4-agnews/adapter_T5small_lora-10/hessian_lora_power_iteration.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_1/outputs/4-agnews/adapter_T5small_lora-10/Hessian_T5small-10_lora'
    # plot_hessian_power_iteration(loss_landscape_lora_file, title="Hessian matrix",savepth = save_pth_lora)




    #------------------Eval 1-1 lora  T5-small  loss landscape  order2------------------------
    # #  order1 - 1-yahoo
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_2/outputs/1-dbpedia/adapter_T5small_lora-10/hessian_lora_only-predictDataset_lanczos.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_2/outputs/1-dbpedia/adapter_T5small_lora-10/hessian_T5small-10_lora'
    # plot_hessian_stats(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 2-amazon
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_2/outputs/2-amazon/adapter_T5small_lora-10/hessian_lora_only-predictDataset_lanczos.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_2/outputs/2-amazon/adapter_T5small_lora-10/hessian_T5small-10_lora'
    # plot_hessian_stats(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 3-agnews
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_2/outputs/3-agnews/adapter_T5small_lora-10/hessian_lora_only-predictDataset_lanczos.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_2/outputs/3-agnews/adapter_T5small_lora-10/hessian_T5small-10_lora'
    # plot_hessian_stats(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 4-dbpedia
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_2/outputs/4-yahoo/adapter_T5small_lora-10/hessian_lora_only-predictDataset_lanczos.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_2/outputs/4-yahoo/adapter_T5small_lora-10/hessian_T5small-10_lora'
    # plot_hessian_stats(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    


    #------------------Eval 1-1 lora  T5-small  loss landscape  order3------------------------
    # #  order1 - 1-yahoo
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_3/outputs/1-yahoo/adapter_T5small_lora-10/hessian_lora_only-predictDataset_lanczos.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_3/outputs/1-yahoo/adapter_T5small_lora-10/hessian_T5small-10_lora'
    # plot_hessian_stats(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 2-amazon
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_3/outputs/2-amazon/adapter_T5small_lora-10/hessian_lora_only-predictDataset_lanczos.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_3/outputs/2-amazon/adapter_T5small_lora-10/hessian_T5small-10_lora'
    # plot_hessian_stats(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 3-agnews
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_3/outputs/3-agnews/adapter_T5small_lora-10/hessian_lora_only-predictDataset_lanczos.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_3/outputs/3-agnews/adapter_T5small_lora-10/hessian_T5small-10_lora'
    # plot_hessian_stats(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    # #  order1 - 4-dbpedia
    # loss_landscape_lora_file = "logs_and_outputs_T5small_lora/order_3/outputs/4-dbpedia/adapter_T5small_lora-10/hessian_lora_only-predictDataset_lanczos.h5"
    # save_pth_lora =            'logs_and_outputs_T5small_lora/order_3/outputs/4-dbpedia/adapter_T5small_lora-10/hessian_T5small-10_lora'
    # plot_hessian_stats(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)


    #------------------Eval 1-1 Nlora   only task T5-small  loss landscape  order1------------------------
    #order1 - 1-dbpedia
    loss_landscape_lora_file = "logs_and_outputs_T5small_Nlora/order_1/outputs/1-dbpedia/adapter_T5small_Nlora_onlytasknew-10/hessian_Nlora_only-predictDataset.h5"
    save_pth_lora =            'logs_and_outputs_T5small_Nlora/order_1/outputs/1-dbpedia/adapter_T5small_Nlora_onlytasknew-10/hessian_Nlora_onlytask'
    plot_hessian_stats(loss_landscape_lora_file, title="NLoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    #  order1 - 2-amazon
    loss_landscape_lora_file = "logs_and_outputs_T5small_Nlora/order_1/outputs/2-amazon/adapter_T5small_Nlora_onlytasknew-10/hessian_Nlora_only-predictDataset.h5"
    save_pth_lora =            'logs_and_outputs_T5small_Nlora/order_1/outputs/2-amazon/adapter_T5small_Nlora_onlytasknew-10/hessian_Nlora_onlytask'
    plot_hessian_stats(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    #  order1 - 3-yahoo
    loss_landscape_lora_file = "logs_and_outputs_T5small_Nlora/order_1/outputs/3-yahoo/adapter_T5small_Nlora_onlytasknew-10/hessian_Nlora_only-predictDataset.h5"
    save_pth_lora =            'logs_and_outputs_T5small_Nlora/order_1/outputs/3-yahoo/adapter_T5small_Nlora_onlytasknew-10/hessian_Nlora_onlytask'
    plot_hessian_stats(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    #  order1 - 4-agnews
    loss_landscape_lora_file = "logs_and_outputs_T5small_Nlora/order_1/outputs/4-agnews/adapter_T5small_Nlora_onlytasknew-10/hessian_Nlora_only-predictDataset.h5"
    save_pth_lora =            'logs_and_outputs_T5small_Nlora/order_1/outputs/4-agnews/adapter_T5small_Nlora_onlytasknew-10/hessian_Nlora_onlytask'
    plot_hessian_stats(loss_landscape_lora_file, title="LoRA Dispeturb Loss Landscape",savepth = save_pth_lora)
    