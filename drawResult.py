import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def readxyz(inputfile):
    xlist = []
    ylist = []
    zlist = []
    try:
        with open(inputfile, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    xlist.append(float(parts[0]))
                    ylist.append(float(parts[1]))
                    zlist.append(float(parts[2]))
    except FileNotFoundError:
        print(f"Warning: File not found: {inputfile}")
    return xlist, ylist, zlist

if __name__ == '__main__':
    # === 你的原始路径配置 ===
    input_name = './build/01' 
    full_path = input_name + '.txt'
    control_name = input_name + "_controls.txt"
    spline_name = input_name + "_spline.txt"
    
    # 1. 读取所有数据
    x, y, z = readxyz(full_path)               # 原始点
    cx, cy, cz = readxyz(control_name)         # 控制点
    sx, sy, sz = readxyz(spline_name)          # 拟合点

    # ==========================================
    # 🌟 新增功能：在代码里手动输入想要渲染的点
    # ==========================================
    # 请在下面这三个列表里填入你想要显示的蓝点的 X, Y, Z 坐标
    # 注意：三个列表里的元素数量必须一致
    custom_x = [21.616,22.6052]  
    custom_y = [-6.9808,-7.0563]  
    custom_z = [-0.9326,-0.9814]  


    # ==========================================

    # 2. 创建画布
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 3. 绘制原始采样点 (红色圆点)
    if x:
        ax.scatter(x, y, z, c='r', marker='o', s=5, label="Sample Points", alpha=0.2)
    else:
        print("Error: Original sample points empty or file not found!")

    # 4. 绘制代码中手动加入的点 (蓝色点)
    if custom_x:
        # c='blue' 设置蓝色，marker='*' 设置为星号比较醒目，s=50 设置点的大小，zorder 设大一点保证不被遮挡
        ax.scatter(custom_x, custom_y, custom_z, c='blue', marker='*', s=50, label="Custom Points", zorder=10)

    # 5. 绘制拟合后的 B 样条曲面 (绿色点)
    # 注意：把你原来的 `if not sx:` 改成了 `if sx:`，否则读到数据反而不会画图
    if sx:
        ax.scatter(sx, sy, sz, c='lime', marker='.', s=5, label="Fitted Spline", zorder=5)
    else:
        print(f"Warning: No spline points found at {spline_name}")

    if x: # 确保原始点云非空
        # 1. 算出 XYZ 各自的最大跨度，取其中最大的一个
        max_range = max(max(x)-min(x), max(y)-min(y), max(z)-min(z)) / 2.0
        
        # 2. 算出 XYZ 各自的中心点
        mid_x = (max(x) + min(x)) / 2.0
        mid_y = (max(y) + min(y)) / 2.0
        mid_z = (max(z) + min(z)) / 2.0
        
        # 3. 把三个轴的刻度范围强行固定为：各自中心点 ± 最大跨度
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
    # 6. 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('B-Spline Fitting Result')
    ax.legend()
    
    # 自动调整视角
    ax.view_init(elev=30, azim=-60)
    
    plt.savefig(input_name + "_3d_result.png")
    plt.show()
