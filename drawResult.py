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

    # 2. 创建画布
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 3. 绘制原始采样点 (红色圆点)
    if x:
        ax.scatter(x, y, z, c='r', marker='o', s=10, label="Sample Points", alpha=0.5)
    else:
        print("Error: Original sample points empty or file not found!")



    # 5. 绘制拟合后的 B 样条曲面 (绿色点)
    if sx:
        ax.scatter(sx, sy, sz, c='lime', marker='.', s=20, label="Fitted Spline", zorder=5)
    else:
        print(f"Warning: No spline points found at {spline_name}")

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
