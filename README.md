# 6DOF_Gen_ros
a simple demo for use 6DOF_Gen_ros

# 软件安装
## CloudCompare - 点云查看及编辑
    - https://www.cloudcompare.org/release/index.html#CloudCompare

## Colmap - 三维重建
*  介绍：Colmap是一个集成的[SfM](https://www.baidu.com/s?sa=re_dqa_generate&wd=SfM&rsv_pq=b9d7c3430000bfd9&oq=colmap%E5%8F%AF%E4%BB%A5%E5%9C%A8linux%E5%BD%93%E4%B8%AD%E8%BF%9B%E8%A1%8C%E5%AE%89%E8%A3%85%E5%90%97%3F&rsv_t=99f4q6VriUjZi07fCArAuPR8VXvGH+HJbODKFXpp0g0oA8cx8coRf1RKEOuXGLUwz5THPQ&tn=15007414_9_dg&ie=utf-8)和[MVS](https://www.baidu.com/s?sa=re_dqa_generate&wd=MVS&rsv_pq=b9d7c3430000bfd9&oq=colmap%E5%8F%AF%E4%BB%A5%E5%9C%A8linux%E5%BD%93%E4%B8%AD%E8%BF%9B%E8%A1%8C%E5%AE%89%E8%A3%85%E5%90%97%3F&rsv_t=99f4q6VriUjZi07fCArAuPR8VXvGH+HJbODKFXpp0g0oA8cx8coRf1RKEOuXGLUwz5THPQ&tn=15007414_9_dg&ie=utf-8)工具，它可以从多视图图像集中自动计算相机位姿并构建高精度的三维点云。在Linux环境下安装Colmap
*  如果你的系统是ubuntu20.04
*  默认为CPU版本
```bash
# 安装
sudo apt-get install colmap

# 测试
colmap gui
```
* 带cuda的版本需要进行源码编译，具体过程如下
```bash
git clone https://github.com/colmap/colmap.git

mkdir build 
cd build

# 构建文件
## （1）编译
cmake .. -GNinja \  
    -D CMAKE_CUDA_COMPILER="/usr/local/cuda-12.2/bin/nvcc" \  
    -D CMAKE_CUDA_ARCHITECTURES='89'

## （2）编译
cmake .. -GNinja -D CMAKE_CUDA_COMPILER="/usr/local/cuda-12.2/bin/nvcc" -D CMAKE_CUDA_ARCHITECTURES='89'

# 编译
ninja
sudo ninja install

# 查看一下当前的cuda路径对不对
echo $LD_LIBRARY_PATH
```

*  构建完成后如下
```bash
lab@lab:~/GenDexGrasp/Gendexgrasp_ros/ros_vision/6DOF_Gen_ros/colmap/build$ sudo ninja install
[0/1] Install the project...
-- Install configuration: "Release"
-- Up-to-date: /usr/local/share/applications/COLMAP.desktop
-- Up-to-date: /usr/local/lib/libcolmap_controllers.a
-- Up-to-date: /usr/local/lib/libcolmap_estimators.a
-- Up-to-date: /usr/local/lib/libcolmap_exe.a
-- Up-to-date: /usr/local/lib/libcolmap_feature_types.a
-- Up-to-date: /usr/local/lib/libcolmap_feature.a
-- Up-to-date: /usr/local/lib/libcolmap_geometry.a
-- Up-to-date: /usr/local/lib/libcolmap_image.a
```