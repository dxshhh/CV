各文件功能说明：

repr2.py:
训练模型的主要代码
会用到repr2loader.py中的数据读取相关函数

repr2loader.py:
读取训练数据相关代码
把训练数据下载到data/disk文件夹下并读取
也包含读取测试匹配数据(verTest)的代码

repr2-last.pkl:
训练好的模型

reg2.py:
用来测试相机角度数据的代码
会用到reg2loader.py

reg2loader.py: 
读取测试相机角度数据(regTest)的代码

tsne.py:
自己实现的tsne算法
读取两个文件中(都硬编码成了in.txt)的特征向量
计算两两的l2距离作为新特征向量
再做tsne
结果输出的到out_data.txt

mtsne.py:
调用现成的tsne算法
对tsne.in里的向量做tsne

hook_tsne.py:
计算tsne.in0文件中指定的图片的特征向量
把结果输出到tsne.in里

cropall.py:
对训练数据进行裁减预处理
会用到croploader.py

green.py:
用来删除树叶遮挡严重的训练数据

drawpoints.py:
用来绘制tsne结果的代码

draw_data.py:
存储drawpoints.py所需要的数据

ctc.cpp:
用来计算两组特征向量的距离的代码

cube:
绘制正方体相关代码
使用OpenGL以及开源库lodepng
会把生成的图片存在data目录下

若要运行训练代码，需在data目录下存在list.txt文件，每行内容为训练数据包编号，如0002
训练数据从https://console.cloud.google.com/storage/browser/streetview_image_pose_3d下载

