# panda_go_grasp
使用panda机械臂接收grasp pose，执行抓取和一些其他操作

接收以**桌面参考系为父坐标系**的GraspConfigList类型数据

读取列表的第一个GraspConfig类型数据（最优grasp config）





panda夹爪的坐标系很奇怪，

- Panda_link8 属于panda arm规划组，是最后一个关节的坐标系

- panda_hand 坐标系只是和panda_link8 原点在一起，但是panda_hand认为不属于panda arm规划组，而是位于夹爪panda_hand规划组上，坐标系固连在hand上

- panda_EE的坐标系原点位于夹爪的指尖中心



<img src="/home/wgk/.config/Typora/typora-user-images/image-20210323222536538.png" alt="image-20210323222536538" style="zoom: 40%;" /><img src="/home/wgk/.config/Typora/typora-user-images/image-20210323222712987.png" alt="image-20210323222712987" style="zoom: 40%;" /><img src="/home/wgk/.config/Typora/typora-user-images/image-20210323222920761.png" alt="image-20210323222920761" style="zoom: 40%;" /><img src="/home/wgk/.config/Typora/typora-user-images/image-20210323223249480.png" alt="image-20210323223249480" style="zoom:33%;" />



图中RGB轴与`GraspConfig`消息类型中定义的三个抓取轴的对应关系为：

- red          对应axis撸轴     对应旋转矩阵的x轴   
- green       对应binormal合并轴        对应旋转矩阵的y轴   
- bule    对应approach接近轴        对应旋转矩阵的z轴   



gpd发送的位置姿态一般是 panda_hand或者panda_EE 的坐标系位置；但是为了方便控制机械臂，一般我们是控制panda_arm 规划组就行了，所以让机械臂轨迹规划之前

首先需要计算出gpd姿态（panda_hand或者panda_EE姿态）对应的panda_link8姿态，然后在panda_arm 规划组执行对panda_link8的轨迹并执行即可





流程是，先构建panda_hand在marker下的4\*4变换矩阵，再通过tf获取base2gripper与gripper2link8  然后使用矩阵点乘来最终计算出  base2grasplink8的4\*4变换矩阵，然后再通过四元数来执行轨迹



因为中间除了很多计算的bug，也没搞明明白，我在想不能直接用tf读取的四元数来对抓取的姿态进行变换？这样不是好很多么？何必变成矩阵再变回来？

https://blog.csdn.net/candycat1992/article/details/41254799

![image-20210323231056595](/home/wgk/.config/Typora/typora-user-images/image-20210323231056595.png)







关键是，goal是用坐标系向量表示的，最后link8姿态是用四元数表示的，避免不了将goal转换到四元数，如果从刚开始就发送的是四元数呢？goal相对于marker的四元数呢？这个pass 没有办法做到这点

将最初坐标系形式的gaol 转换为相当于marke的四元数形式，之后的旋转都是四元数形式







还有一点是，如何从四元数判断是否过于绕？（检查goal2base四元数的某个轴与，可能还是要利用四元数构建rot矩阵）以及如何在四元数层面上，对某个轴进行翻转，这个问题需要考虑



**中间最好不出现欧拉角**

将中间坐标系显示出来





```python
def lookupTransform(self, target_frame, source_frame, time):
```

返回的变换是从`targe_frame`到`source_frame`, 即，以`targe_frame`为父坐标系









四元数可以表示为$q=x\vec{i}+y\vec{j}+z\vec{k}+w$

以下都以$q=(x,y,z,w)$的形式表示一个四元数，也可以表示为$q=(\vec{v},w )$



### 四元数乘四元数变换，就是连续的旋转如何用四元数表示？

复合连续变换，如果是相对运动坐标系的变换，同样只需要像复合句真那样，友乘新的四元数即可



四元数逆运算

四元数从坐标系A到B的旋转四元数是否和从B到A的旋转四元数是逆运算？



### 四元数乘四元数的法则好像也不一样

对于四元数$q_1=(\vec{v_1},w_1),q_2=(\vec{v_2},w_2)$来讲，两者的相乘，满足
$$
q=q_1\cdot q_2 =(\vec{v_1}\times\vec{v_2}+w_1\vec{v_2}+w_2\vec{v_1},w_1w_2-\vec{v_1}\cdot\vec{v_2})
$$
这种操作还是比较麻烦的





### 四元数直接对向量进行旋转，求解新的向量，

假设向量另一点为$P=(x_p,y_p,z_p)$，将其拓展到四元数空间中，对应的四元数$p=(x_p,y_p,z_p,0)=(P,0)$，对该向量施加以四元数$q$表示的旋转，那么旋转后的点$P'$对应的四元数$p'=(P',0)$可以使用如下公式计算
$$
p'=q\cdot p\cdot q^{-1}
$$
注意

- 在运算前需要将$p$拓展为四元数的形式$p=(x_p,y_p,z_p,0)$，因此就是三个四元数相乘，当然结果$p'$的实部仍然会是0
- $q^{-1}=q^*=(-\vec{v},w)$     因为这里的四元数都是单位四元数
- 对向量的旋转需要同时有两个四元数参与，即  $q,q^*$







有哪些python库可以直接使用的？着重tf，numpy等



旋转矩阵和四元数的互换  python

欧拉角和四元数的互换 python







一个问题：

