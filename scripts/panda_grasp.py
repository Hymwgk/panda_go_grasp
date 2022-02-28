#!/usr/bin/env python
#coding=utf-8
"""
    moveit_ik_demo.py - Version 0.1 2014-01-14
    使得机械臂，先在初始状态，然后移动一下机械臂，然后再回到初始状态，停止
    Use inverse kinemtatics to move the end effector to a specified pose
    
    Created for the Pi Robot Project: http://www.pirobot.org
    Copyleft (c) 2014 Patrick Goebel.  All lefts reserved.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.5
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details at:
    
    http://www.gnu.org/licenses/gpl.html
"""

import rospy, sys
import moveit_commander
import tf
import argparse
import math
import numpy as np
from math import pi
import time
import copy
from moveit_msgs.msg import RobotTrajectory,DisplayTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from geometry_msgs.msg import PoseStamped, Pose
from tf.transformations import euler_from_quaternion, quaternion_from_euler,quaternion_multiply,quaternion_from_matrix,quaternion_matrix
from autolab_core import RigidTransform,transformations
from pyquaternion import Quaternion
try:
    from gpd_grasp_msgs.msg import GraspConfig,GraspConfigList
except ImportError:
    print("Please install grasp msgs from https://github.com/TAMS-Group/gpd_grasp_msgs in your ROS workspace")
    exit()

#解析命令行参数
parser = argparse.ArgumentParser(description='Panda go grasp')
parser.add_argument('--test',action='store_true',default=True)  #设置同时处理几个场景
parameters = parser.parse_args()


class MoveItDemo:
    def __init__(self):
        #初始化moveit的 API接口
        moveit_commander.roscpp_initialize(sys.argv)
        #初始化ros节点 名为panda_grasp
        rospy.init_node('panda_grasp', anonymous=True)
        rospy.set_param("/robot_state", "Initializing")
        rospy.loginfo("Robot  initializing")

        #构建tf发布器
        self.tf_broadcaster=tf.TransformBroadcaster()

        self.grasp_config=GraspConfig()

        #创建多用途的TF监听器
        self.tf_listener = tf.TransformListener()
        #变换关系正确读取的标志位
        get_transform=False
        #等待并获取正确的tf变换关系
        while not get_transform:
            try:
                if parameters.test:
                    get_transform = True
                    rospy.loginfo("Test mode")
                else:
                    #尝试查看机器人基座base与桌面标签之间的转换
                    base2marker_trans, self.base2marker_rot = self.tf_listener.lookupTransform('/panda_link0', '/ar_marker_6', rospy.Time(0))
                    euler = tf.transformations.euler_from_quaternion(self.base2marker_rot)
                    self.base2marker = tf.transformations.compose_matrix(translate = base2marker_trans, angles = euler)
                    #将trans转换成为ndarry
                    self.base2marker_trans=np.array(base2marker_trans)
                    #查看gripper到link8之间的变换
                    gripper2link8_trans, self.gripper2link8_rot = self.tf_listener.lookupTransform( '/panda_EE', '/panda_link8',rospy.Time(0))
                    euler = tf.transformations.euler_from_quaternion(self.gripper2link8_rot)
                    self.gripper2link8 = tf.transformations.compose_matrix(translate = gripper2link8_trans, angles = euler)
                    self.gripper2link8_trans=np.array(gripper2link8_trans)
                    #查看base到panda_link8的变换，此时就是查询gripper的初始姿态
                    trans, rot = self.tf_listener.lookupTransform( '/panda_link0', '/panda_link8',rospy.Time(0))
                    euler = tf.transformations.euler_from_quaternion(rot)
                    self.base2Initial_link8 = tf.transformations.compose_matrix(translate = trans, angles = euler)

                    get_transform = True
                    rospy.loginfo("got transform complete")
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.loginfo("got transform failed")
                rospy.sleep(0.5)
                continue


        # 初始化场景对象
        self.scene = moveit_commander.PlanningSceneInterface()
        #为场景添加桌子，防止机械臂碰撞桌面
        self.add_table()
        rospy.sleep(2)
        # 创建机械臂规划组对象
        self.panda_arm = moveit_commander.MoveGroupCommander('panda_arm')
        #创建机械手规划对象
        self.panda_hand=moveit_commander.MoveGroupCommander('hand')
        #通过此发布器发布规划的轨迹
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               DisplayTrajectory,
                                               queue_size=20)
        # 获取末端执行器名称
        self.end_effector_link = self.panda_arm.get_end_effector_link()
        print("检测到末端执行器{}".format(self.end_effector_link))
                        
        # 创建机械臂父坐标系名称字符
        reference_frame = 'panda_link0'
        # 设置父坐标系名称
        #self.panda_arm.set_pose_reference_frame(reference_frame)

        # 设置允许机械臂末位姿的错误余量
        self.panda_arm.set_goal_position_tolerance(0.01)#1cm
        self.panda_arm.set_goal_orientation_tolerance(0.05)#

        #不允许规划失败重规划,规划时间只允许5秒钟,否则很浪费时间
        self.panda_arm.allow_replanning(False)
        self.panda_arm.set_planning_time(5)
        
        # 设置panda的home姿态
        Home_positions = [0.04, -0.70, 0.18, -2.80,  0.19, 2.13, 0.92]
        #移动到home
        self.move_to_home_joint(self.panda_arm,self.Home_joints)
        #张开夹爪
        self.set_gripper(0.08)#张开8cm
        rospy.set_param("/robot_state", "at_home")
        rospy.loginfo("Robot  at home")

        ######################开始等待接收夹爪姿态#########################
        print("Waiting for gripper pose!")
        self.callback_done=False

        if parameters.test:#测试模式
            self.grasp_test()
        else:
            rospy.Subscriber('/detect_grasps/clustered_grasps', GraspConfigList, self.Callback,queue_size=1)

        #######################执行抓取####################################
        while not rospy.is_shutdown():
            #等待回调函数处理完
            if self.callback_done:
                self.callback_done=False
            else:
                rospy.sleep(0.5)
                continue


            #以当前姿态作为规划起始点
            self.panda_arm.set_start_state_to_current_state()  
            # 对末端执行器姿态设定目标姿态
            #self.panda_arm.set_pose_target(target_pose, 'left_gripper')
            
            # 规划轨迹
            #traj = self.panda_arm.plan(target_pose.pose)
            
            # 执行轨迹，运行到预抓取位置
            #self.panda_arm.execute(traj)
            #print(self.end_effector_link)


            print('Moving to pre_grasp_pose')
            #self.panda_arm.pick("test",self.grasp_config,plan_only = True)
            #traj=self.panda_arm.plan(self.pre_grasp_pose)
            #self.panda_arm.set_pose_target(self.pre_grasp_pose,end_effector_link="panda_EE")
            #traj=self.panda_arm.plan()

            #continue

            #success=self.panda_arm.execute(traj)

            #print(target_pose.pose)
            #设置规划
            #self.panda_arm.set_planning_time(5)
            success=self.panda_arm.go(self.pre_grasp_pose,wait=True)
            self.panda_arm.stop()
            self.panda_arm.clear_pose_targets()

            
            
            if not success:
                print('Failed to move to pre_grasp_pose!')
                continue
            
            print('Move to pre_grasp_pose succeed')
            #等待机械臂稳定
            rospy.sleep(1)
            #再设置当前姿态为起始姿态
            self.panda_arm.set_start_state_to_current_state()  
            #
            waypoints = []
            wpose=self.panda_arm.get_current_pose().pose
            #print("#####wpose.position")
            #print(wpose.position)
            #print("#####self.grasp_pose2")
            #print(self.grasp_pose.position)
            wpose.position.x=  self.grasp_pose.position.x
            wpose.position.y=  self.grasp_pose.position.y
            wpose.position.z=  self.grasp_pose.position.z

            waypoints.append(copy.deepcopy(wpose))
            #wpose = self.panda_arm.get_current_pose().pose
            #wpose.position.z -= scale * 0.1

            #规划从当前位姿，保持姿态，转移到目标夹爪姿态的路径
            (plan, fraction) = self.panda_arm.compute_cartesian_path(
                waypoints,   # waypoints to follow
                0.01,        # eef_step
                0.0)         # jump_threshold
             ##显示轨迹
            display_trajectory = DisplayTrajectory()
            display_trajectory.trajectory_start = self.panda_arm.get_current_state()
            display_trajectory.trajectory.append(plan)
            # Publish
            display_trajectory_publisher.publish(display_trajectory)

            #执行,并等待这个轨迹执行成功
            new_plan=self.scale_trajectory_speed(plan,0.3)
            self.panda_arm.execute(new_plan,wait=True)
            #self.panda_arm.shift_pose_target(2,0.05,"panda_link8")
            #self.panda_arm.go()


            #执行抓取
            rospy.sleep(2)
            print("Grasping")
            joint_goal = self.panda_hand.get_current_joint_values()
            joint_goal[0] = 0.015
            joint_goal[1] = 0.015
            #plan=self.panda_hand.plan(joint_goal)
            #new_plan=self.scale_trajectory_speed(plan,0.3)
            self.panda_hand.go(joint_goal,wait=True)
            self.panda_hand.stop()

            ####################抓取完后撤####################
            waypoints = []
            wpose=self.panda_arm.get_current_pose().pose
            
            wpose.position.x=  self.pre_grasp_pose.position.x
            wpose.position.y=  self.pre_grasp_pose.position.y
            wpose.position.z=  self.pre_grasp_pose.position.z

            waypoints.append(copy.deepcopy(wpose))
            
            #规划从当前位姿，保持姿态，转移到目标夹爪姿态的路径
            (plan, fraction) = self.panda_arm.compute_cartesian_path(
                waypoints,   # waypoints to follow
                0.01,        # eef_step
                0.0)         # jump_threshold

            #执行,并等待后撤成功
            new_plan=self.scale_trajectory_speed(plan,0.6)
            self.panda_arm.execute(new_plan,wait=True)
            """
            display_trajectory = DisplayTrajectory()
            display_trajectory.trajectory_start = self.panda_arm.get_current_state()
            display_trajectory.trajectory.append(plan)
            # Publish
            display_trajectory_publisher.publish(display_trajectory)
            """

            ######################暂时设置直接回到Home############################

            #self.panda_arm.remember_joint_values('resting', joint_positions)#存储当前状态为初始状态
            #self.start_state =self.panda_arm.get_current_pose(self.end_effector_link)
            
            # Set the arm's goal configuration to the be the joint positions
            self.panda_arm.set_joint_value_target(Home_positions)
                    
            # Plan and execute the motion，运动到Home位置
            self.panda_arm.go()
            self.panda_arm.stop()

            joint_goal = self.panda_hand.get_current_joint_values()
            joint_goal[0] = 0.04
            joint_goal[1] = 0.04
            self.panda_hand.go(joint_goal, wait=True)
            self.panda_hand.stop()

            print("Grasp done")

            rospy.sleep(5)


        # Shut down MoveIt cleanly
        moveit_commander.roscpp_shutdown()
        
        # Exit MoveIt
        moveit_commander.os._exit(0)

    def lookupTransform(self,tf_listener, target, source):
        tf_listener.waitForTransform(target, source, rospy.Time(), rospy.Duration(4.0)) #等待时间为4秒

        trans, rot = tf_listener.lookupTransform(target, source, rospy.Time())
        euler = tf.transformations.euler_from_quaternion(rot)

        source_target = tf.transformations.compose_matrix(translate = trans, angles = euler)
        return source_target
    def getTfFromMatrix(self,matrix):
        scale, shear, angles, trans, persp = tf.transformations.decompose_matrix(matrix)
        return trans, tf.transformations.quaternion_from_euler(*angles), angles


    def quater_multi_vec(self,quater,vec):
        quater_=tf.transformations.quaternion_inverse(quater)
        vec_quater=np.c_[vec,[0]]
        temp=quaternion_multiply(quater,vec_quater)
        temp=quaternion_multiply(temp,quater_)
        return temp[:3]

    def move_to_home_joint(self,group,Home_pose):
        #先从Initial 移动到HOME
        case  = self.planJointGoal(group,Home_pose)#返回真  就是找到轨迹    
        if case==2:
            print(" Home pose Trajectory found;  Move to home")
            group.go(wait=True)
        elif case==1:
            print(" Already in home position")

        else:
            raise SystemError(" Home pose  trajectory  not found")
        time.sleep(2)

    def planJointGoal(self,movegroup,joint_goal,lable='Next'):
        current_joint = movegroup.get_current_joint_values()
        dis_pose =np.linalg.norm(np.array(joint_goal)-np.array(current_joint))
        #print(current_joint)
        #print(joint_goal)
        if dis_pose<0.008:
            return 1 #已经到位
        else:
            movegroup.set_joint_value_target(joint_goal)
            plan = movegroup.plan()
            if not plan.joint_trajectory.points:
                return 0
            else:#执行规划
                return 2

    
    def Callback(self,data): 
        """根据接收的夹爪抓取姿态，计算预抓取夹爪的位置姿态
        接收的抓取默认以相机坐标系为参考系
        使用的抓取坐标系为典范抓取坐标系(抓取中心点位于两指中心)
        """

        #data是GraspConfigList,data.grasps是GraspConfig[]类型,
        #data.grasps[0]是list中第一个GraspConfig类型的数据，代表的最优的那个抓取配置
        self.grasp_config=data.grasps[0]
        #最终抓取姿态
        self.grasp_pose=Pose()
        #预抓取姿态
        self.pre_grasp_pose=Pose()

        #以下是读取grasp的pose，需要注意的是，此时pose的参考系是谁？是桌面标签参考坐标系，并不是panda_link0
        #读取grasp pose的三个方向向量，转换为ndarray形式
        approach=np.array([self.grasp_config.approach.x,\
            self.grasp_config.approach.y,self.grasp_config.approach.z])#接近轴
        binormal=np.array([self.grasp_config.binormal.x,\
            self.grasp_config.binormal.y,self.grasp_config.binormal.z])#合并轴
        axis=np.array([self.grasp_config.axis.x,\
            self.grasp_config.axis.y,self.grasp_config.axis.z])#
        #进行方向向量归一化
        approach=approach/np.linalg.norm(approach)
        binormal=binormal/np.linalg.norm(binormal)
        axis=axis/np.linalg.norm(axis)
        #读取典范抓取坐标系原点在参考系中的坐标（默认参考系为相机坐标系）
        ctg_trans=np.array([self.grasp_config.top.x,self.grasp_config.top.y,self.grasp_config.top.z])#[3,]
        ctg_rot=np.hstack([approach,binormal,axis]).reshape(3,3).T #[3,3]
        #设置后撤距离(m)
        dis =0.05

        #计算出目标典范抓取坐标系在基座base中的位置姿态
        btg_rot = self.btc_rot.dot(ctg_rot)#旋转矩阵[3,3]
        btg_quater=quaternion_from_matrix(btg_rot)#将姿态转换为四元数形式
        btg_trans = self.btc_rot.dot(ctg_trans.reshape(3,1))+self.btc_trans.reshape(3,1) #[3,1] 
        btg_trans=btg_trans.reshape(3,)#[3,]

        self.grasp_pose.position.x = btg_trans[0]
        self.grasp_pose.position.y = btg_trans[1]
        self.grasp_pose.position.z = btg_trans[2]
        self.grasp_pose.orientation.x = btg_quater[0]
        self.grasp_pose.orientation.y = btg_quater[1]
        self.grasp_pose.orientation.z = btg_quater[2]
        self.grasp_pose.orientation.w = btg_quater[3]

        #计算预抓取与后撤抓取坐标系在基座坐标系中的位置姿态
        self.pre_grasp_pose = copy.deepcopy(self.grasp_pose)
        #计算pre抓取在相机坐标系中的位置
        btp_trans = btg_trans - btg_rot[0]*dis #[3,]
        self.pre_grasp_pose.position.x = btp_trans[0]
        self.pre_grasp_pose.position.y = btp_trans[1]
        self.pre_grasp_pose.position.z = btp_trans[2]


        #发布目标抓取姿态在base坐标系的位置
        self.tf_broadcaster.sendTransform(
            btg_trans,
            btg_quater,
            rospy.Time.now(),
            "base2grasp",
            "panda_link0")        
        #发布预备抓取姿态在base坐标系的位置
        self.tf_broadcaster.sendTransform(
            btp_trans,
            btg_quater,#与抓取姿态相同
            rospy.Time.now(),
            "base2pre",
            "panda_link0")   

        #标志回调函数处理完毕
        self.callback_done=False 

    def grasp_test(self): 
        """机械臂抓取测试，给定一个目标抓取位姿
        (panda_EE坐标系在panda_link0坐标系下的位置姿态)
        """

        #最终抓取姿态
        self.grasp_pose=Pose()
        #预抓取姿态
        self.pre_grasp_pose=Pose()

        #设置后撤距离(m)
        dis =0.05

        #给定目标典范抓取坐标系在基座base中的位置姿态
        btg_quater=np.array([0.98609,0.16538,0.01226,-0.011129])#将姿态转换为四元数形式
        btg_trans =np.array([0.55608,-0.04333,0.072476])
        btg_rot = quaternion_matrix(btg_quater)

        self.grasp_pose.position.x = btg_trans[0]
        self.grasp_pose.position.y = btg_trans[1]
        self.grasp_pose.position.z = btg_trans[2]
        self.grasp_pose.orientation.x = btg_quater[0]
        self.grasp_pose.orientation.y = btg_quater[1]
        self.grasp_pose.orientation.z = btg_quater[2]
        self.grasp_pose.orientation.w = btg_quater[3]

        #计算预抓取与后撤抓取坐标系在基座坐标系中的位置姿态
        self.pre_grasp_pose = copy.deepcopy(self.grasp_pose)
        #计算pre抓取在相机坐标系中的位置
        btp_trans = btg_trans - btg_rot[0]*dis #[3,]
        self.pre_grasp_pose.position.x = btp_trans[0]
        self.pre_grasp_pose.position.y = btp_trans[1]
        self.pre_grasp_pose.position.z = btp_trans[2]


        #发布目标抓取姿态在base坐标系的位置
        self.tf_broadcaster.sendTransform(
            btg_trans,
            btg_quater,
            rospy.Time.now(),
            "base2grasp",
            "panda_link0")        
        #发布预备抓取姿态在base坐标系的位置
        self.tf_broadcaster.sendTransform(
            btp_trans,
            btg_quater,#与抓取姿态相同
            rospy.Time.now(),
            "base2pre",
            "panda_link0")   

        #标志回调函数处理完毕
        self.callback_done=True 


    def scale_trajectory_speed(self,traj,spd=0.1):
        new_traj = RobotTrajectory()
        new_traj = traj

        n_joints = len(traj.joint_trajectory.joint_names)
        n_points = len(traj.joint_trajectory.points)

        #spd = 3.0

        points = list(traj.joint_trajectory.points)

        for i in range(n_points):
            point = JointTrajectoryPoint()
            point.time_from_start = traj.joint_trajectory.points[i].time_from_start / spd
            point.velocities = list(traj.joint_trajectory.points[i].velocities)
            point.accelerations = list(traj.joint_trajectory.points[i].accelerations)
            point.positions = traj.joint_trajectory.points[i].positions

            for j in range(n_joints):
                point.velocities[j] = point.velocities[j] * spd
                point.accelerations[j] = point.accelerations[j] * spd

            points[i] = point

        new_traj.joint_trajectory.points = points     
        return   new_traj

    def add_table(self):
        """为场景中添加抓取桌面，防止机械臂与桌子发生碰撞
        """
        #清除场景可能存在的遗留物体
        self.scene.remove_world_object('table') 
        #设置桌面尺寸      x  y   z
        table_size = [0.6, 1.2, 0.01]
        #设置桌子的位置姿态
        table_pose = PoseStamped()
        table_pose.header.frame_id = 'panda_link0'
        table_pose.pose.position.x = 0.55
        table_pose.pose.position.y = 0.0
        table_pose.pose.position.z = 0.025
        table_pose.pose.orientation.w = 1.0
        # 将table加入场景当中
        self.scene.add_box('table', table_pose, table_size)

    def set_gripper(self,gripper_width):
        """设置panda 夹爪的开合大小
        gripper_width 最大0.08m
        """
        if gripper_width>0.08 or gripper_width<0.0:
             raise Exception
        oneside=gripper_width/2
        joint_goal = self.panda_hand.get_current_joint_values()
        joint_goal[0] = oneside
        joint_goal[1] = oneside
        self.panda_hand.go(joint_goal, wait=True)
        self.panda_hand.stop()
        rospy.loginfo("Gripper ready")



if __name__ == "__main__":
    try:
        MoveItDemo()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Arm tracker node terminated.")

    
    