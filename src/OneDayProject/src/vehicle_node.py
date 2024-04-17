#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import pickle

import numpy as np
import random
import rospy
import math
import copy
import rospkg
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
import glob
import scipy.io as sio
import tf

from geometry_msgs.msg import Twist, Point32, PolygonStamped, Polygon, Vector3, Pose, Quaternion, Point
from visualization_msgs.msg import MarkerArray, Marker

from std_msgs.msg import Float32, Float64, Header, ColorRGBA, UInt8, String, Float32MultiArray, Int32MultiArray
from hmmlearn.hmm import GMMHMM, GaussianHMM

from msgs.msg import dataset_array_msg, dataset_msg, map_array_msg, map_msg, point_msg


class Environments(object):
    def __init__(self):
        rospy.init_node('Environments')

        self.init_variable()
        self.set_subscriber()
        self.set_publisher()
        self.load_map()

        r = rospy.Rate(20)
        while not rospy.is_shutdown():

            self.loop()
            r.sleep()


    def load_map(self):
        self.map_file = []
        map_path = rospy.get_param("map_path")
        matfiles = ["waypoints_0_rev.mat",
                    "waypoints_1_rev.mat",
                    "waypoints_2_rev.mat",
                    "waypoints_3_rev.mat"
                    ]

        # 124.20403395607009 738.5587856439931 1356.4053875886939
        station_offset = [0, 0, 0, 0]

        for i, matfile in enumerate(matfiles):
            mat = sio.loadmat(map_path+matfile)

            easts = mat["east"][0]
            norths = mat["north"][0]
            stations = mat["station"][0]+station_offset[i]

            # if i==2:
            self.map_file.append(np.stack([easts, norths, stations],axis=-1))


        self.D_list = [ 0, -3.85535188 ,-7.52523438, -7.37178602]
        self.D_list = np.array(self.D_list)


    def init_variable(self):
        self.pause = False
        self.time = 11
        self.br = tf.TransformBroadcaster()

        SamplePath = rospy.get_param("SamplePath")
        SampleList = sorted(glob.glob(SamplePath+"/*.pickle"))
        SampleId = (int)(rospy.get_param("SampleId"))

        ######################### Load Vehicle #######################

        with open(SampleList[SampleId], 'rb') as f:
            self.vehicles = pickle.load(f)
        # self.Logging[veh.track_id].append([veh.lane_id, veh.target_lane_id, veh.s,
        #         #                          veh.d, veh.pose[0], veh.pose[1], veh.pose[2], veh.v, veh.yawrate, MODE[veh.mode], veh.ax, veh.steer, veh.length, veh.width])

        ########################## HMM ###############################
        # with open("/home/mmc_ubuntu/Work/system-infra/Simulation/log/model_LC.pickle", 'rb') as f:
        #     self.hmm_lc = pickle.load(f)

        # with open("/home/mmc_ubuntu/Work/system-infra/Simulation/log/model_LK.pickle", 'rb') as f:
        #     self.hmm_lk = pickle.load(f)


    def loop(self):

        if self.pause:
            pass
        else:
            self.publish()
            self.pub_map()
            self.time+=1

        if self.time>=(len(self.vehicles[0])-1):
            rospy.signal_shutdown("End of the logging Time")
            # asdf


    def callback_plot(self, data):

        if data.linear.x>0 and data.angular.z>0: #u
            self.pause = True
        else:
            self.pause = False

    def publish(self, is_delete = False):

        ObjectsData = dataset_array_msg()

        for i in range(len(self.vehicles)):

            ObjectData = dataset_msg()
            ObjectData.id = i
            ObjectData.lane_id = self.vehicles[i][self.time][0]
            ObjectData.length = self.vehicles[i][self.time][12]
            ObjectData.width = self.vehicles[i][self.time][13]

            for t in range(self.time-10, self.time+1):
                ObjectData.x.append(self.vehicles[i][t][4])
                ObjectData.y.append(self.vehicles[i][t][5])
                ObjectData.yaw.append(self.vehicles[i][t][6])
                ObjectData.vx.append(self.vehicles[i][t][7])
                ObjectData.s.append(self.vehicles[i][t][2])
                ObjectData.d.append(self.vehicles[i][t][3])


            ObjectsData.data.append(ObjectData)

        self.history_pub.publish(ObjectsData)


    def callback_result(self, data):

        Objects = MarkerArray()
        Texts = MarkerArray()

        for i in range(len(self.vehicles)):

            veh_data = np.array(self.vehicles[i][self.time-10:self.time+1])


            # 위치(d값)의 변화 계산
            d_changes = np.diff(veh_data[:, 3])

            pred = "LC" if np.mean(np.abs(d_changes)) > 0.5 else "LK"
            gt = "LC" if self.vehicles[i][self.time][9] == 1 else "LK"


            q = tf.transformations.quaternion_from_euler(0, 0, self.vehicles[i][self.time][6])

            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.id = i
            marker.type = Marker.CUBE

            marker.pose.position.x = self.vehicles[i][self.time][4]
            marker.pose.position.y = self.vehicles[i][self.time][5]
            marker.pose.position.z = 0.5

            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]

            marker.scale.x = self.vehicles[i][self.time][12]
            marker.scale.y = self.vehicles[i][self.time][13]
            marker.scale.z = 1
            marker.color.a = 1.0

            Objects.markers.append(marker)

            text = Marker()
            text.header.frame_id = "world"
            text.ns = "text"
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING

            text.action = Marker.ADD

            text.color = ColorRGBA(1, 1, 1, 1)
            text.scale.z = 5
            text.text = str(i)+" / True : " + gt+" / Pred : "  + pred
            text.pose.position = Point(self.vehicles[i][self.time][4], self.vehicles[i][self.time][5], 3)

            Texts.markers.append(text)

        self.sur_pose_plot.publish(Objects)
        self.text_plot.publish(Texts)


        self.br.sendTransform((self.vehicles[0][self.time][4], self.vehicles[0][self.time][5], 0),
                                tf.transformations.quaternion_from_euler(0, 0,self.vehicles[0][self.time][6]),
                                rospy.Time.now(),
                                "base_link",
                                "world")

    def pub_map(self, is_delete = False):

        MapData = map_array_msg()
        for i in range(len(self.map_file)):
            MapSeg = map_msg()
            MapSeg.path_id = i

            temp = self.map_file[i]
            for j in range(len(temp)):

                point = point_msg()
                point.x = temp[j,0]
                point.y = temp[j,1]
                point.s = temp[j,2]
                point.d = self.D_list[i]
                MapSeg.center.append(point)

            MapData.data.append(MapSeg)
        self.map_pub.publish(MapData)

        Maps = MarkerArray()
        for i in range(len(self.map_file)):
            MapSeg = map_msg()
            MapSeg.path_id = i

            line_strip = Marker()
            line_strip.type = Marker.LINE_STRIP
            line_strip.id = i
            line_strip.scale.x = 2
            line_strip.scale.y = 0.1
            line_strip.scale.z = 0.1

            line_strip.color = ColorRGBA(1.0,1.0,1.0,0.5)
            line_strip.header = Header(frame_id='world')

            temp = self.map_file[i]
            for j in range(len(temp)):
                point = Point()
                point.x = temp[j,0]
                point.y = temp[j,1]
                point.z = 0

                line_strip.points.append(point)

                point = point_msg()
                point.x = temp[j,0]
                point.y = temp[j,1]
                point.s = temp[j,2]
                point.d = self.D_list[i]
                MapSeg.center.append(point)

            Maps.markers.append(line_strip)

        self.map_plot.publish(Maps)

    def set_subscriber(self):
        rospy.Subscriber('/cmd_vel',Twist, self.callback_plot,queue_size=1)
        rospy.Subscriber('/result', dataset_array_msg, self.callback_result, queue_size=1)

    def set_publisher(self):

        self.sur_pose_plot = rospy.Publisher('/rviz/sur_obj_pose', MarkerArray, queue_size=1)
        self.map_plot = rospy.Publisher('/rviz/maps', MarkerArray, queue_size=1)
        self.text_plot = rospy.Publisher('/rviz/text', MarkerArray, queue_size=1)
        self.map_pub = rospy.Publisher('/map_data', map_array_msg, queue_size=1)
        self.history_pub = rospy.Publisher('/history', dataset_array_msg, queue_size=1)



if __name__ == '__main__':

    try:
        f = Environments()

    except rospy.ROSInterruptException:
        rospy.logerr('Could not start node.')

