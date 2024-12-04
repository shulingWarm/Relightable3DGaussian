#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from arguments import ModelParams
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


#读取camera list时用到的args
class CameraListParams:
    def __init__(self):
        self.resolution=-1
        self.data_device='cuda'

class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0], custom_load = False):
        """b
        :param path: Path to colmap scene main folder.
        """
        # 判断是不是使用自定义的加载方法
        if(custom_load):
            self.customLoad()
            return
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval,
                                                          debug=args.debug_cuda)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            if "stanford_orb" in args.source_path:
                print("Found keyword stanford_orb, assuming Stanford ORB data set!")
                scene_info = sceneLoadTypeCallbacks["StanfordORB"](args.source_path, args.white_background, args.eval, 
                                                                   debug=args.debug_cuda)
            elif "Synthetic4Relight" in args.source_path:
                print("Found transforms_train.json file, assuming Synthetic4Relight data set!")
                scene_info = sceneLoadTypeCallbacks["Synthetic4Relight"](args.source_path, args.white_background, args.eval,
                                                            debug=args.debug_cuda)
            else:
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, 
                                                               debug=args.debug_cuda)
        elif os.path.exists(os.path.join(args.source_path, "inputs/sfm_scene.json")):
            print("Found sfm_scene.json file, assuming NeILF data set!")
            scene_info = sceneLoadTypeCallbacks["NeILF"](args.source_path, args.white_background, args.eval,
                                                         debug=args.debug_cuda)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)

        self.scene_info = scene_info

    # 自己实现的载入scene的方法，主要是用于debug的
    def customLoad(self):
        #当前工程的模型位置
        self.model_path = "/media/zzh/data/temp/RelightModelPath/"
        #读取colmap的工程
        colmap_project = sceneLoadTypeCallbacks["Colmap"]("/media/zzh/data/temp/phone_workspace/",
            "/media/zzh/data/temp/images/", False,
            debug=False)
        #读取点云内容
        with open(colmap_project.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                               'wb') as dest_file:
            dest_file.write(src_file.read())
        json_cams = []
        camlist = []
        if colmap_project.test_cameras:
            camlist.extend(colmap_project.test_cameras)
        if colmap_project.train_cameras:
            camlist.extend(colmap_project.train_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

        #打乱图片序列
        random.shuffle(colmap_project.train_cameras)  # Multi-res consistent random shuffling
        random.shuffle(colmap_project.test_cameras)  # Multi-res consistent random shuffling

        #获取相机的扩展，但目前并不知道这是做什么用的
        self.cameras_extent = colmap_project.nerf_normalization["radius"]

        # 初始化训练相机和测试相机的空字典
        self.train_cameras = {}
        self.test_cameras = {}

        #载入不同层级的相机分辨率
        cameraArgs = CameraListParams()
        print("Loading Training Cameras")
        self.train_cameras[1.0] = cameraList_from_camInfos(colmap_project.train_cameras, 1.0,
                                                                        cameraArgs)
        print("Loading Test Cameras")
        self.test_cameras[1.0] = cameraList_from_camInfos(colmap_project.test_cameras, 1.0,
                                                                       cameraArgs)

        #记录自身的场景信息
        self.scene_info = colmap_project



    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
