import open3d
from open3d import *
import numpy as np
import cv2
import yaml
import time
import os


class Camera:
    def __init__(self, cfg):
        cm = cfg['camera_setting']
        self.camera_channel = cm['camera_channel']
        self.resolution_option = cm['resolution_option']
        self.crop_width = cm['crop_width']
        self.crop_height = cm['crop_height']
        self.cap = cv2.VideoCapture(self.camera_channel)
        print('-----------Camera is initializing-----------')
        if self.cap.isOpened():
            print('------Camera is open--------')
        img_width = [320, 640, 800, 1280]
        img_height = [240, 480, 600, 720]
        self.cap_row = img_height[self.resolution_option]
        self.cap_col = img_width[self.resolution_option]
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cap_col)  # width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_row)  # height
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.width_begin = int((self.cap_col - self.crop_width) / 2)
        self.width_end = int((self.cap_col + self.crop_width) / 2)
        self.height_begin = int((self.cap_row - self.crop_height) / 2)
        self.height_end = int((self.cap_row + self.crop_height) / 2)
        print('-----------Camera is initialized-----------')

    def crop_image(self, img):
        return img[self.height_begin:self.height_end, self.width_begin:self.width_end]

    def get_image(self):
        # img = self.crop_image(self.cap.read()[1]) #0.038s
        time_here = time.time()
        ret, img = self.cap.read()  # 0.04s
        # print(time.time()-time_here)
        img = self.crop_image(img)
        return img

    def get_ref_image(self):
        # time.sleep(10)
        global ref_img_1
        while True:
            ref_img_1 = self.crop_image(self.cap.read()[1])
            cv2.imshow('ref_img_1', ref_img_1)
            key = cv2.waitKey(1)
            if key == ord('w'):  # begin shooting
                cv2.destroyWindow('ref_img_1')
                break
            if key == ord('q'):
                quit()
        ref_img_add = np.zeros_like(ref_img_1, float)
        ref_img_number = 10
        for i in range(ref_img_number):
            raw_image = self.crop_image(self.cap.read()[1])
            # cv2.imshow('raw_image', raw_image)
            # cv2.waitKey()
            ref_img_add += raw_image
            time.sleep(0.2)
            # print(time.time())
        ref_img_avg = ref_img_add / ref_img_number
        ref_img_avg = ref_img_avg.astype(np.uint8)
        # cv2.imshow('ref_img_avg', ref_img_avg)
        # cv2.waitKey()
        return ref_img_avg


class HeightMap:
    def __init__(self, cfg):
        hmap = cfg['heightmap']
        self.heightmap_type = hmap['heightmap_type']
        table_saving_directory = hmap['look-up_table_path'] + hmap['directory_prefix'] + str(hmap['sensor_id']) + hmap[
            'table_directory_id']
        self.channel_used = hmap['channel_used']
        self.lighting_threshold = hmap['lighting_threshold']
        self.diff_blur = hmap['diff_blur']
        self.diff_blur_kernel = hmap['diff_blur_kernel']
        self.heightmap_blur = hmap['heightmap_blur']
        self.heightmap_blur_kernel = hmap['heightmap_blur_kernel']
        if self.heightmap_type:
            # self.B_G_R_table = np.load(hmap['B_G_R_table_path'])
            # self.B_G_R_table_smooth = np.load(hmap['B_G_R_table_smooth_path'])
            if self.channel_used == 0:
                self.channel_H_table = np.load(table_saving_directory + hmap['B_H_name'])
            elif self.channel_used == 1:
                self.channel_H_table = np.load(table_saving_directory + hmap['G_H_name'])
            elif self.channel_used == 2:
                self.channel_H_table = np.load(table_saving_directory + hmap['R_H_name'])
            else:
                self.channel_H_table = np.load(table_saving_directory + hmap['GRAY_H_name'])
                # print(self.channel_H_table)
        else:
            self.h_w_R_table = np.load(table_saving_directory + hmap['h_w_R_table_name'])
            self.h_w_R_table_smooth = np.load(table_saving_directory + hmap['h_w_R_table_smooth_name'])
            self.row_index = np.linspace(0, self.h_w_R_table_smooth.shape[0] - 1, self.h_w_R_table_smooth.shape[0])
            self.col_index = np.linspace(0, self.h_w_R_table_smooth.shape[1] - 1, self.h_w_R_table_smooth.shape[1])
            self.row, self.col = np.meshgrid(self.row_index, self.col_index, indexing='ij')

    def get_height_map(self, img, ref):
        diff_raw_rgb = ref - img
        diff_raw_mask = (diff_raw_rgb < 150).astype(np.uint8)
        diff_raw_rgb = diff_raw_mask * diff_raw_rgb
        if self.channel_used == 3:
            # time_here = time.time()
            ref_channel = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
            img_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # print(time.time() - time_here)  # 0.0002s
        else:  # 1 channel
            ref_channel = ref[::, ::, self.channel_used]
            img_channel = img[::, ::, self.channel_used]

        # time_here = time.time()
        # diff_raw = ref_channel - img_channel
        diff_r = ref_channel - img_channel - self.lighting_threshold
        diff_mask = (diff_r < 100).astype(np.uint8)
        diff_channel = diff_r * diff_mask + self.lighting_threshold
        # print(time.time() - time_here)  # 0.0004s

        if self.diff_blur:
            # time_here = time.time()
            diff_channel = cv2.GaussianBlur(diff_channel.astype(np.float32),
                                            (self.diff_blur_kernel, self.diff_blur_kernel), 0).astype(int)
            # print(time.time() - time_here) #0.006s

        if self.heightmap_type:
            # time_here = time.time()
            height_map = self.channel_H_table[diff_channel] - self.channel_H_table[self.lighting_threshold]
            # print(time.time() - time_here) #0.006s
        else:
            height_map = self.h_w_R_table_smooth[self.row, self.col, diff_channel]

        # threshold_2 = 6
        # diff_R_raw = ref[::,::,2] - img[::,::,2]
        # diff_mask_1 = (diff_R_raw < 100).astype(np.uint8)
        # diff_mask_2 = (diff_R_raw > threshold_2).astype(np.uint8)
        # diff_R = diff_R_raw * diff_mask_1*diff_mask_2 + threshold_2
        # if self.heightmap_type:
        #     height_map = self.R_H[diff_R] - self.R_H[threshold_2]
        #     print(height_map[300, 300])
        # else:
        #     height_map = self.h_w_R_table_smooth[self.row, self.col, diff_R]
        return diff_raw_rgb, height_map


class Reconstruction:
    def __init__(self, cfg):
        cm = cfg['camera_setting']
        cali = cfg['calibration']
        hmap = cfg['heightmap']
        self.n, self.m = cm['crop_width'], cm['crop_height']
        self.points = np.zeros([self.n * self.m, 3])
        self.Pixmm = cali['Pixmm'][hmap['sensor_id'] - 1]
        self.init_points()

    def init_points(self):
        x = np.arange(self.n)
        y = np.arange(self.m)
        self.X, self.Y = np.meshgrid(x, y)
        Z = np.sin(self.X)
        self.points[:, 0] = np.ndarray.flatten(self.X) * self.Pixmm
        self.points[:, 1] = np.ndarray.flatten(self.Y) * self.Pixmm
        self.points[:, 2] = np.ndarray.flatten(Z)

    def get_points(self, Z):
        self.points[:, 2] = np.ndarray.flatten(Z)
        return self.points

    def get_points_grandient(self, Z):
        height_gradient = np.gradient(Z)
        self.points[:, 2] = np.ndarray.flatten(Z)
        return self.points, height_gradient


class Visualizer:
    def __init__(self, cfg, points):
        cali = cfg['calibration']
        vsl = cfg['visualizer']
        hmap = cfg['heightmap']
        self.Pixmm = cali['Pixmm'][hmap['sensor_id'] - 1]
        self.show_ref_image = vsl['show_ref_image']
        self.show_raw_image = vsl['show_raw_image']
        self.show_diff_image = vsl['show_diff_image']
        self.show_height_map = vsl['show_height_map']
        self.show_height_map_blur = vsl['show_height_map_blur']
        self.show_point_cloud = vsl['show_point_cloud']
        self.init_visualizer(points)

    def init_visualizer(self, points):
        self.pcd = open3d.geometry.PointCloud()
        self.pcd.points = open3d.utility.Vector3dVector(points)

        self.vis = open3d.visualization.Visualizer()
        # self.vis.get_render_option()
        # opt.background_color = np.asarray([0,0,0])
        self.vis.create_window(window_name='TouchSensor', width=2000, height=2000)
        self.vis.add_geometry(self.pcd)

        self.ctr = self.vis.get_view_control()
        self.ctr.change_field_of_view(-10)
        print("fov", self.ctr.get_field_of_view())
        self.ctr.convert_to_pinhole_camera_parameters()
        self.ctr.set_zoom(0.9)
        self.ctr.rotate(0, -400)  # mouse drag in x-axis, y-axis
        self.vis.update_renderer()

    def update(self, points, gradient):
        dx, dy = gradient
        dx, dy = dx * 10, dy * 10
        np_colors = dx + 0.5
        np_colors[np_colors < 0] = 0
        np_colors[np_colors > 1] = 1
        np_colors = np.ndarray.flatten(np_colors)
        colors = np.zeros([points.shape[0], 3])

        for _ in range(3):
            colors[:, _] = np_colors

        self.pcd.points = open3d.utility.Vector3dVector(points)
        self.pcd.colors = open3d.utility.Vector3dVector(colors)

        try:
            self.vis.update_geometry()
        except:
            self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()


if __name__ == '__main__':
    f = open("config.yaml", 'r+', encoding='utf-8')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg_vis = cfg['visualizer']
    cfg_save_path = cfg['save_path']
    print(cfg_save_path['object_name']) # check if successfully changes

    camera = Camera(cfg)
    hm = HeightMap(cfg)
    ref = camera.get_ref_image()
    print('Got ref_images successfully!')
    rcs = Reconstruction(cfg)
    global vis
    if cfg_vis['show_point_cloud']:
        vis = Visualizer(cfg, rcs.points)

    ################# preparing to save image ################
    ctr = 0  # counter for images
    # creating paths
    current_path = os.getcwd()
    object_path = os.path.join(current_path, cfg_save_path['root_path'], cfg_save_path['object_name'])
    raw_path = os.path.join(object_path, 'raw')
    os.makedirs(raw_path, exist_ok=True)
    diff_path = os.path.join(object_path, 'diff')
    os.makedirs(diff_path, exist_ok=True)
    depth_map_path = os.path.join(object_path, 'depth')
    os.makedirs(depth_map_path, exist_ok=True)
    height_path = os.path.join(object_path, 'height')
    os.makedirs(height_path, exist_ok=True)

    # while camera.cap.isOpened():

    while True:

        # input("Press any key to continue...")
        time_begin = time.time()

        ##################### Getting Image #####################

        # img = camera.crop_image(camera.cap.read()[1]) #0.038s
        # ret, image = camera.cap.read()
        # img = camera.crop_image(image)
        img = camera.get_image()  # 0.038s
        diff_raw_rgb, height_map = hm.get_height_map(img, ref)  # 0.003s
        global height_map_blur, height_map_blur_2
        if hm.heightmap_blur:
            # time_here = time.time()
            height_map_blur = cv2.GaussianBlur(height_map.astype(np.float32),
                                               (hm.heightmap_blur_kernel, hm.heightmap_blur_kernel), 0)
            height_map_blur_2 = cv2.GaussianBlur(height_map_blur.astype(np.float32),
                                                 (hm.heightmap_blur_kernel, hm.heightmap_blur_kernel), 0)
            # print(time.time()-time_here) #0.002s
        # if vis_2.show_ref_image:
        #     cv2.imshow('ref_image', ref)
        depth_k = 140.0  # for visualizing depth map to the order of 255

        ####################### Showing Image ########################

        if cfg_vis['show_raw_image']:
            cv2.imshow('raw_image', img)
            cv2.imshow('diff_raw_image_rgb', diff_raw_rgb)
        if cfg_vis['show_height_map']:
            height_map_show = height_map * depth_k
            height_map_show_uint = height_map_show.astype(np.uint8)
            cv2.imshow('height_map', height_map_show_uint)
        if cfg_vis['show_height_map_blur'] and hm.heightmap_blur:
            height_map_blur_show = height_map_blur * depth_k
            height_map_blur_show_uint = height_map_blur_show.astype(np.uint8)
            cv2.imshow('height_map_blur', height_map_blur_show_uint)
            height_map_blur_show_2 = height_map_blur_2 * depth_k
            height_map_blur_show_uint_2 = height_map_blur_show_2.astype(np.uint8)
            cv2.imshow('height_map_blur_show_uint_2', height_map_blur_show_uint_2)
            # height_map_blur_1 = height_map_blur_show_uint.astype(np.float)/depth_k
            # cv2.imshow('height_map_blur_1', height_map_blur_1)

        #################### Saving Image ###########################
        # default image size is 600*600 for our camera

        if cfg_vis['show_ref_image'] or cfg_vis['show_raw_image'] or cfg_vis['show_diff_image'] or cfg_vis[
            'show_height_map'] or cfg_vis['show_height_map_blur']:

            print("-----Preparing to save image-----")
            key = cv2.waitKey(1)
            if key == ord('s'):  # save image
                # fix later
                print("Saving image number", ctr)
                os.chdir(raw_path)
                cv2.imwrite('raw_' + str(ctr).zfill(3) + ".png", img)
                os.chdir(diff_path)
                cv2.imwrite('diff_' + str(ctr).zfill(3) + ".png", diff_raw_rgb)
                os.chdir(depth_map_path)
                cv2.imwrite('depth_' + str(ctr).zfill(3) + ".png", height_map_blur_show_uint_2)  # may vary if cfg is changed; fixed for now
                os.chdir(height_path)
                np.save('height_' + str(ctr).zfill(3) + ".npy", height_map_blur_2)
                ctr += 1
                print(ctr)
                os.chdir(current_path)
            if key == ord('q') or ctr >= 500:
                break
        if cfg_vis['show_point_cloud']:
            if not vis.vis.poll_events():
                break
            else:
                if hm.heightmap_blur:
                    points, gradient = rcs.get_points_grandient(height_map_blur_2)
                else:
                    points, gradient = rcs.get_points_grandient(height_map)
                # points, gradient = rcs.get_points_grandient(height_map)
                vis.update(points, gradient)

        ################ Calculating time (and frequency) ##################

        time_end = time.time()
        time_used = time_end - time_begin
        print('Time used to process: {}'.format(time_used))

    # prepating to exit
    cv2.destroyAllWindows()