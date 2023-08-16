import numpy as np
from isaacgym.terrain_utils import *


class Terrain:
    def __init__(self, cfg, num_robots, play) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.mesh_type = cfg.mesh_type

        if self.mesh_type == 'plane':
            return

        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols

        self.sub_terrain_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                terrain = SubTerrain("terrain",
                                     width=self.width_per_env_pixels,
                                     length=self.width_per_env_pixels,
                                     vertical_scale=self.cfg.vertical_scale,
                                     horizontal_scale=self.cfg.horizontal_scale)
                if play:
                    terrain_type = self.cfg.play_terrain + "_terrain"
                else:
                    if i < int(self.cfg.frac_pit * self.cfg.num_rows):
                        terrain_type = "pit_terrain"
                    else:
                        terrain_type = "boxr_terrain"

                eval(terrain_type)(terrain, self.cfg.params[j])
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels
                self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                x1 = int((self.env_length / 2. - 1) / terrain.horizontal_scale)
                x2 = int((self.env_length / 2. + 1) / terrain.horizontal_scale)
                y1 = int((self.env_width / 2. - 1) / terrain.horizontal_scale)
                y2 = int((self.env_width / 2. + 1) / terrain.horizontal_scale)
                env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
                self.sub_terrain_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

        self.height_samples = self.height_field_raw

        if self.mesh_type == "trimesh":
            self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw,
                                                                           self.cfg.horizontal_scale,
                                                                           self.cfg.vertical_scale,
                                                                           self.cfg.slope_threshold)


def box_terrain(terrain, height):
    height = int(height / terrain.vertical_scale)
    x = [1.0, 3.0, 5.0, 7.0]
    x = [int(i / terrain.horizontal_scale) for i in x]
    terrain.height_field_raw[x[0]:x[1], x[2]:x[3]] = height
    terrain.height_field_raw[x[2]:x[3], x[2]:x[3]] = height
    terrain.height_field_raw[x[0]:x[1], x[0]:x[1]] = height
    terrain.height_field_raw[x[2]:x[3], x[0]:x[1]] = height


def boxr_terrain(terrain, height):
    height = int(height / terrain.vertical_scale)
    num_boxes = 10
    box_size = [0.5, 1.5]
    sampled_size = np.random.uniform(box_size[0], box_size[1], num_boxes)
    sampled_size = [int(i / terrain.horizontal_scale) for i in sampled_size]
    for i in range(num_boxes):
        x = np.random.uniform(0.0, 8.0 - box_size[1])
        y = np.random.uniform(0.0, 8.0 - box_size[1])
        x = int(x / terrain.horizontal_scale)
        y = int(y / terrain.horizontal_scale)
        terrain.height_field_raw[x:x + sampled_size[i], y:y + sampled_size[i]] = height


def box2_terrain(terrain, height):
    height = int(height / terrain.vertical_scale)
    x = [1.5, 2.5, 4 - 0.5, 4 + 0.5]
    # x = [1.25, 2.75, 0.0, 8.0]
    x = [int(i / terrain.horizontal_scale) for i in x]
    terrain.height_field_raw[x[0]:x[1], x[2]:x[3]] = height
    # terrain.height_field_raw[x[2]:x[3], x[2]:x[3]] = height
    # terrain.height_field_raw[x[0]:x[1], x[0]:x[1]] = height
    # terrain.height_field_raw[x[2]:x[3], x[0]:x[1]] = height


def pit_terrain(terrain, height):
    # for 6 x 6 terrain
    height = int(height / terrain.vertical_scale)
    x = [2.0, 6.0]
    x = [int(i / terrain.horizontal_scale) for i in x]
    terrain.height_field_raw[x[0]:x[1], x[0]:x[1]] = -height
    # terrain.height_field_raw[x[0]:x[1], x[0]:x[3]] = height
    # terrain.height_field_raw[x[2]:x[3], x[0]:x[3]] = height
    # terrain.height_field_raw[x[0]:x[3], x[0]:x[1]] = height
    # terrain.height_field_raw[x[0]:x[3], x[2]:x[3]] = height

