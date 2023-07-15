import numpy as np
from isaacgym.terrain_utils import *


class Terrain:
    def __init__(self, cfg, num_robots) -> None:

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

        terrain_type = self.cfg.type + "_terrain"
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                terrain = SubTerrain("terrain",
                                     width=self.width_per_env_pixels,
                                     length=self.width_per_env_pixels,
                                     vertical_scale=self.cfg.vertical_scale,
                                     horizontal_scale=self.cfg.horizontal_scale)
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


def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x - x2: center_x + x2, center_y - y2: center_y + y2] = -1000
    terrain.height_field_raw[center_x - x1: center_x + x1, center_y - y1: center_y + y1] = 0


def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth


def box_terrain(terrain, height, platform_size=1.):
    height = int(height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = height


def special_box_terrain(terrain, height):
    # for 10 x 10 terrain
    height = int(height / terrain.vertical_scale)
    x = [1, 2, 4, 5]
    x = [int(i / terrain.horizontal_scale) for i in x]
    terrain.height_field_raw[x[0]:x[1], x[0]:x[1]] = height
    terrain.height_field_raw[x[0]:x[1], x[2]:x[3]] = height
    terrain.height_field_raw[x[2]:x[3], x[0]:x[1]] = height
    terrain.height_field_raw[x[2]:x[3], x[2]:x[3]] = height
