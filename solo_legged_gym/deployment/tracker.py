from pyvicon import *
import torch


class Tracker:
    def __init__(self, server_ip) -> None:
        self.client = PyVicon()
        self.server_ip = server_ip
        self.stream_mode = StreamMode.ClientPullPreFetch
        self.num_diff_frames = 10
        self.base_global_translation_buf = torch.zeros(self.num_diff_frames, 3, requires_grad=False)
        self.base_global_rotation_matrix_buf = torch.zeros(self.num_diff_frames, 3, 3, requires_grad=False)
        self.base_global_quaternion_buf = torch.zeros(self.num_diff_frames, 4, requires_grad=False)
        self.base_frame_number_buf = torch.zeros(self.num_diff_frames, requires_grad=False)
        self.base_global_lin_vel = torch.zeros(3)
        self.base_local_lin_vel = torch.zeros(3)
        self.base_global_ang_vel = torch.zeros(3)
        self.base_local_ang_vel = torch.zeros(3)
        print("[Tracker] SDK version : {}".format(self.client.__version__))

    def connect(self):
        print("[Tracker] Connecting to server {}:".format(self.server_ip), self.client.connect(self.server_ip))
        print("[Tracker] Enable segment data:", self.client.enable_segment_data())
        print("[Tracker] Set stream mode {}:".format(self.stream_mode), self.client.set_stream_mode(self.stream_mode))
        while self.client.is_connected():
            if self.client.get_frame():
                self.subject_name = self.client.get_subject_name(0)
                if self.subject_name:
                    self.frame_rate = self.client.get_frame_rate()
                    print("[Tracker] Frame rate:", self.frame_rate)
                    print("[Tracker] Total latency:", self.client.get_latency_total())
                    print("[Tracker] Subject name:", self.subject_name)
                    print("[Tracker] Subject quality:", self.client.get_subject_quality(self.subject_name))
                    self.segment_name = self.client.get_subject_root_segment_name(self.subject_name)
                    print("[Tracker] Segment name:", self.segment_name)
                    break

    def disconnect(self):
        print("[Tracker] Disconnecting:", self.client.disconnect())

    def get_base_quat(self):
        self.client.get_frame()
        return torch.from_numpy(self.client.get_segment_global_quaternion(self.subject_name, self.segment_name))

    def compute_base_vel(self):
        self.base_frame_number_buf[:-1] = self.base_frame_number_buf[1:].clone()
        self.base_global_translation_buf[:-1, :] = self.base_global_translation_buf[1:, :].clone()
        self.base_global_rotation_matrix_buf[:-1, ...] = self.base_global_rotation_matrix_buf[1:, ...].clone()
        self.base_global_quaternion_buf[:-1, :] = self.base_global_quaternion_buf[1:, :].clone()

        while self.client.is_connected():
            self.client.get_frame()
            global_translation = self.client.get_segment_global_translation(self.subject_name, self.segment_name)
            global_rotation_matrix = self.client.get_segment_global_rotation_matrix(self.subject_name,
                                                                                    self.segment_name)
            global_quaternion = self.client.get_segment_global_quaternion(self.subject_name, self.segment_name)
            if global_translation is None or global_rotation_matrix is None or global_quaternion is None:
                continue
            else:
                self.base_frame_number_buf[-1] = self.client.get_frame_number()
                self.base_global_translation_buf[-1, :] = torch.from_numpy(
                    global_translation) / 1000  # convert mm/s to m/s
                self.base_global_rotation_matrix_buf[-1, ...] = torch.from_numpy(global_rotation_matrix)
                self.base_global_quaternion_buf[-1, :] = torch.from_numpy(global_quaternion[[1, 2, 3, 0]])
                break

        dt_buf = (self.base_frame_number_buf[1:] - self.base_frame_number_buf[:-1]) / self.frame_rate

        # lin_vel
        base_global_translation_buf = self.base_global_translation_buf[1:, :] - self.base_global_translation_buf[:-1, :]
        base_global_lin_vel_buf = base_global_translation_buf / dt_buf.view(-1, 1)
        self.base_global_lin_vel = base_global_lin_vel_buf.nanmean(dim=0)
        base_local_lin_vel_buf = self.base_global_rotation_matrix_buf[1:, ...].transpose(1,
                                                                                         2) @ base_global_lin_vel_buf.view(
            -1, 3, 1)
        self.base_local_lin_vel = base_local_lin_vel_buf.nanmean(dim=0).squeeze(1)

        # ang_vel
        base_global_rotation_matrix_diff_buf = (self.base_global_rotation_matrix_buf[1:,
                                                ...] - self.base_global_rotation_matrix_buf[:-1, ...]) / dt_buf.view(-1,
                                                                                                                     1,
                                                                                                                     1)
        base_global_ang_vel_skewed_buf = base_global_rotation_matrix_diff_buf @ self.base_global_rotation_matrix_buf[1:,
                                                                                ...].transpose(1, 2)
        base_global_ang_vel_buf = torch.cat((base_global_ang_vel_skewed_buf[:, 2, 1].view(-1, 1),
                                             base_global_ang_vel_skewed_buf[:, 0, 2].view(-1, 1),
                                             base_global_ang_vel_skewed_buf[:, 1, 0].view(-1, 1)),
                                            dim=1)
        self.base_global_ang_vel = base_global_ang_vel_buf.nanmean(dim=0)
        base_local_ang_vel_buf = self.base_global_rotation_matrix_buf[1:, ...].transpose(1,
                                                                                         2) @ base_global_ang_vel_buf.view(
            -1, 3, 1)
        self.base_local_ang_vel = base_local_ang_vel_buf.nanmean(dim=0).squeeze(1)

    def get_root_states(self):
        self.compute_base_vel()
        return torch.cat((self.base_global_translation_buf[-1, :], self.base_global_quaternion_buf[-1, :],
                          self.base_global_lin_vel, self.base_global_ang_vel))
