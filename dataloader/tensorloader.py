import torch
import numpy as np
from .dataconfigparser import DataConfigParser


class TensorLoader():

    @classmethod
    def from_pathlist(cls, m_pathlist):
        self = cls.__new__(cls)
        self._m_pathlist = m_pathlist
        return self

    @classmethod
    def from_cp(cls, cp: DataConfigParser):
        self = cls.__new__(cls)
        self._id_list = cp.get_id_list()
        self._m_pathlist = cp.get_metric_path_list(cp._type == "train")
        return self

    def load_info(self, index):
        cur_info = np.genfromtxt(
            self._m_pathlist[index], delimiter=',', names=True)

        s1_sq = cur_info["min_cur"]
        s2_sq = cur_info["max_cur"]

        v_min = [cur_info[name]
                 for name in ["min_pd_x", "min_pd_y", "min_pd_z"]]
        v_min = np.stack(v_min, axis=1)
        v_max = [cur_info[name]
                 for name in ["max_pd_x", "max_pd_y", "max_pd_z"]]
        v_max = np.stack(v_max, axis=1)
        normal = [cur_info[name]
                  for name in ["normal_x", "normal_y", "normal_z"]]
        normal = np.stack(normal, axis=1)

        return s1_sq, s2_sq, v_min, v_max, normal

    def load_normal(self, index):
        _, _, _, _, normal = self.load_info(index)
        return torch.tensor(normal, dtype=torch.float)

    def load_m(self, index):
        s1_sq, s2_sq, v_min, v_max, normal = self.load_info(index)
        scale = np.tile(np.eye(3), (len(s1_sq), 1, 1))

        scale[:, 1, 1] = np.sqrt(s2_sq/s1_sq)

        R = np.stack([v_min, v_max, normal], axis=2)
        m = R @ scale @ np.transpose(R, (0, 2, 1))

        return m

    def load_q(self, index):
        s1_sq, s2_sq, v_min, v_max, normal = self.load_info(index)
        scale = np.tile(np.eye(3), (len(s1_sq), 1, 1))

        scale[:, 1, 1] = np.sqrt(s2_sq/s1_sq)

        R = np.stack([v_min, v_max, normal], axis=2)
        Q = scale @ np.transpose(R, (0, 2, 1))
        return Q
