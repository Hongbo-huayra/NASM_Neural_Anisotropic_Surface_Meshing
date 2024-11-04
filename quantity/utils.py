import trimesh
import os
import numpy as np


def sort_by_id(id_list, path_list):
    id_idx_map = {id_: idx for idx, id_ in enumerate(id_list)}
    sorted_path_list = sorted(
        path_list, key=lambda x: id_idx_map[int(os.path.basename(x).split("_")[0])])
    return sorted_path_list


def normalize(mesh: trimesh.Trimesh):
    bbox_min, bbox_max = mesh.bounds
    center = (bbox_min + bbox_max) / 2
    scale = np.linalg.norm(bbox_max - bbox_min)
    mesh.vertices -= center
    mesh.vertices /= scale
    return mesh
