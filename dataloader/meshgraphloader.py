from . import utils as ut
from .dataconfigparser import DataConfigParser
import trimesh
import torch


class MeshGraphLoader():
    def __init__(self, cp: DataConfigParser) -> None:
        self._id_list = cp.get_id_list()
        self._ori_obj_pathlist = cp.get_obj_path_list()
        self._hd_obj_pathlist = cp.get_gt_path_list()

    def load_mesh_attr(self, index):
        mesh = trimesh.load(self._ori_obj_pathlist[index])
        verts = torch.tensor(mesh.vertices, dtype=torch.float)
        vert_nghb = mesh.vertex_neighbors

        edges = torch.tensor(mesh.edges, dtype=torch.long).T.contiguous()
        faces = torch.tensor(mesh.faces, dtype=torch.long)
        return verts, vert_nghb, edges, faces

    def load_hd_verts(self, index):
        hd_verts, _ = ut.load_hd_mesh(self._hd_obj_pathlist[index])
        return hd_verts


class MeshGraphLoaderEval():
    def __init__(self, mesh_path) -> None:
        self._ori_obj_pathlist = [mesh_path]

    def load_mesh_attr(self, index):
        mesh = trimesh.load(self._ori_obj_pathlist[index])
        verts = torch.tensor(mesh.vertices, dtype=torch.float)
        vert_nghb = mesh.vertex_neighbors

        edges = torch.tensor(mesh.edges, dtype=torch.long).T.contiguous()
        faces = torch.tensor(mesh.faces, dtype=torch.long)
        return verts, vert_nghb, edges, faces

    def load_normal(self, index):
        mesh = trimesh.load(self._ori_obj_pathlist[index])
        return torch.tensor(mesh.vertex_normals, dtype=torch.float)

    def load_hd_verts(self, index):
        hd_verts, _ = ut.load_hd_mesh(self._hd_obj_pathlist[index])
        return hd_verts
