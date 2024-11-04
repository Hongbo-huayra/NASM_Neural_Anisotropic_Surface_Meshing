import torch_geometric.data as data
import torch
from .meshgraphloader import MeshGraphLoader as mgl
from .meshgraphloader import MeshGraphLoaderEval as mgle
from .tensorloader import TensorLoader as tl
from .utils import gen_dot_id
from .dataconfigparser import DataConfigParser as cp


class MeshGraphDataset(data.Dataset):
    def __init__(self, id_list, root_path, ext_config, is_augment):
        super(MeshGraphDataset, self).__init__()
        data_conf = {
            "id_list": [id_list],
            "root_path": root_path,
            "is_augment": is_augment
        }
        data_conf.update(ext_config)

        self._cp = cp(data_conf, "train")
        self._data_loader = mgl(self._cp)
        self._tmp_loader = tl.from_cp(self._cp)

    def len(self):
        return len(self._data_loader._ori_obj_pathlist)

    def get(self, index):
        verts, _, edges, faces = self._data_loader.load_mesh_attr(
            index)

        node_ft = verts
        metric = self._tmp_loader.load_normal(index)

        node_ft = torch.cat((node_ft, metric), dim=-1)

        graph = data.Data(x=node_ft, edge_index=edges)

        dot_id = gen_dot_id(faces)

        hd_verts = self._data_loader.load_hd_verts(index)

        graph_gt = data.Data(x=hd_verts,
                             dot_id=dot_id
                             )
        return graph, graph_gt


class MeshGraphDatasetEval(data.Dataset):
    def __init__(self, mesh_path):
        super(MeshGraphDatasetEval, self).__init__()
        self._data_loader = mgle(mesh_path)

    def len(self):
        return len(self._data_loader._ori_obj_pathlist)

    def get(self, index):
        verts, _, edges, faces = self._data_loader.load_mesh_attr(
            index)
        node_ft = verts

        metric = self._data_loader.load_normal(index)

        node_ft = torch.cat((node_ft, metric), dim=-1)

        graph = data.Data(x=node_ft, edge_index=edges)
        graph_info = data.Data(x=faces, verts=verts)

        return graph, graph_info
