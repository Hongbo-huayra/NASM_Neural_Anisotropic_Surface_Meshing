import torch
from loguru import logger
from network.graph_Unet import GraphUNet
from dataloader.meshgraph import MeshGraphDatasetEval
from torch_geometric.loader import DataLoader
from dataloader.utils import save_hd_obj
import subprocess
import os
import time
import json
import configargparse
import yaml


def join_path(ori_path, model_id):
    dir_ = os.path.dirname(ori_path)
    fname = os.path.basename(ori_path)
    fname = "{}_{}".format(model_id, fname)
    result = os.path.join(dir_, fname)
    return result


def run_nmcvt(geo_path, emb_path, tri_path, emb_output, output_vert_count=-1, bd_vert=""):
    geo_exe = geo_path
    param = ["",
             "",
             "8",
             "400",
             "",
             "2",
             "6",
             "true",
             "",
             "",
             ""]
    param[0] = emb_path
    param[1] = tri_path
    point_num = int(
        emb_output.shape[0]) if output_vert_count == -1 else output_vert_count
    param[4] = "{}".format(point_num)
    param[8] = bd_vert

    print(param)
    command = [geo_exe] + param
    print(command)
    try:
        running_time = time.time()
        return_code = subprocess.call(command)
        running_time = time.time() - running_time

        print("Return code:", return_code)
    except Exception as e:
        print("An error occurred:", str(e))
    return running_time


if __name__ == "__main__":
    conf = configargparse.ArgParser()

    # Arg
    conf.add_argument("--pretrained_path",
                      help="Path to save the models", default="weights/pretrained/NASM_pretrain.pth")
    conf.add_argument("--mesh_path", help="Path to mesh model",
                      default="test_mesh/107910_sf_norm.obj")
    conf.add_argument("--log_path", default="logs/test.log")
    conf.add_argument("--output_folder", default="test_mesh")
    conf.add_argument(
        "--nmcvt_exe", required=True, help="Path to the LpCVT executable")

    conf = conf.parse_args()

    logger.add(conf.log_path, level='INFO')

    device = 2
    torch.cuda.set_device(device)
    torch.manual_seed(42)

    mgd = MeshGraphDatasetEval(conf.mesh_path)
    trainset_loader = DataLoader(mgd,
                                 1,
                                 shuffle=False)
    MODEL = GraphUNet(6, 256, 8, 3, 5, 0.5).to(device)

    model_dict = torch.load(conf.pretrained_path)
    MODEL.load_state_dict(model_dict)

    eval_time = time.time()

    obj_stem = os.path.basename(conf.mesh_path).split(".")[0]

    emb_time_map = {}
    cvt_time_map = {}
    for index in range(len(mgd)):

        eval_data = mgd.get(index)
        test_mesh = eval_data[0].to(device)

        graph_info = eval_data[1].cpu().detach()
        faces = graph_info.x
        verts = graph_info.verts

        MODEL.eval()

        emb_time = time.time()
        emb_output = MODEL(test_mesh.x, test_mesh.edge_index)

        emb_time = time.time() - emb_time
        logger.info("Embedding time: {}: time:{}".format(obj_stem, emb_time))

        emb_output = emb_output.cpu().detach()
        emb_output = torch.cat([verts, emb_output], dim=-1)

        emb_path = f"{conf.output_folder}/{obj_stem}_emb.obj"
        save_hd_obj(emb_path, emb_output.numpy(), faces)

        tri_dir = f"{conf.output_folder}/{obj_stem}_tri.obj"

        lpcvt_time = run_nmcvt(conf.nmcvt_exe, emb_path,
                               tri_dir, emb_output)

        logger.info("LPCVT time: {} time:{}".format(obj_stem, lpcvt_time))

    eval_time = time.time() - eval_time
    logger.info("Evaluation time: {}".format(eval_time))
    logger.info("Evaluation average time: {}".format(eval_time/len(mgd)))
