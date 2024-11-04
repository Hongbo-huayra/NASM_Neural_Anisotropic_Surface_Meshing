import torch
import time
from network.graph_Unet import GraphUNet
from dataloader.meshgraph import MeshGraphDataset
from torch_geometric.utils import scatter, degree
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from loguru import logger
from torch.nn import MSELoss
import os
import numpy as np
import configargparse
import yaml


if __name__ == "__main__":
    conf = configargparse.ArgParser()

    # Arg
    conf.add_argument("--dataset_root",
                      help="Root path of the dataset", default="/media/hongbo/45ad552c-e83b-4f01-9864-7d87cfa1377e/hongbo/Thing10k_tetwild/hd_output_sqrt")
    conf.add_argument("--model_path",
                      help="Path to save the models", default="/home/hongbo/Desktop/code/tmp/Graph_Embedding/weights")
    conf.add_argument("--id_list", default="conf/train_ids.txt")
    conf.add_argument("--log_path", default="logs/train.log")

    conf = conf.parse_args()

    logger.add(conf.log_path, level='INFO')
    logger.info("New Experiment")

    device = 2
    torch.cuda.set_device(device)

    with open("conf/data_config.yaml", 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)

    mgd = MeshGraphDataset(conf.id_list,
                           conf.dataset_root,
                           data_config,
                           True)

    trainset_loader = DataLoader(mgd,
                                 batch_size=4,
                                 shuffle=True,
                                 num_workers=8)

    MODEL = GraphUNet(6, 256, 8, 3, 5, 0.5).to(device)

    EPOCHS = 600
    LR = 0.01
    milestones = [100, 200, 300, 400, 500]
    gamma = 0.5
    OPTIMIZER = torch.optim.AdamW(MODEL.parameters(), LR)
    SCHEDULER = torch.optim.lr_scheduler.MultiStepLR(OPTIMIZER,
                                                     milestones=milestones,
                                                     gamma=gamma)
    loss_fuc = MSELoss().to(device)

    loss_log = []
    current_loss = float(1000)
    best_dict = MODEL.state_dict()
    best_epoch = 0
    checkpoint = 100

    w_dot = 1.0
    w_lapgt = 0.1

    for epoch in range(EPOCHS):
        logger.info('\n==>Epoch {}, gpu = {}, lr = {} ==>\n'.format(epoch, device,
                    OPTIMIZER.param_groups[0]['lr']))
        MODEL.train()

        epoch_loss = 0
        for data in tqdm(trainset_loader, total=len(trainset_loader)):
            batch_graph, batch_graph_gt = data
            batch_graph = batch_graph.to(device)
            batch_graph_gt = batch_graph_gt.to(device)

            OPTIMIZER.zero_grad()
            output = MODEL(batch_graph.x,
                           batch_graph.edge_index)

            hd_verts = batch_graph_gt.x
            output = torch.cat((hd_verts[:, :3], output), dim=-1)

            loss = 0

            # dot loss
            for i in range(batch_graph.batch_size):
                output_sub = output[batch_graph.batch == i]
                dot_id = batch_graph_gt[i].dot_id
                e_1 = output_sub[dot_id[:, 1]]-output_sub[dot_id[:, 0]]
                e_2 = output_sub[dot_id[:, 3]]-output_sub[dot_id[:, 2]]
                dot_pred = torch.sum(e_1*e_2, dim=-1)

                gt_sub = batch_graph_gt[i].x
                e_1_gt = gt_sub[dot_id[:, 1]]-gt_sub[dot_id[:, 0]]
                e_2_gt = gt_sub[dot_id[:, 3]]-gt_sub[dot_id[:, 2]]
                dot_gt = torch.sum(e_1_gt*e_2_gt, dim=-1)
                dot_loss = loss_fuc(dot_pred, dot_gt)
                loss += w_dot*dot_loss

            # laplacian loss
            edge_length = output[batch_graph.edge_index[1]
                                 ] - output[batch_graph.edge_index[0]]
            lap_loss = scatter(edge_length, batch_graph.edge_index[0],
                               dim=0, dim_size=output.size(0), reduce='sum')
            deg = degree(batch_graph.edge_index[1], output.size(
                0), dtype=output.dtype)
            lap_loss = lap_loss/deg.unsqueeze(-1)

            # laplacian loss with gt
            edge_length_gt = hd_verts[batch_graph.edge_index[1]
                                      ] - hd_verts[batch_graph.edge_index[0]]
            lap_loss_gt = scatter(edge_length_gt,
                                  batch_graph.edge_index[0],
                                  dim=0, dim_size=hd_verts.size(0),
                                  reduce='sum')
            deg_gt = degree(batch_graph.edge_index[1], hd_verts.size(
                0), dtype=hd_verts.dtype)
            lap_loss_gt = lap_loss_gt/deg_gt.unsqueeze(-1)
            lap_loss_wgt = loss_fuc(lap_loss, lap_loss_gt)

            loss += w_lapgt*lap_loss_wgt

            loss.backward()
            OPTIMIZER.step()

            epoch_loss += loss.item()

        epoch_loss /= len(trainset_loader)
        logger.info(
            ('\n==>End of epoch {}, loss {} ==>'.format(epoch, epoch_loss)))
        loss_log.append(epoch_loss)
        SCHEDULER.step()

        if epoch % checkpoint == 0:
            torch.save(MODEL.state_dict(), os.path.join(
                conf.model_path, "ep{}_{:.6f}.pth".format(epoch, epoch_loss)))

        if current_loss > epoch_loss:
            current_loss = epoch_loss
            best_dict = MODEL.state_dict()
            best_epoch = epoch

    torch.save(best_dict,
               os.path.join(conf.model_path,
                            "best_ep{}_{:.6f}.pth".format(best_epoch, current_loss)))
    torch.save(MODEL.state_dict(), os.path.join(
        conf.model_path, "last_{:.6f}.pth".format(epoch_loss)))
    logger.info("Training Finished!")
    np.save(os.path.join(conf.model_path, "loss_log.npy"),
            np.array(loss_log))
