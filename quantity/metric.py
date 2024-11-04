import trimesh
import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from pytorch3d import loss


def hausdorff_distance(ori_sample, pred_sample):
    dist1 = directed_hausdorff(ori_sample, pred_sample)[0]
    dist2 = directed_hausdorff(pred_sample, ori_sample)[0]
    return max(dist1, dist2)


def chamfer_distance_pt(ori_sample,
                        pred_sample,
                        ):
    ori_sample = torch.tensor(ori_sample, dtype=torch.float32).cuda()
    pred_sample = torch.tensor(pred_sample, dtype=torch.float32).cuda()

    ori_sample = ori_sample.unsqueeze(0)
    pred_sample = pred_sample.unsqueeze(0)

    dist, _ = loss.chamfer_distance(ori_sample, pred_sample)
    return dist.cpu().numpy().item()


def f1_loss(ori_sample, pred_sample, ori_tree, pred_tree, threshold=0.004):
    pred_dist, _ = ori_tree.query(pred_sample, k=1)
    precision = (pred_dist < threshold).sum().item() / float(len(pred_sample))

    ori_dist, _ = pred_tree.query(ori_sample, k=1)
    recall = (ori_dist < threshold).sum().item() / float(len(pred_sample))

    f1 = 2*recall*precision/(recall+precision) if recall+precision > 0 else 0
    return f1


def normal_consistency(ori_sample, pred_sample, ori_sample_normal, pred_sample_normal, ori_tree, pred_tree):
    ori_sample_normal_gpu = torch.tensor(ori_sample_normal).cuda()
    pred_sample_normal_gpu = torch.tensor(pred_sample_normal).cuda()

    _, pred_ids = pred_tree.query(ori_sample, k=1)
    pred_neigh_normals = pred_sample_normal[pred_ids]
    pred_neigh_normals_gpu = torch.tensor(pred_neigh_normals).squeeze(1).cuda()

    ori2pred_dot = torch.abs(
        torch.sum(ori_sample_normal_gpu*pred_neigh_normals_gpu, dim=-1))
    ori2pred_nc = ori2pred_dot.mean()

    _, ori_ids = ori_tree.query(pred_sample, k=1)
    ori_neigh_normals = ori_sample_normal[ori_ids]
    ori_neigh_normals_gpu = torch.tensor(ori_neigh_normals).squeeze(1).cuda()

    pred2ori_dot = torch.abs(
        torch.sum(pred_sample_normal_gpu*ori_neigh_normals_gpu, dim=-1))
    pred2ori_nc = pred2ori_dot.mean()

    nc = (ori2pred_nc + pred2ori_nc)/2
    return nc.cpu().numpy()


def sample_on_edge(mesh_samples, sample_normal, mesh_tree, query_radius=0.004, dot_threshold=0.2):
    ids_list = mesh_tree.query_radius(mesh_samples, r=query_radius)
    flags = np.zeros([len(mesh_samples)], bool)
    for p in range(len(mesh_samples)):
        inds = ids_list[p]
        if len(inds) > 0:
            this_normals = sample_normal[p:p+1]
            neighbor_normals = sample_normal[inds]
            dotproduct = np.abs(
                np.sum(this_normals*neighbor_normals, axis=1))
            if np.any(dotproduct < dot_threshold):
                flags[p] = True
    edge_points = np.ascontiguousarray(mesh_samples[flags])
    return edge_points


def sample_on_edge_ponq(mesh, angle_treshold, N_sampling):

    sharp = mesh.face_adjacency_angles > np.radians(angle_treshold)
    sharp_edges = mesh.face_adjacency_edges[sharp]

    if len(sharp_edges) == 0:
        return np.array([])
    v = mesh.vertices

    edge_length = np.sqrt(
        ((v[sharp_edges[:, 1]]-v[sharp_edges[:, 0]])**2).sum(-1))
    selected_edges = np.random.choice(
        len(edge_length), N_sampling, p=edge_length/edge_length.sum())
    lambdas = np.random.rand(len(selected_edges))[:, None]
    sampled_points = v[sharp_edges[selected_edges][:, 1]] * \
        lambdas + v[sharp_edges[selected_edges][:, 0]]*(1-lambdas)
    return sampled_points


def anisotropic_quality_avg(ori_mesh, ori_cur, eval_mesh):
    eval2ori_v, _, tri_ids = trimesh.proximity.closest_point(
        ori_mesh, eval_mesh.vertices)

    # metric by triangle
    ori_vert_ids = ori_mesh.faces[tri_ids]
    eval2ori_v_cur = ori_cur[ori_vert_ids]
    eval2ori_v_cur = np.mean(eval2ori_v_cur, axis=1)

    eval_tri_cur = eval2ori_v_cur[eval_mesh.faces]
    eval_tri_cur = np.mean(eval_tri_cur, axis=1)

    for i in range(len(eval_tri_cur)):
        tmp_mat = np.identity(3)
        try:
            tmp_mat = np.linalg.cholesky(eval_tri_cur[i])
        except:
            pass
        eval_tri_cur[i] = tmp_mat

    eval_tri_cur = np.asarray(eval_tri_cur)

    tri_verts = eval_mesh.vertices[eval_mesh.faces]
    ev_0 = tri_verts[:, 1] - tri_verts[:, 0]
    ev_1 = tri_verts[:, 2] - tri_verts[:, 0]
    eval_ev_0 = np.matmul(eval_tri_cur, ev_0[:, :, np.newaxis])
    eval_ev_1 = np.matmul(eval_tri_cur, ev_1[:, :, np.newaxis])

    eval_tri_verts = np.stack([tri_verts[:, 0], tri_verts[:, 0] +
                               eval_ev_0[:, :, 0], tri_verts[:, 0] + eval_ev_1[:, :, 0]], axis=1)

    eval_tri_area = trimesh.triangles.area(eval_tri_verts)
    edge_0 = np.linalg.norm(
        eval_tri_verts[:, 0] - eval_tri_verts[:, 1], axis=1)
    edge_1 = np.linalg.norm(
        eval_tri_verts[:, 0] - eval_tri_verts[:, 2], axis=1)
    edge_2 = np.linalg.norm(
        eval_tri_verts[:, 1] - eval_tri_verts[:, 2], axis=1)
    eval_tri_perimeter = edge_0 + edge_1 + edge_2

    eval_tri_le = np.max([edge_0, edge_1, edge_2], axis=0)

    denom = 0.5*eval_tri_perimeter*eval_tri_le
    G = 2*np.sqrt(3)*eval_tri_area/denom

    avg_G = np.mean(G)
    min_G = np.min(G)
    return G, avg_G, min_G
