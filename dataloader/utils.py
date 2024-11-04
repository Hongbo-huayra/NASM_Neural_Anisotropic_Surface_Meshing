import os.path as path
import torch


def parse_id_list(file_path):
    id_list = []
    with open(file_path, 'r') as f:
        for line in f:
            obj_id = int(line.strip())
            id_list.append(obj_id)
    return id_list


def get_obj_path_list(id_list, root_path, ext):
    path_list = []
    for obj_id in id_list:
        obj_path = path.join(root_path,
                             str(obj_id),
                             "{}{}".format(obj_id, ext))
        path_list.append(obj_path)
    return path_list


def load_hd_mesh(mesh_path):
    hd_verts = []
    faces = []
    with open(mesh_path, 'r') as f:
        for line in f:
            if line.startswith("v "):
                values = [float(x) for x in line.split()[1:]]
                hd_verts.append(values)
            elif line.startswith("f "):
                values = [int(x) - 1 for x in line.split()[1:]]
                faces.append(values)
    hd_verts = torch.tensor(hd_verts, dtype=torch.float)
    faces = torch.tensor(faces, dtype=torch.long)
    return hd_verts, faces


def save_hd_obj(mesh_path, verts, faces):
    with open(mesh_path, 'w') as f:
        for v in verts:
            vert = 'v '
            for i in v:
                vert += str(i) + ' '
            f.write(vert + '\n')
        for face in faces:
            f.write('f {} {} {}\n'.format(
                face[0] + 1, face[1] + 1, face[2] + 1))


def gen_dot_id(faces):
    cos_id_list = []
    for f_i in faces:
        cos_id_list.append([f_i[0], f_i[1], f_i[0], f_i[2]])
        cos_id_list.append([f_i[1], f_i[0], f_i[1], f_i[2]])
        cos_id_list.append([f_i[2], f_i[0], f_i[2], f_i[1]])
    cos_id_list = torch.tensor(cos_id_list, dtype=torch.long)
    return cos_id_list
