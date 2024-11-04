from . import utils as ut


class DataConfigParser:
    def __init__(self, config, type) -> None:
        self._type = type
        self._id_type = config.get("eval_type", "list")

        self._id_list = list()
        self._ori_obj_pathlist = list()
        self._metric_pathlist = list()
        self._config = config

    def get_id_list(self):
        if self._id_type == "list":
            self._id_list = []
            for id_list in self._config["id_list"]:
                self._id_list.extend(ut.parse_id_list(id_list))

        elif self._id_type == "model":
            self._id_list = self._config.eval_mesh_config.mesh_id

        return self._id_list

    def get_augment_list(self, pathlist: list, aug_ext_dict: dict):
        augment_type = self._config.get("augment", [])
        if len(augment_type) > 0:
            for aug_type in augment_type:
                aug_ext = aug_ext_dict[aug_type]
                pathlist.extend(
                    ut.get_obj_path_list(self._id_list,
                                         self._config["root_path"],
                                         aug_ext))

    def get_obj_path_list(self):
        if self._id_type == "list":
            if len(self._id_list) == 0:
                self.get_id_list(self._config)
            self._ori_obj_pathlist = ut.get_obj_path_list(self._id_list,
                                                          self._config["root_path"],
                                                          self._config["ori_obj_ext"])
        elif self._id_type == "model":
            self._ori_obj_pathlist = self._config.eval_mesh_config.mesh_path

        if self._config["is_augment"]:
            self.get_augment_list(self._ori_obj_pathlist,
                                  self._config["augment_ext"])

        return self._ori_obj_pathlist

    def get_gt_path_list(self):
        if len(self._id_list) == 0:
            self.get_id_list(self._config)

        if self._id_type == "list":
            self._hd_obj_pathlist = ut.get_obj_path_list(self._id_list,
                                                         self._config["root_path"],
                                                         self._config["hd_obj_ext"])
        elif self._id_type == "model":
            self._hd_obj_pathlist = ut.get_obj_path_list(self._id_list,
                                                         self._config.eval_mesh_config.root_path,
                                                         self._config.eval_mesh_config.gt_ext)

        if self._config["is_augment"]:
            self.get_augment_list(self._hd_obj_pathlist,
                                  self._config["augment_gt_ext"])

        return self._hd_obj_pathlist

    def get_metric_path_list(self, augment=True):
        if len(self._id_list) == 0:
            self.get_id_list(self._config)
        self._metric_pathlist = ut.get_obj_path_list(self._id_list,
                                                     self._config["root_path"],
                                                     self._config["m_ext"])
        if augment:
            self.get_augment_list(self._metric_pathlist,
                                  self._config["m_augment_ext"])

        return self._metric_pathlist
