


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

#定义Mydataset继承自Dataset,并重写__getitem__和__len__
class TTQAdataset(Dataset):
    def __init__(self, all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                       all_graph1_nodeid_to_inputids, all_blag_to_uri_dict, all_graph1_nx,
                       all_graph2_nodeid_to_inputids, all_graph2_nx):
        super(TTQAdataset, self).__init__()
        self.num = len(all_input_ids) #生成多少个点（多少个数据）

        self.all_input_idx = all_input_ids
        self.all_input_mask = all_input_mask
        self.all_segment_ids = all_segment_ids
        self.all_label_ids = all_label_ids

        self.all_graph1_nodeid_to_inputids = all_graph1_nodeid_to_inputids
        self.all_blag_to_uri_dict = all_blag_to_uri_dict
        self.all_graph1_nx = all_graph1_nx
        self.all_graph2_nodeid_to_inputids = all_graph2_nodeid_to_inputids
        self.all_graph2_nx = all_graph2_nx


    # indexing
    def __getitem__(self, index):
        input_idx = self.all_input_idx[index]
        input_mask = self.all_input_mask[index]
        segment_ids = self.all_segment_ids[index]
        label_ids = self.all_label_ids[index]

        graph1_nodeid_to_inputids = self.all_graph1_nodeid_to_inputids[index]
        blag_to_uri_dict = self.all_blag_to_uri_dict[index]
        graph1_nx = self.all_graph1_nx[index]
        graph2_nodeid_to_inputids = self.all_graph2_nodeid_to_inputids[index]
        graph2_nx = self.all_graph2_nx[index]

        return (input_idx, input_mask, segment_ids, label_ids, graph1_nodeid_to_inputids, blag_to_uri_dict, graph1_nx, graph2_nodeid_to_inputids, graph2_nx)

    def __len__(self):
        return self.num


from collections import OrderedDict

class TTQADataLoader(DataLoader):
    def __init__(self, dataset, device, eval=False, **kwargs):
        if kwargs.get('collate_fn', None) is None:
            kwargs['collate_fn'] = self._collate_fn

        self.eval = eval
        self.device = device
        super().__init__(dataset, **kwargs)

    def _collate_fn(self, batch_data):
        """
            input_ids, input_mask, segment_ids, label_ids,\
            graph1_nodeid_to_inputids, all_blag_to_uri_dict, all_graph1_nx,\
            graph2_nodeid_to_inputids, all_graph2_nx = batch
        :param batch_data:
        :return:
        """

        # generate batch
        batch_size = len(batch_data)
        batch = list(zip(*batch_data))
        assert len(batch) == 9

        tensorized = OrderedDict()
        tensorized['input_ids'] = torch.LongTensor(batch[0]).to(self.device)
        tensorized['input_mask'] = torch.LongTensor(batch[1]).to(self.device)
        tensorized['segment_ids'] = torch.LongTensor(batch[2]).to(self.device)
        tensorized['label_ids'] = torch.LongTensor(batch[3]).to(self.device)

        tensorized['graph1_nodeid_to_inputids'] = batch[4]
        tensorized['all_blag_to_uri_dict'] = batch[5]
        tensorized['all_graph1_nx'] = batch[6]

        tensorized['graph2_nodeid_to_inputids'] = batch[7]
        tensorized['all_graph2_nx'] = batch[8]

        return tensorized

