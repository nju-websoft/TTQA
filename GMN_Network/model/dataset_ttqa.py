
import abc
import collections
import torch

import numpy as np
import dgl


"""A general Interface"""


class GraphSimilarityDataset(object):
    """Base class for all the graph similarity learning datasets.
  This class defines some common interfaces a graph similarity dataset can have,
  in particular the functions that creates iterators over pairs and triplets.
  """

    @abc.abstractmethod
    def triplets(self, batch_size):
        """Create an iterator over triplets.
    Args:
      batch_size: int, number of triplets in a batch.
    Yields:
      graphs: a `GraphData` instance.  The batch of triplets put together.  Each
        triplet has 3 graphs (x, y, z).  Here the first graph is duplicated once
        so the graphs for each triplet are ordered as (x, y, x, z) in the batch.
        The batch contains `batch_size` number of triplets, hence `4*batch_size`
        many graphs.
    """
        pass

    @abc.abstractmethod
    def pairs(self, batch_size):
        """Create an iterator over pairs.
    Args:
      batch_size: int, number of pairs in a batch.
    Yields:
      graphs: a `GraphData` instance.  The batch of pairs put together.  Each
        pair has 2 graphs (x, y).  The batch contains `batch_size` number of
        pairs, hence `2*batch_size` many graphs.
      labels: [batch_size] int labels for each pair, +1 for similar, -1 for not.
    """
        pass


"""Graph Edit Distance Task"""

class GraphMatchingKBQADataset(GraphSimilarityDataset):
    """Graph edit distance dataset."""
    def __init__(self, data_pairs_json, data_triplet_json, bert_encoder):
        """Constructor.
            Args:
                data_json:  dataset json
        """
        self._data_pairs_json = data_pairs_json
        self._data_triplet_json = data_triplet_json
        self.bert_encoder = bert_encoder

    def triplets(self, batch_size):
        """Yields batches of triplet data."""
        ptr = 0
        while ptr + batch_size <= len(self._data_triplet_json):
            samples = self._data_triplet_json[ptr:ptr + batch_size]
            positive_samples = []
            negative_samples = []
            for sample in samples:
                abstract_question = sample['abstract_question']
                g1 = sample['g1']
                g2 = sample['g2']
                g3 = sample['g3']
                positive_samples.append({'abstract_question': abstract_question, 'g1': g1, 'g2': g2})
                negative_samples.append({'abstract_question': abstract_question, 'g1': g1, 'g2': g3})

            positive_cls_list, positive_batch_graphs, positive_batch_node_to_vectors, positive_batch_edge_to_vectors = self.bert_encoder(positive_samples)
            negative_cls_list, negative_batch_graphs, negative_batch_node_to_vectors, negative_batch_edge_to_vectors = self.bert_encoder(negative_samples)
            assert len(positive_batch_graphs) == len(negative_batch_graphs)
            assert len(positive_batch_node_to_vectors) == len(negative_batch_node_to_vectors)
            assert len(positive_batch_edge_to_vectors) == len(negative_batch_edge_to_vectors)

            cls_list = []
            batch_graphs = []
            batch_node_to_vectors = []
            batch_edge_to_vectors = []
            for index in range(len(positive_batch_graphs)):
                cls_list.append(positive_cls_list[index])
                cls_list.append(negative_cls_list[index])
                batch_graphs.append(positive_batch_graphs[index])
                batch_graphs.append(negative_batch_graphs[index])
                batch_node_to_vectors.append(positive_batch_node_to_vectors[index])
                batch_node_to_vectors.append(negative_batch_node_to_vectors[index])
                batch_edge_to_vectors.append(positive_batch_edge_to_vectors[index])
                batch_edge_to_vectors.append(negative_batch_edge_to_vectors[index])

            # packed_batch = self._pack_batch(batch_graphs,
            #                                 batch_node_to_vectors=batch_node_to_vectors,
            #                                 batch_edge_to_vectors=batch_edge_to_vectors)
            packed_batch = self._pack_batch_gnn(cls_list=cls_list, graphs=batch_graphs,
                                            batch_node_to_vectors=batch_node_to_vectors,
                                            batch_edge_to_vectors=batch_edge_to_vectors)
            yield packed_batch
            ptr += batch_size

    def pairs(self, batch_size):
        """Yield pairs and labels."""
        ptr = 0
        while ptr + batch_size <= len(self._data_pairs_json):
            samples = self._data_pairs_json[ptr:ptr + batch_size]
            batch_labels = []
            for sample in samples:
                batch_labels.append(1 if sample['label'] == 1 else -1)
            batch_labels = np.array(batch_labels, dtype=np.int32)
            cls_list, batch_graphs, batch_node_to_vectors, batch_edge_to_vectors = self.bert_encoder(samples)
            packed_batch = self._pack_batch_gnn(cls_list=cls_list, graphs=batch_graphs,
                                            batch_node_to_vectors=batch_node_to_vectors,
                                            batch_edge_to_vectors=batch_edge_to_vectors)
            yield packed_batch, batch_labels
            ptr += batch_size

    def _pack_batch(self, graphs, batch_node_to_vectors, batch_edge_to_vectors):

        GraphData = collections.namedtuple('GraphData', [
            'from_idx',
            'to_idx',
            'node_features',
            'edge_features',
            'graph_idx',
            'n_graphs'])

        """"""
        # graphs = np.array(graphs, dtype=nx.DiGraph)
        # graphs = graphs.flatten()
        """"""
        graphs_new = []
        for g1, g2 in graphs:
            graphs_new.append(g1)
            graphs_new.append(g2)
        graphs = graphs_new
        """"""

        node_to_vectors = np.array(batch_node_to_vectors).flatten()
        edge_to_vectors = np.array(batch_edge_to_vectors).flatten()
        from_idx = []
        to_idx = []
        node_features = []
        edge_features = []
        graph_idx = []
        n_total_nodes = 0
        n_total_edges = 0
        for i, g in enumerate(graphs):
            n_nodes = g.number_of_nodes()
            n_edges = g.number_of_edges()
            new_edges = []
            new_nodes = []
            for node in g.nodes():
                new_nodes.append(node)
            new_nodes.sort()
            for new_node in new_nodes:
                node_features.append(node_to_vectors[i][new_node])
            for edge in g.edges():
                edge_features.append(edge_to_vectors[i][edge])
                new_edges.append(edge)
            edges = np.array(new_edges, dtype=np.int32)
            from_idx.append(edges[:, 0] + n_total_nodes)
            to_idx.append(edges[:, 1] + n_total_nodes)
            graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)
            n_total_nodes += n_nodes
            n_total_edges += n_edges

        return GraphData(
            from_idx=np.concatenate(from_idx, axis=0),
            to_idx=np.concatenate(to_idx, axis=0),
            node_features=torch.stack(node_features),
            edge_features=torch.stack(edge_features),
            graph_idx=np.concatenate(graph_idx, axis=0),
            n_graphs=len(graphs))

    def _pack_batch_gnn(self, cls_list, graphs, batch_node_to_vectors, batch_edge_to_vectors):
        GraphData = collections.namedtuple('GraphData', [
            'node_features',
            'edge_features',
            'graphs',
            'graph_idx',
            'cls_info'
        ])

        graphs_new = []
        for g1, g2 in graphs:
            graphs_new.append(g1)
            graphs_new.append(g2)
        graphs = graphs_new
        # cls_list_new = []
        # for cls_1, cls_2 in cls_list:
        #     cls_list_new.append(cls_1)
        #     cls_list_new.append(cls_2)
        # cls_list = cls_list_new

        node_to_vectors = np.array(batch_node_to_vectors).flatten()
        edge_to_vectors = np.array(batch_edge_to_vectors).flatten()
        # from_idx = []
        # to_idx = []
        node_features = []
        edge_features = []
        graph_idx = []
        # n_total_nodes = 0
        # n_total_edges = 0
        for i, g in enumerate(graphs):
            n_nodes = g.number_of_nodes()
            # n_edges = g.number_of_edges()
            # new_edges = []
            new_nodes = []
            for node in g.nodes():
                new_nodes.append(node)
            new_nodes.sort()
            for new_node in new_nodes:
                node_features.append(node_to_vectors[i][new_node])
            for edge in g.edges():
                edge_features.append(edge_to_vectors[i][edge])
                # new_edges.append(edge)
            # edges = np.array(new_edges, dtype=np.int32)
            # from_idx.append(edges[:, 0] + n_total_nodes)
            # to_idx.append(edges[:, 1] + n_total_nodes)
            graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)
            # n_total_nodes += n_nodes
            # n_total_edges += n_edges

        dgl_graphs_list = []
        for graph in graphs:
            dgl_graph = dgl.from_networkx(graph)
            dgl_graph = dgl.to_bidirected(dgl_graph)
            dgl_graph = dgl.add_self_loop(dgl_graph)
            dgl_graphs_list.append(dgl_graph)
        batch_graphs = dgl.batch(dgl_graphs_list)
        return GraphData(
            # from_idx=np.concatenate(from_idx, axis=0),
            # to_idx=np.concatenate(to_idx, axis=0),
            node_features=torch.stack(node_features),
            edge_features=torch.stack(edge_features),
            graph_idx=np.concatenate(graph_idx, axis=0),
            graphs=batch_graphs,
            cls_info=torch.stack(cls_list),
        )


class FixedGraphMatchingKBQADataset(GraphMatchingKBQADataset):
    """A fixed dataset of pairs or triplets for the graph edit distance task.
  This dataset can be used for evaluation.
  """

    def __init__(self, data_pairs_json, data_triplet_json, bert_encoder):
        super(FixedGraphMatchingKBQADataset, self).__init__(data_pairs_json, data_triplet_json, bert_encoder)

    def triplets(self, batch_size):
        """Yield triplets."""
        ptr = 0
        while ptr + batch_size <= len(self._data_triplet_json):
            samples = self._data_triplet_json[ptr:ptr + batch_size]
            positive_samples = []
            negative_samples = []
            for sample in samples:
                abstract_question = sample['abstract_question']
                g1 = sample['g1']
                g2 = sample['g2']
                g3 = sample['g3']
                positive_samples.append({'abstract_question': abstract_question, 'g1': g1, 'g2': g2})
                negative_samples.append({'abstract_question': abstract_question, 'g1': g1, 'g2': g3})

            positive_cls_list, positive_batch_graphs, positive_batch_node_to_vectors, positive_batch_edge_to_vectors = self.bert_encoder(positive_samples)
            negative_cls_list, negative_batch_graphs, negative_batch_node_to_vectors, negative_batch_edge_to_vectors = self.bert_encoder(negative_samples)
            assert len(positive_batch_graphs) == len(negative_batch_graphs)
            assert len(positive_batch_node_to_vectors) == len(negative_batch_node_to_vectors)
            assert len(positive_batch_edge_to_vectors) == len(negative_batch_edge_to_vectors)

            cls_list = []
            batch_graphs = []
            batch_node_to_vectors = []
            batch_edge_to_vectors = []
            for index in range(len(positive_batch_graphs)):
                cls_list.append(positive_cls_list[index])
                cls_list.append(negative_cls_list[index])
                batch_graphs.append(positive_batch_graphs[index])
                batch_graphs.append(negative_batch_graphs[index])
                batch_node_to_vectors.append(positive_batch_node_to_vectors[index])
                batch_node_to_vectors.append(negative_batch_node_to_vectors[index])
                batch_edge_to_vectors.append(positive_batch_edge_to_vectors[index])
                batch_edge_to_vectors.append(negative_batch_edge_to_vectors[index])

            packed_batch = self._pack_batch_gnn(cls_list=cls_list, graphs=batch_graphs,
                                            batch_node_to_vectors=batch_node_to_vectors,
                                            batch_edge_to_vectors=batch_edge_to_vectors)
            yield packed_batch
            ptr += batch_size

    def pairs(self, batch_size):
        """Yield pairs and labels."""
        ptr = 0
        while ptr + batch_size <= len(self._data_pairs_json):
            samples = self._data_pairs_json[ptr:ptr + batch_size]
            batch_labels = []
            for sample in samples:
                batch_labels.append(1 if sample['label'] == 1 else -1)
            batch_labels = np.array(batch_labels, dtype=np.int32)
            cls_list, batch_graphs, batch_node_to_vectors, batch_edge_to_vectors = self.bert_encoder(samples)
            packed_batch = self._pack_batch_gnn(cls_list=cls_list, graphs=batch_graphs,
                                            batch_node_to_vectors=batch_node_to_vectors,
                                            batch_edge_to_vectors=batch_edge_to_vectors)
            yield packed_batch, batch_labels
            ptr += batch_size


