import heapq
import torch
import torch.nn.functional as F


class BinaryIndexTree:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.depth = max(1, (vocab_size - 1).bit_length())
        self.max_path_length = self.depth
        self.num_internal_nodes = 2 ** self.depth - 1
        
        self.paths = {}
        self.codes = {}
        for idx in range(vocab_size):
            self.paths[idx], self.codes[idx] = self._path_and_targets(idx)
    
    def _targets_for_index(self, word_index):
        return format(word_index, f'0{self.depth}b')
    
    def _node_id_from_prefix(self, prefix_bits):
        if not prefix_bits:
            return 0
        d = len(prefix_bits)
        p = int(prefix_bits, 2)
        return (2 ** d - 1) + p
    
    def _path_and_targets(self, word_index):
        bits = self._targets_for_index(word_index)
        path = []
        targets = []
        prefix = ""
        for bit in bits:
            node_id = self._node_id_from_prefix(prefix)
            path.append(node_id)
            targets.append(int(bit))
            prefix += bit
        return path, targets
    
    def __call__(self, context_word):
        device = context_word.device
        context_word = context_word.detach().cpu().view(-1).tolist()
        paths = []
        targets = []
        masks = []
        for word_index in context_word:
            path = self.paths[int(word_index)]
            target = self.codes[int(word_index)]
            padding = self.max_path_length - len(path)
            paths.append(path + [0] * padding)
            targets.append(target + [0] * padding)
            masks.append([1.0] * len(path) + [0.0] * padding)
        return {
            'path': torch.tensor(paths, dtype=torch.long, device=device),
            'targets': torch.tensor(targets, dtype=torch.float32, device=device),
            'mask': torch.tensor(masks, dtype=torch.float32, device=device),
        }


class HierarchicalSoftmaxTargets:
    def __init__(self, word_counts, word_to_index):
        vocab_size = len(word_to_index)
        self.tree = BinaryIndexTree(vocab_size)
        self.paths = self.tree.paths
        self.codes = self.tree.codes
        self.num_internal_nodes = self.tree.num_internal_nodes
        self.max_path_length = self.tree.max_path_length

    def __call__(self, context_word):
        return self.tree(context_word)


class HierarchicalSoftmaxLoss(torch.nn.Module):
    def __init__(self, model, targets):
        super().__init__()
        self.model = model
        self.targets = targets

    def forward(self, batch):
        target_tensors = self.targets(batch['data']['context_word'])
        batch['data'].update(target_tensors)
        embedding = batch['signals']['embedding']
        node_vectors = self.model.decoder(batch['data']['path'])
        logits = torch.einsum('bd,bld->bl', embedding, node_vectors)
        batch['signals']['logits'] = logits
        batch['signals']['probabilities'] = torch.sigmoid(logits)
        batch['postprocessed']['code'] = (batch['signals']['probabilities'] >= 0.5).long()
        per_node_loss = F.binary_cross_entropy_with_logits(
            logits,
            batch['data']['code'],
            reduction='none',
        )
        masked_loss = per_node_loss * batch['data']['mask']
        return masked_loss.sum() / batch['data']['mask'].sum().clamp_min(1.0)


class HierarchicalSoftmax(torch.nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        
        self.tree = BinaryIndexTree(vocab_size)
        self.max_path_length = self.tree.max_path_length 
        self.num_internal_nodes = self.tree.num_internal_nodes
        
        self.decoder = torch.nn.Embedding(self.num_internal_nodes, embedding_dim)
    
    def forward(self, center_embeddings, context_words):
        tree_output = self.tree(context_words)
        path = tree_output['path']
        targets = tree_output['targets']
        mask = tree_output['mask']
        
        node_vectors = self.decoder(path)
        logits = torch.einsum('bd,bld->bl', center_embeddings, node_vectors)
        probabilities = torch.sigmoid(logits)
        
        target_probabilities = torch.where(
            targets.bool(),
            probabilities,
            1.0 - probabilities
        )
        total_probability = (target_probabilities * mask).prod(dim=1)
        
        per_node_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none'
        )
        masked_loss = (per_node_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        
        return {
            'path': path,
            'targets': targets,
            'mask': mask,  
            'logits': logits,
            'probabilities': probabilities,
            'total_probability': total_probability,
            'loss': masked_loss,}
        


class Word2VecHierarchicalSoftmax(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_internal_nodes):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.num_internal_nodes = int(num_internal_nodes)
        self.encoder = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.decoder = torch.nn.Embedding(self.num_internal_nodes, self.embedding_dim)

    def __forward_kernel(self, center_word, path):
        embedding = self.encoder(center_word)
        node_vectors = self.decoder(path)
        logits = torch.einsum('bd,bld->bl', embedding, node_vectors)
        return embedding, logits

    def forward(self, batch):
        if 'signals' not in batch:
            batch['signals'] = {
                'embedding': self.encoder(batch['data']['center_word']),
            }
            batch['postprocessed'] = {}
        return batch
    
class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        
        self.encoder = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        
        torch.nn.init.normal_(self.encoder.weight, mean=0.0, std=0.02)
        
        self.hierarchical_softmax = HierarchicalSoftmax(self.embedding_dim, self.vocab_size)
        
        self.decoder = self.hierarchical_softmax.decoder
        self.num_internal_nodes = self.hierarchical_softmax.num_internal_nodes
        
        self.max_path_length = self.hierarchical_softmax.tree.max_path_length
    
    def forward(self, batch):
        center_word = batch['data']['center_word']
        
        embedding = self.encoder(center_word)
        
        batch['signals'] = {'embedding': embedding}
        batch['postprocessed'] = {}
        
        if 'context_word' in batch['data']:
            hsoftmax_output = self.hierarchical_softmax(embedding, batch['data']['context_word'])
            
            batch['data']['path'] = hsoftmax_output['path']
            batch['data']['targets'] = hsoftmax_output['targets']
            batch['data']['mask'] = hsoftmax_output['mask']
            
            batch['signals']['logits'] = hsoftmax_output['logits']
            batch['signals']['probabilities'] = hsoftmax_output['probabilities']
            batch['signals']['target_probabilities'] = hsoftmax_output['target_probabilities'] if 'target_probabilities' in hsoftmax_output else None
            batch['signals']['total_probability'] = hsoftmax_output['total_probability']
            batch['signals']['loss'] = hsoftmax_output['loss']
            
            batch['postprocessed']['targets'] = (hsoftmax_output['probabilities'] >= 0.5).long()
        
        return batch