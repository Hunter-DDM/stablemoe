# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
import sys
import os
import re
import numpy as np
import json
from fairseq import utils
from fairseq.distributed import utils as distributed_utils
from fairseq.modules.layer_norm import LayerNorm
from omegaconf import II
from transformers import GPTNeoModel, GPTNeoConfig
import torch.nn.functional as F
import torch.distributed as dist


class DenseBaseLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense_ffn = nn.Sequential(*([DenseMoESublayer(args) for _ in range(args.moe_sublayers * args.deepx)]))

    def forward(self, input_features, *args, **kwargs):
        return self.dense_ffn(input_features), None, None


class DenseMoESublayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        self.norm = LayerNorm(args.decoder_embed_dim, export=False)
        self.ff1 = torch.nn.Linear(args.decoder_embed_dim, args.decoder_ffn_embed_dim * args.widex)
        self.ff2 = torch.nn.Linear(args.decoder_ffn_embed_dim * args.widex, args.decoder_embed_dim)
        self.ff2.weight.data.zero_()

    def forward(self, xs):
        return xs + self.ff2(self.activation_fn(self.ff1(self.norm(xs))))


class BaseLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_workers = distributed_utils.get_data_parallel_world_size()
        routing_dim = args.decoder_embed_dim
        distill_routing_dim = 50

        if args.distill_assignment:
            if args.distilled_model == 'wordemb':
                self.routing_emb = nn.Embedding(args.vocab_size, distill_routing_dim, padding_idx=args.dict_pad_idx)
            elif args.distilled_model == 'bigram_emb':
                self.routing_emb = nn.Embedding(args.vocab_size, distill_routing_dim, padding_idx=args.dict_pad_idx)
            elif args.distilled_model == 'cnn':
                self.routing_emb = nn.Embedding(args.vocab_size, distill_routing_dim, padding_idx=args.dict_pad_idx)
                routing_kernel = torch.rand(5) / 0.5 * 0.447 - 0.447
                self.register_parameter("routing_kernel", torch.nn.Parameter(routing_kernel))
            elif 'trm' in args.distilled_model:
                num_layers = int(re.findall(r'trm(\d+)l', args.distilled_model)[0])
                config = GPTNeoConfig.from_pretrained(args.hf_plm_dir)
                config.vocab_size = args.vocab_size
                config.attention_types = [[["global"], num_layers]]
                config.bos_token_id = args.dict_bos_idx
                config.eos_token_id = args.dict_eos_idx
                config.num_layers = num_layers
                config.max_position_embeddings = 1024
                config.num_heads = 12
                config.activation_function = 'relu'
                distill_routing_dim = config.hidden_size
                self.routing_model = GPTNeoModel(config)
            else:
                raise Exception('Now only support wordemb, bigram_emb, cnn, trmxl.')
            distill_expert_centroids = torch.empty(self.num_workers, distill_routing_dim)
            torch.nn.init.orthogonal_(distill_expert_centroids, gain=0.1)
            self.register_parameter("distill_expert_centroids", torch.nn.Parameter(distill_expert_centroids))

        expert_centroids = torch.empty(self.num_workers, routing_dim)
        torch.nn.init.orthogonal_(expert_centroids, gain=0.1)
        self.register_parameter("expert_centroids", torch.nn.Parameter(expert_centroids))
        self.expert_network = nn.Sequential(*([MoESublayer(args) for _ in range(args.moe_sublayers)]))
        self.expert_id = distributed_utils.get_data_parallel_rank()
        # If using GA, then shuffle is not necessary
        # self.shuffle = args.train_token_shuffle and args.assignment_algorithm != 'GA'
        self.shuffle = args.train_token_shuffle

        # record num_updates, used for two-stage training
        self.num_updates = 0

        if args.assignment_algorithm == 'BA':
            self.assignment_algorithm = self.balanced_assignment
            self.cpp = self.load_assignment()
        elif args.assignment_algorithm == 'GBA':
            self.assignment_algorithm = self.greedy_balanced_assignment
        elif args.assignment_algorithm == 'GA':
            self.assignment_algorithm = self.greedy_assignment

        if args.balance_loss == 'balance':
            self.balance_func = self.calculate_balance_loss
        elif args.balance_loss == 'subopt_margin':
            self.balance_func = self.calculate_subopt_balance_margin_loss
        elif args.balance_loss == 'subopt':
            self.balance_func = self.calculate_subopt_balance_loss

        # Add a special attribute to the expert parameters, so we know not to sync their gradients
        for param in self.expert_network.parameters():
            param.expert = True

    def is_stage2(self):
        return self.num_updates >= self.args.two_stage_updates

    def routing_wrapper(self, routing_ids):
        if self.args.distilled_model == 'wordemb':
            return self.routing_emb(routing_ids)
        elif self.args.distilled_model == 'cnn':
            emb = self.routing_emb(routing_ids)  # len, batch, hidden
            seq_len = emb.shape[0]
            padding_emb = torch.cat((torch.zeros([4] + list(emb.shape[1:]), device=emb.device, dtype=emb.dtype), emb), dim=0)  # len + 4, batch, hidden
            stack_emb = torch.stack((padding_emb[4:4 + seq_len, :, :],
                                    padding_emb[3:3 + seq_len, :, :],
                                    padding_emb[2:2 + seq_len, :, :],
                                    padding_emb[1:1 + seq_len, :, :],
                                    padding_emb[0:0 + seq_len, :, :]
                                    ), dim=3)  # len, batch, hidden, 5
            cnn_emb = stack_emb.matmul(self.routing_kernel)
            return cnn_emb
        elif self.args.distilled_model == 'bigram_emb':
            emb = self.routing_emb(routing_ids)  # len, batch, hidden
            pre_emb = torch.cat((emb[0:1, :, :], emb[:-1, :, :]), dim=0)
            bigram_emb = (emb + pre_emb) / 2.0
            return bigram_emb
        elif 'trm' in self.args.distilled_model:
            return self.routing_model(routing_ids.transpose(0, 1))[0].transpose(0, 1)

    def forward(self, input_features, *args, **kwargs):
        assert kwargs['input_ids'].shape == input_features.shape[:-1]  # len, batch, hidden

        if self.args.distill_assignment:
            routing_ids = kwargs['input_ids']
            routing_features = self.routing_wrapper(routing_ids)
            routing_features = routing_features.reshape(-1, routing_features.size(-1))
        features = input_features.reshape(-1, input_features.size(-1))
        tpe = features.shape[0]
        is_training = input_features.requires_grad

        if self.shuffle and is_training:
            # Send each token to a random worker, to break correlations within the batch
            shuffle_sort = torch.randperm(features.size(0), device=features.device)
            features = All2AllDDM.apply(features[shuffle_sort])
            if self.args.distill_assignment:
                routing_features = All2AllDDM.apply(routing_features[shuffle_sort])

        with torch.no_grad():
            # Compute similarity of each token to each expert for routing, and make the affinities finite
            if self.args.distill_assignment and (self.is_stage2() or not is_training):
                token_expert_affinities = routing_features.matmul(self.distill_expert_centroids.transpose(0, 1))
            else:
                token_expert_affinities = features.matmul(self.expert_centroids.transpose(0, 1))
            token_expert_affinities = self.make_finite(token_expert_affinities)

        # calculate distill loss
        if self.args.distill_assignment and not self.is_stage2() and is_training:
            distill_token_expert_affinities = routing_features.matmul(self.distill_expert_centroids.transpose(0, 1))  # len, ne
            distill_target = token_expert_affinities.max(dim=1).indices  # len
            distill_loss = F.cross_entropy(distill_token_expert_affinities, distill_target, reduction='sum')
        else:
            distill_loss = 0

        # Compute which token goes to which expert
        if self.is_stage2():
            sort_by_expert, input_splits, output_splits = self.greedy_assignment(token_expert_affinities)
        else:
            sort_by_expert, input_splits, output_splits = self.assignment_algorithm(token_expert_affinities) if is_training \
                                                    else self.greedy_assignment(token_expert_affinities)

        # Swap these tokens for the right ones for our expert
        routed_features = All2AllDDM.apply(features[sort_by_expert], output_splits, input_splits)

        # calculate balance loss
        if not is_training or self.args.balance_loss is None or self.is_stage2():
            balance_loss = 0
        else:
            balance_loss = self.balance_func(routed_features, tpe)

        if routed_features.size(0) > 0:
            capacity = int(tpe * self.args.capacity_factor)
            if routed_features.size(0) <= capacity:
                alpha = torch.sigmoid(routed_features.mv(self.expert_centroids[self.expert_id])).unsqueeze(1)
                routed_features = alpha * self.expert_network(routed_features) + (1 - alpha) * routed_features
            else:
                alpha = torch.sigmoid(routed_features[:capacity].mv(self.expert_centroids[self.expert_id])).unsqueeze(1)
                routed_features1 = alpha * self.expert_network(routed_features[:capacity]) + (1 - alpha) * routed_features[:capacity]
                routed_features2 = routed_features[capacity:]
                routed_features = torch.cat((routed_features1, routed_features2), dim=0)

        # Return to original worker and ordering
        result = All2AllDDM.apply(routed_features, input_splits, output_splits)[self.inverse_sort(sort_by_expert)]

        if self.shuffle and is_training:
            # Undo shuffling
            result = All2AllDDM.apply(result)[self.inverse_sort(shuffle_sort)]

        # Return additional Nones for compatibility with TransformerDecoderLayer
        return result.view(input_features.size()), None, None, balance_loss, distill_loss

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def calculate_balance_loss(self, token_features, tpe):
        # In order to reduce communication overheads, do not use switch loss
        # Instead, amplify or suppress a subset of token-to-expert scores according to the assignment balance condition of an expert.
        token_expert_affinities = token_features.matmul(self.expert_centroids.transpose(0, 1))
        _, input_splits, _ = self.greedy_assignment(token_expert_affinities)
        n_assigned_tokens = sum(input_splits)
        ave_assigned_tokens = tpe
        sum_score_partial = torch.sigmoid(token_expert_affinities[:, self.expert_id]).sum() if token_features.shape[0] > 0 else 0.0
        return (n_assigned_tokens - ave_assigned_tokens) / ave_assigned_tokens * sum_score_partial

    def calculate_subopt_balance_loss(self, token_features):
        token_expert_affinities = token_features.matmul(self.expert_centroids.transpose(0, 1))
        _, top1expert = token_expert_affinities.max(dim=1)
        subopt_token = ~(top1expert == self.expert_id)
        if (~subopt_token).all():
            subopt_loss = 0
        else:
            subopt_loss = -torch.sigmoid(token_expert_affinities[subopt_token, self.expert_id]).sum()
        return subopt_loss

    def calculate_subopt_balance_margin_loss(self, token_features):
        token_expert_affinities = token_features.matmul(self.expert_centroids.transpose(0, 1))
        max_affinities, top1expert = token_expert_affinities.max(dim=1)
        subopt_token = ~(top1expert == self.expert_id)
        if (~subopt_token).all():
            subopt_loss = 0
        else:
            subopt_loss = (torch.sigmoid(max_affinities[subopt_token]) - torch.sigmoid(token_expert_affinities[subopt_token, self.expert_id])).sum()
        return subopt_loss

    def inverse_sort(self, order):
        # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
        return torch.empty_like(order).scatter_(0, order, torch.arange(0, order.size(0), device=order.device))

    def make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores

    def balanced_assignment(self, scores):
        return self.cpp.balanced_assignment(scores), None, None

    # Assigns each token to the top k experts
    def greedy_assignment(self, scores, k=1):
        token_to_workers = torch.topk(scores, dim=1, k=k, largest=True).indices.view(-1)
        token_to_workers, sort_ordering = torch.sort(token_to_workers)
        worker2token = sort_ordering // k

        # Find how many tokens we're sending to each other worker (being careful for sending 0 tokens to some workers)
        output_splits = torch.zeros((self.num_workers,), dtype=torch.long, device=scores.device)
        workers, counts = torch.unique_consecutive(token_to_workers, return_counts=True)
        output_splits[workers] = counts
        # Tell other workers how many tokens to expect from us
        input_splits = All2AllDDM.apply(output_splits)
        return worker2token, input_splits.tolist(), output_splits.tolist()

    # Balancedly assigns each token to top experts, with greedy strategy
    def greedy_balanced_assignment(self, scores):
        # scores: [n_experts, n_tokens]
        scores = scores.t()
        n_experts, n_tokens = scores.shape[0], scores.shape[1]
        tpe = n_tokens // n_experts
        random_assign_order = np.array(range(n_experts))
        np.random.shuffle(random_assign_order)

        top_idx = torch.zeros((n_experts, tpe)).long()
        token_assigned = None
        minn = scores.min() - 1.0
        for exp_id in random_assign_order:
            exp_score = scores[exp_id, :]
            if token_assigned is not None:
                exp_score[token_assigned] = minn
            top_idx[exp_id, :] = torch.topk(exp_score, dim=0, k=tpe).indices
            if token_assigned is None:
                token_assigned = top_idx[exp_id, :]
            else:
                token_assigned = torch.cat((token_assigned, top_idx[exp_id, :]), dim=0)

        return top_idx.reshape(-1), None, None

    def load_assignment(self):
        try:
            from fairseq import libbase

            return libbase

        except ImportError as e:
            sys.stderr.write(
                "ERROR: missing libbase. run `python setup.py build_ext --inplace`\n"
            )
            raise e


class MoESublayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        self.norm = LayerNorm(args.decoder_embed_dim, export=False)
        self.ff1 = torch.nn.Linear(args.decoder_embed_dim, args.decoder_ffn_embed_dim)
        self.ff2 = torch.nn.Linear(args.decoder_ffn_embed_dim, args.decoder_embed_dim)
        self.ff2.weight.data.zero_()

    def forward(self, xs):
        return xs + self.ff2(self.activation_fn(self.ff1(self.norm(xs))))


class HashLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_workers = distributed_utils.get_data_parallel_world_size()
        self.expert_id = distributed_utils.get_data_parallel_rank()
        self.expert_network = nn.Sequential(*([MoESublayer(args) for _ in range(args.moe_sublayers)]))
        with open(self.args.hash_dict_path, 'r') as f:
            hash_dict = torch.tensor(json.load(f)).long()
            self.register_parameter("hash_dict", torch.nn.Parameter(hash_dict, requires_grad=False))

        # Add a special attribute to the expert parameters, so we know not to sync their gradients
        for param in self.expert_network.parameters():
            param.expert = True

    def forward(self, input_features, *args, **kwargs):
        assert kwargs['input_ids'].shape == input_features.shape[:-1]
        features = input_features.reshape(-1, input_features.size(-1))
        input_ids = kwargs['input_ids'].reshape(-1)

        # Compute which token goes to which expert
        sort_by_expert, input_splits, output_splits = self.hash_assignment(input_ids)

        # Swap these tokens for the right ones for our expert
        routed_features = All2AllDDM.apply(features[sort_by_expert], output_splits, input_splits)

        if routed_features.size(0) > 0:
            routed_features = self.expert_network(routed_features)

        # Return to original worker and ordering
        result = All2AllDDM.apply(routed_features, input_splits, output_splits)[self.inverse_sort(sort_by_expert)]

        # Return additional Nones for compatibility with TransformerDecoderLayer
        return result.view(input_features.size()), None, None, None, None

    def inverse_sort(self, order):
        # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
        return torch.empty_like(order).scatter_(0, order, torch.arange(0, order.size(0), device=order.device))

    def hash_assignment(self, input_ids):
        token_to_workers = self.hash_dict[input_ids]
        token_to_workers, sort_ordering = torch.sort(token_to_workers)
        worker2token = sort_ordering

        # Find how many tokens we're sending to each other worker (being careful for sending 0 tokens to some workers)
        output_splits = torch.zeros((self.num_workers,), dtype=torch.long, device=input_ids.device)
        workers, counts = torch.unique_consecutive(token_to_workers, return_counts=True)
        output_splits[workers] = counts
        # Tell other workers how many tokens to expect from us
        input_splits = All2AllDDM.apply(output_splits)
        return worker2token, input_splits.tolist(), output_splits.tolist()


class SwitchLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_workers = distributed_utils.get_data_parallel_world_size()
        self.expert_id = distributed_utils.get_data_parallel_rank()
        self.expert_network = nn.Sequential(*([MoESublayer(args) for _ in range(args.moe_sublayers)]))
        expert_centroids = torch.empty(self.num_workers, args.decoder_embed_dim)
        torch.nn.init.orthogonal_(expert_centroids, gain=0.1)
        self.register_parameter("expert_centroids", torch.nn.Parameter(expert_centroids))

        # Add a special attribute to the expert parameters, so we know not to sync their gradients
        for param in self.expert_network.parameters():
            param.expert = True

    def forward(self, input_features, *args, **kwargs):
        features = input_features.reshape(-1, input_features.size(-1))
        tpe = features.shape[0]
        is_training = input_features.requires_grad

        with torch.no_grad():
            # Compute similarity of each token to each expert for routing, and make the affinities finite
            token_expert_affinities = features.matmul(self.expert_centroids.transpose(0, 1))
            token_expert_affinities = self.make_finite(token_expert_affinities)

        # Compute which token goes to which expert
        sort_by_expert, input_splits, output_splits = self.greedy_assignment(token_expert_affinities)

        if not is_training:
            balance_loss = 0
        else:
            balance_loss = self.calculate_switch_balance_loss(features, sum(input_splits))

        # Swap these tokens for the right ones for our expert
        routed_features = All2AllDDM.apply(features[sort_by_expert], output_splits, input_splits)

        if routed_features.size(0) > 0:
            capacity = int(tpe * self.args.capacity_factor)
            if routed_features.size(0) <= capacity:
                alpha = routed_features.matmul(self.expert_centroids.transpose(0, 1)).softmax(dim=1)[:, self.expert_id].unsqueeze(1)
                routed_features = alpha * self.expert_network(routed_features) + (1 - alpha) * routed_features
            else:
                alpha = routed_features[:capacity].matmul(self.expert_centroids.transpose(0, 1)).softmax(dim=1)[:, self.expert_id].unsqueeze(1)
                routed_features1 = alpha * self.expert_network(routed_features[:capacity]) + (1 - alpha) * routed_features[:capacity]
                routed_features2 = routed_features[capacity:]
                routed_features = torch.cat((routed_features1, routed_features2), dim=0)

        # Return to original worker and ordering
        result = All2AllDDM.apply(routed_features, input_splits, output_splits)[self.inverse_sort(sort_by_expert)]

        # Return additional Nones for compatibility with TransformerDecoderLayer
        return result.view(input_features.size()), None, None, balance_loss, None

    def calculate_switch_balance_loss(self, token_features, n_assigned_tokens):
        n_assigned_tokens = torch.tensor([n_assigned_tokens], device=token_features.device).float().squeeze()

        all_features = [torch.empty_like(token_features) for i in range(self.num_workers)]
        all_features = AllGatherDDM.apply(all_features, token_features)
        all_features = torch.cat(all_features, dim=0)

        p_i_x = F.softmax(all_features.matmul(self.expert_centroids.transpose(0, 1)), dim=1)[:, self.expert_id]
        P_i = p_i_x.sum()
        #! smooth to avoid stucking
        f_i = n_assigned_tokens / all_features.shape[0] + 1e-6
        N = self.num_workers
        return P_i * f_i * N

    def make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores

    # Assigns each token to the top k experts
    def greedy_assignment(self, scores, k=1):
        token_to_workers = torch.topk(scores, dim=1, k=k, largest=True).indices.view(-1)
        token_to_workers, sort_ordering = torch.sort(token_to_workers)
        worker2token = sort_ordering // k

        # Find how many tokens we're sending to each other worker (being careful for sending 0 tokens to some workers)
        output_splits = torch.zeros((self.num_workers,), dtype=torch.long, device=scores.device)
        workers, counts = torch.unique_consecutive(token_to_workers, return_counts=True)
        output_splits[workers] = counts
        # Tell other workers how many tokens to expect from us
        input_splits = All2AllDDM.apply(output_splits)
        return worker2token, input_splits.tolist(), output_splits.tolist()

    def inverse_sort(self, order):
        # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
        return torch.empty_like(order).scatter_(0, order, torch.arange(0, order.size(0), device=order.device))


# Wraps torch.distributed.all_to_all_single as a function that supports autograd
class All2AllDDM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xs, input_splits=None, output_splits=None):
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits

        ys = torch.empty_like(xs) if output_splits is None else \
            xs.new_empty(size=[sum(output_splits)] + list(xs.size()[1:]))
        if input_splits is not None and output_splits is not None:
            xs_list = list(xs.split(input_splits, dim=0))
            for i in range(len(input_splits)):
                if input_splits[i] == 0:
                    xs_list[i] = xs.new_empty(size=[1] + list(xs.size()[1:]))
            ys_list = list(ys.split(output_splits, dim=0))
            for i in range(len(output_splits)):
                if output_splits[i] == 0:
                    ys_list[i] = xs.new_empty(size=[1] + list(xs.size()[1:]))
            torch.distributed.all_to_all(ys_list, xs_list)
            for i in range(len(output_splits)):
                if output_splits[i] == 0:
                    ys_list[i] = xs.new_empty(size=[0] + list(xs.size()[1:]))
            ys = torch.cat(ys_list, dim=0)
        else:
            torch.distributed.all_to_all_single(ys, xs)
        return ys

    @staticmethod
    def backward(ctx, grad_output):
        result = torch.empty_like(grad_output) if ctx.input_splits is None else \
            grad_output.new_empty(size=[sum(ctx.input_splits)] + list(grad_output.size()[1:]))
        if ctx.input_splits is not None and ctx.output_splits is not None:
            grad_output_list = list(grad_output.split(ctx.output_splits, dim=0))
            for i in range(len(ctx.output_splits)):
                if ctx.output_splits[i] == 0:
                    grad_output_list[i] = grad_output.new_empty(size=[1] + list(grad_output.size()[1:]))
            result_list = list(result.split(ctx.input_splits, dim=0))
            for i in range(len(ctx.input_splits)):
                if ctx.input_splits[i] == 0:
                    result_list[i] = grad_output.new_empty(size=[1] + list(grad_output.size()[1:]))
            torch.distributed.all_to_all(result_list, grad_output_list)
            for i in range(len(ctx.input_splits)):
                if ctx.input_splits[i] == 0:
                    result_list[i] = grad_output.new_empty(size=[0] + list(grad_output.size()[1:]))
            result = torch.cat(result_list, dim=0)
        else:
            torch.distributed.all_to_all_single(result, grad_output)
        return result, None, None


# Wraps torch.distributed.all_gather as a function that supports autograd
class AllGatherDDM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor_list, tensor):
        dist.all_gather(tensor_list, tensor)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = dist.get_rank()

        dist_ops = [
            dist.reduce(grad_list[i], i, async_op=True) for i in range(dist.get_world_size())
        ]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank]