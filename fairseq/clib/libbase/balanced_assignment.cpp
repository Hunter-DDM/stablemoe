/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
C++ code for solving the linear assignment problem.
Based on the Auction Algorithm from https://dspace.mit.edu/bitstream/handle/1721.1/3265/P-2108-26912652.pdf and the implementation from:
https://github.com/bkj/auction-lap
Adapted to be more efficient when each worker is looking for k jobs instead of 1.
*/
#include <torch/extension.h>
#include <iostream>
using namespace torch::indexing;
torch::Tensor balanced_assignment(torch::Tensor job_and_worker_to_score) {
    // job_and_worker_to_score: [n_tokens, n_experts]
    int max_iterations = 100;
    torch::Tensor epsilon = (job_and_worker_to_score.max() - job_and_worker_to_score.min()) / 50;
    epsilon.clamp_min_(1e-04);
    torch::Tensor worker_and_job_to_score = job_and_worker_to_score.detach().transpose(0,1).contiguous();  // [n_experts, n_tokens]，价值
    int num_workers = worker_and_job_to_score.size(0);  // n_experts
    int num_jobs = worker_and_job_to_score.size(1);  // n_tokens
    auto device = worker_and_job_to_score.device();
    int jobs_per_worker = num_jobs / num_workers;
    torch::Tensor value = worker_and_job_to_score.clone();  // [n_experts, n_tokens]，净收益
    int counter = 0;
    torch::Tensor max_value = worker_and_job_to_score.max();

    torch::Tensor bid_indices;
    torch::Tensor cost = worker_and_job_to_score.new_zeros({1, num_jobs});  // [1, n_tokens]，物品价格
    torch::Tensor bids = worker_and_job_to_score.new_empty({num_workers, num_jobs});  // [n_experts, n_tokens]，当前轮加价
    torch::Tensor bid_increments = worker_and_job_to_score.new_empty({num_workers, jobs_per_worker});  // [n_experts, r]
    torch::Tensor top_values = worker_and_job_to_score.new_empty({num_workers, jobs_per_worker + 1});  // [n_experts, r + 1]
    torch::Tensor high_bids = worker_and_job_to_score.new_empty({num_jobs});  // [n_tokens]

    torch::Tensor top_index = top_values.to(torch::kLong);  // [n_experts, r + 1]
    torch::Tensor high_bidders = top_index.new_empty({num_jobs});  // [n_tokens]
    torch::Tensor have_bids = high_bidders.to(torch::kBool);  // [n_tokens]
    torch::Tensor jobs_indices = torch::arange({num_jobs}, torch::dtype(torch::kLong).device(device));  // [n_tokens]
    torch::Tensor true_tensor = torch::ones({1}, torch::dtype(torch::kBool).device(device));  // [1]

    while (true) {
        bids.zero_();
        // top_values, top_index 保存了每个 experts 最倾向的 r + 1 个 tokens 的分数和位置
        // top_values, top_index = torch.topk(value, k=jobs_per_worker + 1, dim=1)
        torch::topk_out(top_values, top_index, value, jobs_per_worker + 1, 1);

        // Each worker bids the difference in value between that job and the (k+1)th job
        // bid_increments = top_values[:, :-1] - top_values[:, -1:] + eps
        torch::sub_out(bid_increments,
                        top_values.index({Slice(None, None), Slice(0, jobs_per_worker)}),
                        top_values.index({Slice(None, None), jobs_per_worker}).unsqueeze(1));
        bid_increments.add_(epsilon);

        // bids = torch.zeros(n_experts, n_tokens).scatter_(dim=1, index=top_index[:, :-1], src=bid_increments)
        // bids = torch.zeros(n_experts, n_tokens)
        // bids[i][top_index[:, :-1][i][j]] = bid_increments[i][j], for i, j
        bids.scatter_(1,
            top_index.index({Slice(None, None), Slice(0, jobs_per_worker)}),
            bid_increments);

        if (counter < max_iterations && counter > 0) {
            // Put in a minimal bid to retain items from the last round if no-one else bids for them this round
            // bids.view(-1)[bid_indices] = eps
            bids.view(-1).index_put_({bid_indices}, epsilon);
        }

        // 为每个 token 找到最高的 bid 和对应的 expert，如果 bid 大于 0，那么它就 have_bid
        // Find the highest bidding worker per job
        // high_bids, high_bidders = torch.max(bids, dim=0)
        torch::max_out(high_bids, high_bidders, bids, 0);
        // have_bids = torch.gt(high_bids, 0)
        torch::gt_out(have_bids, high_bids, 0);

        // 如果所有 tokens 都 have_bid，那么可以提前退出了
        if (have_bids.all().item<bool>()) {
            // All jobs were bid for
            break;
        }

        // Make popular items more expensive
        // cost += high_bids
        cost.add_(high_bids);
        // value[i, :] = scores[i, :] - cost for i
        torch::sub_out(value, worker_and_job_to_score, cost);

        // 取出 have_bid 的 token 和 expert 在 value 里的下标 pair: (high_expert_idx, token_idx) for {token|token have_bid}
        // bid_indices = (high_bidders * num_jobs + jobs_indices)[have_bids]
        bid_indices = ((high_bidders * num_jobs) + jobs_indices).index({have_bids});

        if (counter < max_iterations)
        {
            // Make sure that this item will be in the winning worker's top-k next time.
            // value[high_expert_idx, token_idx] = inf for {token|token have_bid}
            value.view(-1).index_put_({bid_indices}, max_value);
        }
        else
        {
            // Suboptimal approximation that converges quickly from current solution
            // value[high_expert_idx, token_idx] = scores[high_expert_idx, token_idx] for {token|token have_bid}
            value.view(-1).index_put_({bid_indices}, worker_and_job_to_score.view(-1).index({bid_indices}));
        }

        counter += 1;
        if (counter > 10000)
            throw "May be dead loop in balanced assignment!";
    }

    // return top_index[:, :-1].reshape(-1)
    return top_index.index({Slice(None, None), Slice(0, jobs_per_worker)}).reshape(-1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("balanced_assignment", &balanced_assignment, "Balanced Assignment");
}
