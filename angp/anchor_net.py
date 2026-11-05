"""AnchorNet landmark selection (Section 4 of the paper)."""
import math

import torch


def farthest_point_sampling(X, k):
    """Indices of k points chosen by farthest point sampling, starting near the mean."""
    n = X.shape[0]
    ind = torch.zeros(k, dtype=torch.long, device=X.device)

    if n == 1:
        return ind

    center = X.mean(dim=0, keepdim=True)
    dist_to_center = torch.cdist(X, center).squeeze()
    ind[0] = torch.argmin(dist_to_center)

    # minimum distance from each point to the selected set
    min_dists = torch.cdist(X, X[ind[0]].unsqueeze(0)).squeeze()

    for i in range(1, k):
        ind[i] = torch.argmax(min_dists)
        dist_to_new = torch.cdist(X, X[ind[i]].unsqueeze(0)).squeeze()
        min_dists = torch.minimum(min_dists, dist_to_new)

    return ind


def get_anchor_net_indices(X, r, r_fps=None):
    """Select r landmark indices from X (n x d).

    r_fps anchors are chosen by farthest point sampling. The remaining points are
    assigned to their nearest anchor, and each cluster contributes landmarks in
    proportion to its size, stratified by distance to its anchor. If rounding leaves
    fewer than r landmarks, the rest are taken uniformly by distance to the data mean.
    The experiments use r_fps = r / 10.
    """
    n = X.shape[0]
    if r >= n:
        return torch.arange(n, device=X.device)

    if r_fps is None:
        r_fps = min(50, math.ceil(r / 5.0))
    r_fps = min(r_fps, r)

    # 1. anchors
    id_fps = farthest_point_sampling(X, r_fps)
    xox = id_fps.tolist()

    id_fps_set = set(xox)
    id_rest = torch.tensor([i for i in range(n) if i not in id_fps_set], device=X.device)

    if len(xox) == r:
        return torch.tensor(xox, device=X.device)

    # 2. assign every other point to its nearest anchor
    X_fps = X[id_fps]
    X_rest = X[id_rest]
    dists = torch.cdist(X_fps, X_rest)                 # (r_fps, n - r_fps)
    group = torch.argmin(dists, dim=0)

    # 3. stratified sampling within each cluster
    for k in torch.unique(group):
        idx_in_rest = torch.where(group == k)[0]
        idXk = id_rest[idx_in_rest]

        rk = math.floor((r - r_fps) * len(idXk) / len(id_rest))

        if rk == 0:
            continue
        elif rk == 1:
            # the point farthest from the anchor
            dd = torch.cdist(X_fps[k].unsqueeze(0), X[idXk]).squeeze(0)
            xox.append(idXk[torch.argmax(dd)].item())
        else:
            dd = torch.cdist(X_fps[k].unsqueeze(0), X[idXk]).squeeze(0)
            _, iii = torch.sort(dd)

            step = round(len(idXk) / rk)
            if step > 0:
                # MATLAB iii(step:step:end-step), followed by iii(end)
                matlab_slice = list(range(step, len(idXk) - step + 1, step))
                ipick = [iii[idx - 1].item() for idx in matlab_slice]
                ipick.append(iii[-1].item())
                xox.extend([idXk[idx].item() for idx in ipick])
            else:
                xox.extend(idXk.tolist())

    # 4. padding if rounding left fewer than r landmarks
    if len(xox) < r:
        rAdd = r - len(xox)
        xox_set = set(xox)
        iirest = torch.tensor([i for i in range(n) if i not in xox_set], device=X.device)

        if len(iirest) > 0:
            center = X.mean(dim=0, keepdim=True)
            dd = torch.cdist(X[iirest], center).squeeze(1)
            _, ii = torch.sort(dd)

            # MATLAB iirest(ii(ceil((1:rAdd) * length(iirest) / rAdd)))
            for j in range(1, rAdd + 1):
                idx_python = int(math.ceil(j * len(iirest) / rAdd)) - 1
                xox.append(iirest[ii[idx_python]].item())

    return torch.tensor(xox[:r], device=X.device)
