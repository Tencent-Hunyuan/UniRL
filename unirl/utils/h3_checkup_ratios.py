"""r3b：诊断anchor与八rank的rollout-anchor偏离统计。"""
from pathlib import Path

import torch

from unirl.utils.h3_checkup_data import atomic_json, split_factor


def select_anchor(new_logp, rollout_old_logp, enabled):
    return new_logp.detach() if enabled else rollout_old_logp


def ratio_rows(new_logp, rollout_old_logp, advantages, diagnostic_old_logp):
    assert new_logp.ndim == 2 and new_logp.shape[1] == 1
    # exp在前向所在设备/精度上算，保留实际训练ratio的舍入。
    with torch.no_grad():
        ratio = (new_logp - rollout_old_logp).exp()
        actual = (new_logp - diagnostic_old_logp).exp()
        return torch.stack([new_logp.flatten(), rollout_old_logp.flatten(), advantages.flatten(),
                            ratio.flatten(), actual.flatten()], dim=1).detach().cpu()


def ratio_statistics(rows, clip_range=0.001):
    assert rows.ndim == 2 and rows.shape[1] == 5 and rows.shape[0] > 0
    assert torch.isfinite(rows).all()
    ratio, adv = rows[:, 3], rows[:, 2]
    clipped = ((ratio > 1.0 + clip_range) & (adv > 0)) | ((ratio < 1.0 - clip_range) & (adv < 0))
    deviations = (ratio.double() - 1.0).abs()
    quantiles = torch.quantile(deviations, torch.tensor([0.5,0.9,0.99,1.0],dtype=torch.float64))
    actual = rows[:, 4]
    actual_clipped = ((actual > 1.0 + clip_range) & (adv > 0)) | ((actual < 1.0 - clip_range) & (adv < 0))
    return {'rows': len(rows), 'rollout_anchor': {
        'ratio_min':float(ratio.min()),'ratio_max':float(ratio.max()),'ratio_mean':float(ratio.double().mean()),
        'abs_ratio_minus_one':dict(zip(['p50','p90','p99','max'],map(float,quantiles))),
        'clip_range':clip_range,'would_clip_count':int(clipped.sum()),
        'would_clip_fraction':float(clipped.double().mean()),'advantage_nonzero_count':int((adv!=0).sum()),
        'criterion':'(r > 1+clip and advantage > 0) or (r < 1-clip and advantage < 0)'},
        'diagnostic_anchor':{'ratio_min':float(actual.min()),'ratio_max':float(actual.max()),
                             'ratio_mean':float(actual.double().mean()),
                             'clip_fraction':float(actual_clipped.double().mean())}}


def write_ratio_json(directory, variant, split, micro):
    directory = Path(directory)
    ranks = [r for r in range(8) if split_factor(split,r) != 0]
    collected = [torch.load(directory/f'ratio_rank{r}.pt',map_location='cpu',weights_only=True) for r in range(8)]
    assert all(x['rows'].shape==(24,5) and len(x['sample_ids'])==24 for x in collected)
    assert len({sid for x in collected for sid in x['sample_ids']})==192
    for x in collected:
        assert torch.equal(x['rows'][:,4],torch.ones_like(x['rows'][:,4])), 'diagnostic ratio must equal one'
    stats=ratio_statistics(torch.cat([collected[r]['rows'] for r in ranks]))
    result={'schema':1,'design_revision':'r3b-on-policy-anchor','variant':variant,'split':split,'micro':micro,
        'old_logp_source':'this_forward_new_logp.detach()', 'all_ranks_collected':list(range(8)),
        'selected_ranks':ranks,'population':'192 rows' if split=='full' else '96 rows in selected half',
        **stats,'all_192_rows_statistics':ratio_statistics(torch.cat([x['rows'] for x in collected])),
        'modalities':{'joint':'rollout_anchor and diagnostic_anchor above','video':None,'audio':None},
        'modality_limit':'Capture stores only element-weighted joint old_logp, not separate video/audio anchors; '
                         'a single joint scalar cannot be uniquely decomposed. Capture was left unchanged.'}
    atomic_json(directory/'ratio.json',result)
    return result
