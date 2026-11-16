# micro=1 与 micro=8 梯度对比：整体与逐 block 余弦、范数比、相对 L2
import json
import re
import sys
from pathlib import Path

import torch

T = Path(__file__).resolve().parents[1]


def load(name):
    d = T / 'gradients' / name
    assert (d / 'DONE').exists(), d
    return [torch.load(d / f'rank{r}.pt', map_location='cpu') for r in range(8)]


def stats(a, b):
    out = {}
    groups = {}
    for ra, rb in zip(a, b):
        assert ra.keys() == rb.keys()
        for k in ra:
            m = re.match(r'transformer_blocks\.(\d+)\.', k)
            g = f'block-{int(m.group(1)):02d}' if m else 'other'
            x, y = ra[k].double().flatten(), rb[k].double().flatten()
            s = groups.setdefault(g, [0.0, 0.0, 0.0, 0.0])
            s[0] += float(x @ y); s[1] += float(x @ x); s[2] += float(y @ y); s[3] += float((x - y) @ (x - y))
    tot = [sum(v[i] for v in groups.values()) for i in range(4)]
    for g, s in [('all', tot)] + sorted(groups.items()):
        out[g] = {'cos': s[0] / (s[1] * s[2]) ** 0.5, 'norm_ratio_a_over_b': (s[1] / s[2]) ** 0.5, 'rel_l2_vs_b': (s[3] / s[2]) ** 0.5}
    return out


names = sys.argv[1:]
grads = {n: load(n) for n in names}
res = {}
for i, a in enumerate(names):
    for b in names[i + 1:]:
        res[f'{a}_vs_{b}'] = stats(grads[a], grads[b])
(T / 'results' / 'micro-compare.json').write_text(json.dumps(res, indent=1))
for pair, r in res.items():
    c = [r[f'block-{i:02d}']['cos'] for i in range(50)]
    print(pair, 'all cos=%.6f ratio=%.4f relL2=%.4f' % (r['all']['cos'], r['all']['norm_ratio_a_over_b'], r['all']['rel_l2_vs_b']),
          '| b0-2', [round(x, 3) for x in c[:3]], 'b3-18 %.3f-%.3f' % (min(c[3:19]), max(c[3:19])), 'b19-49 %.3f-%.3f' % (min(c[19:]), max(c[19:])))
