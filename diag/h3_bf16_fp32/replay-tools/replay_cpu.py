"""CPU语义与保护关卡；不声称替代真实H3 GPU验收。"""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import types
import unittest

import torch
from torch.distributed.tensor.placement_types import Shard

from runtime import T, configuration
from unirl.algorithms.base import grpo_clip_loss
from unirl.utils.h3_checkup_ratios import ratio_rows, ratio_statistics, select_anchor, write_ratio_json
from unirl.types.sample import Part
from unirl.types.segments.latent import LatentSegment
from unirl.utils.h3_checkup_capture import CheckupBackend, forbidden_update
from unirl.utils.h3_checkup_data import (
    assert_same, atomic_json, atomic_save, canonical, local_slice, map_tensors, masked_loss, split_factor,
)
from unirl.utils.h3_checkup_replay import audit_fp32, tensors, write_full_meta


class Checks(unittest.TestCase):
    def test_mask_policy_and_kl_half_means(self):
        torch.manual_seed(42)
        x=torch.randn(192,5,dtype=torch.float64)
        adv=torch.randn(24,8,dtype=torch.float64)
        adv=(adv-adv.mean(1,keepdim=True)).flatten()
        theta=torch.randn(5,dtype=torch.float64,requires_grad=True)
        policy=-(x@theta)*adv
        kl=0.004*((x@theta)**2)
        row_loss=policy+kl
        gradients={}
        for split in ['full','A','B']:
            distributed=sum(masked_loss(row_loss[r*24:(r+1)*24].mean(),split,r) for r in range(8))/8
            grad=torch.autograd.grad(distributed,theta,retain_graph=True)[0]
            direct=row_loss.mean() if split=='full' else row_loss[:96].mean() if split=='A' else row_loss[96:].mean()
            expected=torch.autograd.grad(direct,theta,retain_graph=True)[0]
            torch.testing.assert_close(grad,expected,rtol=1e-13,atol=1e-13)
            gradients[split]=grad
        torch.testing.assert_close(gradients['full'],(gradients['A']+gradients['B'])/2,rtol=1e-13,atol=1e-13)
        # KL-only也必须正确分半，不能只把优势置零。
        for split,sel in [('A',slice(0,96)),('B',slice(96,192))]:
            actual=sum(masked_loss(kl[r*24:(r+1)*24].mean(),split,r) for r in range(8))/8
            torch.testing.assert_close(actual,kl[sel].mean())

    def test_full_mask_is_identity(self):
        x=torch.tensor(3.,requires_grad=True)
        self.assertIs(masked_loss(x,'full',0),x)

    def test_partition_24_complete_groups(self):
        a,b=set(),set()
        for rank in range(8):
            rows=set(range(rank*24,(rank+1)*24))
            (a if split_factor('A',rank) else b).update(rows)
        self.assertEqual(len(a),96)
        self.assertEqual(len(b),96)
        self.assertEqual(a|b,set(range(192)))
        self.assertFalse(a&b)
        for prompt in range(24):
            group=set(range(prompt*8,(prompt+1)*8))
            self.assertTrue(group<=a or group<=b)

    def test_placements_offsets_and_canonical_names(self):
        x=torch.arange(13*11).reshape(13,11)
        for dim in [0,1]:
            total=0
            for rank in range(8):
                size,start=Shard(dim)._local_shard_size_and_offset(x.shape[dim],8,rank)
                shape=list(x.shape);shape[dim]=size
                offset=[0,0];offset[dim]=start
                info={'global_shape':list(x.shape),'local_shape':shape,'offset':offset}
                local=local_slice(x,info)
                self.assertEqual(list(local.shape),shape)
                self.assertTrue(torch.equal(local,x.narrow(dim,start,size)))
                total+=size
            self.assertEqual(total,x.shape[dim])
        self.assertEqual(canonical('transformer_blocks.3._checkpoint_wrapped_module.attn.lora_A.default.weight'),
                         'transformer_blocks.3.attn.lora_A.default.weight')

    def test_part_roundtrip_and_fp32_promotion(self):
        part=Part(sample_ids=['a/0','a/1'],segment=LatentSegment(latents=torch.randn(2,3,4,dtype=torch.float16),
                  aux_latents=torch.randn(2,3,4,dtype=torch.float16),policy_loss_weights=torch.ones(2,1)),
                  conditions={'nested':{'embeds':torch.randn(2,3,4,dtype=torch.bfloat16),
                                        'mask':torch.ones(2,3,dtype=torch.bool)}},advantages=torch.tensor([1.,-1.]))
        with tempfile.TemporaryDirectory(dir=T/'results') as d:
            p=Path(d)/'part.pt';atomic_save(p,part)
            restored=torch.load(p,weights_only=False,map_location='cpu')
            assert_same(part,restored)
            promoted=map_tensors(restored,lambda t:t.float() if t.is_floating_point() else t)
            for before,after in zip(tensors(part),tensors(promoted),strict=True):
                self.assertTrue(torch.equal(before.float(),after.float()))
                if before.is_floating_point():self.assertEqual(after.dtype,torch.float32)
                else:self.assertEqual(after.dtype,before.dtype)

    def test_optimizer_guard_and_exact_unchanged(self):
        model=torch.nn.Linear(3,2)
        opt=torch.optim.AdamW(model.parameters())
        obj=types.SimpleNamespace(model=model,optimizer=opt,_optimizer_step_count=0,
              _checkup_before={canonical(n):p.detach().clone() for n,p in model.named_parameters()})
        # dist.get_rank由verify返回值使用；单rankGloo只在此CPU测试临时目录初始化。
        with tempfile.TemporaryDirectory(dir=T/'results') as d:
            torch.distributed.init_process_group('gloo',init_method='file://'+str(Path(d)/'gloo'),rank=0,world_size=1)
            try:
                opt.step=forbidden_update
                with self.assertRaisesRegex(RuntimeError,'FORBIDDEN'):opt.step()
                with self.assertRaisesRegex(RuntimeError,'FORBIDDEN'):CheckupBackend.optimizer_step(obj,max_grad_norm=1.)
                self.assertTrue(CheckupBackend.checkup_verify(obj)['lora_unchanged'])
                model(torch.ones(1,3)).sum().backward()
                self.assertTrue(CheckupBackend.checkup_verify(obj)['lora_unchanged'])
                with torch.no_grad():model.weight[0,0]+=1
                with self.assertRaises(AssertionError):CheckupBackend.checkup_verify(obj)
            finally:torch.distributed.destroy_process_group()

    def test_fp32_audit_no_gradient_change(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.RMSNorm(4),torch.nn.Linear(4,2))
        x=torch.randn(2,3)
        model(x).sum().backward()
        before=[p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        audit=audit_fp32(model)
        model(x).sum().backward()
        for expected,p in zip(before,model.parameters()):self.assertTrue(torch.equal(expected,p.grad))
        self.assertGreater(audit['forward_tensors'],0)
        self.assertGreater(audit['backward_tensors'],0)
        with self.assertRaises(AssertionError):audit_fp32(torch.nn.Linear(3,2).bfloat16())

    def test_done_metadata_contract(self):
        with tempfile.TemporaryDirectory(dir=T/'results') as d:
            d=Path(d)
            for rank in range(8):
                info={'global_shape':[16,3],'local_shape':[2,3],'placements':['Shard(0)'],
                      'mesh_shape':[8],'mesh_rank_order':list(range(8)),'offset':[rank*2,0]}
                atomic_save(d/f'rank{rank}.pt',{'p':torch.zeros(2,3)})
                ratio_data=torch.zeros(24,5);ratio_data[:,3:]=1
                atomic_save(d/f'ratio_rank{rank}.pt',{'rows':ratio_data,
                             'sample_ids':[f'r{rank}/{i}' for i in range(24)]})
                atomic_json(d/f'meta_rank{rank}.json',{'parameters':{'p':info},'lora_unchanged':True,
                    'optimizer_updates':0,'seconds':1.,'precision_audit':{}})
            write_full_meta(d,'V1','A',8)
            meta=json.loads((d/'meta.json').read_text())
            self.assertTrue((d/'DONE').exists())
            self.assertEqual(meta['loss']['rank_multipliers'],[2.,2.,2.,2.,0.,0.,0.,0.])
            self.assertEqual(meta['parameters']['p']['shards']['7']['offset'],[14,0])

    def test_real_loss_mask_position(self):
        path=T/'source/unirl/algorithms/flashgrpo.py'
        source=path.read_text()
        self.assertLess(source.index('loss = loss + kl_loss'),source.index('backward_loss = masked_loss('))
        self.assertLess(source.index('backward_loss = masked_loss('),source.index('backward_loss.backward()'))
        ast.parse(source)

    def test_on_policy_anchor_ratio_one_with_nonzero_gradient(self):
        new=torch.tensor([[0.4],[-0.7],[0.2]],requires_grad=True)
        rollout=torch.tensor([[0.1],[-0.9],[0.5]])
        adv=torch.tensor([[1.],[-2.],[0.5]])
        anchor=select_anchor(new,rollout,True)
        self.assertFalse(anchor.requires_grad)
        loss,metrics=grpo_clip_loss(new_logp=new,old_logp=anchor,advantages=adv,clip_range=0.001)
        self.assertEqual(float(metrics['ratio_min']),1.)
        self.assertEqual(float(metrics['ratio_max']),1.)
        self.assertEqual(float(metrics['clip_fraction']),0.)
        loss.sum().backward()
        self.assertTrue(torch.equal(new.grad,-adv))
        rows=ratio_rows(new,rollout,adv,anchor)
        self.assertTrue(torch.equal(rows[:,4],torch.ones(3)))
        self.assertFalse(torch.equal(rows[:,3],rows[:,4]))

    def test_ratio_statistics_known_values(self):
        rows=torch.zeros(8,5,dtype=torch.float64)
        rows[:,2]=torch.tensor([1,1,-1,-1,1,0,0,-1])
        rows[:,3]=torch.tensor([1,1.002,.998,1.003,.997,1.004,.996,1],dtype=torch.float64)
        rows[:,4]=1
        result=ratio_statistics(rows)
        stats=result['rollout_anchor']
        self.assertEqual(stats['would_clip_count'],2)
        self.assertEqual(stats['would_clip_fraction'],0.25)
        self.assertAlmostEqual(stats['ratio_min'],.996)
        self.assertAlmostEqual(stats['ratio_max'],1.004)
        self.assertAlmostEqual(stats['ratio_mean'],1.)
        for q,v in [('p50',.0025),('p90',.004),('p99',.004),('max',.004)]:
            self.assertAlmostEqual(stats['abs_ratio_minus_one'][q],v)
        self.assertEqual(result['diagnostic_anchor']['clip_fraction'],0.)

    def test_all_eight_rank_ratio_aggregation(self):
        with tempfile.TemporaryDirectory(dir=T/'results') as d:
            d=Path(d)
            for rank in range(8):
                rows=torch.zeros(24,5);rows[:,2]=1;rows[:,3]=1+rank*.002;rows[:,4]=1
                atomic_save(d/f'ratio_rank{rank}.pt',{'rows':rows,
                            'sample_ids':[f'r{rank}/{i}' for i in range(24)]})
            for split,expected,n in [('full',7/8,192),('A',3/4,96),('B',1.,96)]:
                result=write_ratio_json(d,'V1',split,8)
                self.assertEqual(result['rows'],n)
                self.assertEqual(result['rollout_anchor']['would_clip_fraction'],expected)
                self.assertEqual(result['all_192_rows_statistics']['rows'],192)
                self.assertEqual(result['all_ranks_collected'],list(range(8)))

    def test_normal_anchor_path_bitwise_identical(self):
        new=torch.tensor([[.2],[.7],[-.1],[.4]],dtype=torch.float32,requires_grad=True)
        old=torch.tensor([[.198],[.704],[-.1],[.3995]],dtype=torch.float32)
        adv=torch.tensor([[1.],[-1.],[0.],[2.]])
        loss0,m0=grpo_clip_loss(new_logp=new,old_logp=old,advantages=adv,clip_range=.001)
        grad0=torch.autograd.grad(loss0.sum(),new)[0]
        copied=new.detach().clone().requires_grad_()
        anchor=select_anchor(copied,old,False)
        self.assertIs(anchor,old)
        loss1,m1=grpo_clip_loss(new_logp=copied,old_logp=anchor,advantages=adv,clip_range=.001)
        grad1=torch.autograd.grad(masked_loss(loss1.sum(),'full',0),copied)[0]
        self.assertTrue(torch.equal(loss0,loss1))
        self.assertTrue(torch.equal(grad0,grad1))
        self.assertTrue(all(torch.equal(m0[k],m1[k]) for k in m0))

    def test_configs_preserve_design(self):
        for variant,micro in [('capture',8),('V1',8),('V3',4),('V3',2),('V2',1)]:
            cfg=configuration(variant,micro)
            self.assertEqual(cfg.algorithm.beta,0.004)
            self.assertEqual(cfg.batch_size*cfg.sampling.samples_per_prompt,192)
            self.assertFalse('ema_cfg' in cfg.backend)
            self.assertEqual(cfg.backend._target_,'unirl.utils.h3_checkup_capture.CheckupBackend')
            if variant=='V3':
                self.assertEqual(cfg.bundle.config.model_precision,'fp32')
                self.assertFalse(cfg.backend.fsdp_cfg.mixed_precision)
            if variant in ['V2','V3']:self.assertFalse(cfg.bundle.config.frozen_adaln_cache)


started=time.time()
stream=io.StringIO()
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
text=stream.getvalue()
assert not torch.cuda.is_initialized(), 'CPU checks must not initialize CUDA'
(T/'results/cpu-checks-replay.txt').write_text(text)
atomic_json(T/'results/cpu-checks-replay.json',{'status':'PASS' if result.wasSuccessful() else 'FAIL',
    'design_revision':'r3b-on-policy-anchor',
    'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
    'gpu_used':False,'seconds':time.time()-started,'torch':torch.__version__})
print(text,flush=True)
raise SystemExit(0 if result.wasSuccessful() else 1)
