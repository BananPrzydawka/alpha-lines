"""Native integration and objective regression tests; no GPU required."""
import contextlib
import io
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from config import settings
from klent.native import Arena, Batch
from klent.train import losses, run, validate, evaluation_rows, ModelHistory

LIBRARY = Path(__file__).resolve().parents[2]/'rust/target/release/libalpha_lines_game.so'


class KlentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.options = dict(settings['klent'],n=2,m=320,train_minibatch=30,test_games=4)

    def test_full_spatial_loss_and_padding(self):
        logits = torch.zeros(4,10,16,requires_grad=True)
        q = torch.zeros_like(logits,requires_grad=True)
        target = torch.zeros(4,160)
        target[:,0] = 1
        actions = torch.tensor([0,2,4,6])
        returns = torch.ones(4)
        pl,vl = losses(logits,q,target,actions,returns,torch.tensor([1.,1.,0.,0.]))
        (pl+vl).backward()
        self.assertGreater(logits.grad[0,0,1].item(),0)  # permanently unplayable
        self.assertEqual(logits.grad[2:].abs().sum().item(),0)
        self.assertEqual(q.grad.count_nonzero().item(),2)
        self.assertAlmostEqual(pl.item(),torch.tensor(160.).log().item())
        self.assertEqual(vl.item(),1)

    def test_native_batch_reproducibility_and_perspectives(self):
        with Arena(LIBRARY,self.options) as a, Arena(LIBRARY,self.options) as other:
            pi = torch.zeros(4,80)
            for _ in range(160):
                ab,asc = a.inputs()
                bb,bsc = other.inputs()
                torch.testing.assert_close(ab,bb)
                torch.testing.assert_close(asc,bsc)
                full = a.step(pi,pi)
                self.assertEqual(full,other.step(pi,pi))
                if full:
                    break
            self.assertTrue(full)
            count = a.stats()[0]
            batch = Batch(30)
            seen = 0
            for start in range(0,count,15):
                size = a.batch(start,batch)
                seen += size
                b,s,target,actions,returns = batch.tensors
                torch.testing.assert_close(b[0:2*size:2,3],b[1:2*size:2,4])
                torch.testing.assert_close(s[:2*size:2],s[1:2*size:2].flip(1))
                torch.testing.assert_close(target[:2*size].float().sum(1),torch.ones(2*size),atol=.004,rtol=0)
                self.assertTrue((target[:2*size].gather(1,actions[:2*size,None])>0).all())
                self.assertEqual(b[2*size:].count_nonzero().item(),0)
                self.assertEqual(target[2*size:].count_nonzero().item(),0)
            self.assertEqual(seen,count)
            a.reset(); a.clear()
            self.assertEqual(a.stats()[0],0)

    def test_two_complete_cycles_both_models(self):
        for name in ('katago','resnet'):
            key = name+'_model'
            small = dict(settings[key],filters=8,blocks=1,se_hidden=4,
                         score_embed_hidden=8,policy_filters=4,action_value_filters=4)
            options = dict(self.options,model=name)
            with patch.dict(settings,{key:small}),contextlib.redirect_stdout(io.StringIO()), torch.backends.mkldnn.flags(enabled=False):
                summaries = run(LIBRARY,options,cycles=2,device='cpu',compile_model=False)
            self.assertEqual(len(summaries),2)
            for summary in summaries:
                self.assertEqual(sum(summary[k] for k in ('wins','draws','losses')),4)
                self.assertGreater(summary['states'],0)
                self.assertGreater(summary['loss'],0)

    def test_history_is_independent_bounded_and_correctly_aged(self):
        model = torch.nn.Linear(1,1)
        history = ModelHistory()
        for cycle in range(12):
            with torch.no_grad():
                model.weight.fill_(cycle)
            history.remember(model)
            self.assertEqual(len(history.snapshots),min(cycle+1,8))
            for age,weights in history.opponents():
                self.assertEqual(weights['weight'].item(),cycle+1-age)
            with torch.no_grad():
                model.weight.fill_(-100)
            self.assertEqual(history.opponents()[0][1]['weight'].item(),cycle)

    def test_nine_cycles_evaluate_available_history(self):
        small = dict(settings['resnet_model'],filters=8,blocks=1,se_hidden=4,
                     score_embed_hidden=8,policy_filters=4,action_value_filters=4)
        with patch.dict(settings,{'resnet_model':small}),contextlib.redirect_stdout(io.StringIO()), torch.backends.mkldnn.flags(enabled=False):
            summaries = run(LIBRARY,dict(self.options,model='resnet'),
                            cycles=9,device='cpu',compile_model=False)
        for cycle,summary in enumerate(summaries,1):
            self.assertEqual([r['age'] for r in summary['evaluations']],
                             [age for age in (1,2,4,8) if age<=cycle])
            for result in summary['evaluations']:
                self.assertEqual(result['opponent_cycle'],cycle-result['age'])
                self.assertEqual(sum(result[k] for k in ('wins','draws','losses')),4)

    def test_resume_continues_cycle_and_restores_configuration(self):
        from klent.checkpoint import save
        small = dict(settings['resnet_model'],filters=8,blocks=1,se_hidden=4,
                     score_embed_hidden=8,policy_filters=4,action_value_filters=4)
        with tempfile.TemporaryDirectory() as directory, patch.dict(settings,{'resnet_model':small}), contextlib.redirect_stdout(io.StringIO()), torch.backends.mkldnn.flags(enabled=False):
            def checkpoint(model,optimizer,options,summary):
                save(directory,model,optimizer,options,summary)
            run(LIBRARY,dict(self.options,model='resnet'),cycles=1,device='cpu',
                compile_model=False,checkpoint=checkpoint)
            path = Path(directory)/'cycle-000001.pt'
            # Deliberately provide different defaults: saved configuration wins.
            results = run(LIBRARY,dict(self.options,model='katago'),cycles=1,
                          device='cpu',compile_model=False,resume=path,
                          checkpoint=checkpoint)
            self.assertEqual(results[0]['cycle'],2)
            self.assertEqual(results[0]['evaluations'][0]['opponent_cycle'],1)
            saved = torch.load(Path(directory)/'cycle-000002.pt',weights_only=True)
            self.assertEqual(saved['options']['model'],'resnet')
            self.assertEqual(saved['model_config'],small)
            self.assertGreater(next(iter(saved['optimizer']['state'].values()))['step'].item(),
                               next(iter(torch.load(path,weights_only=True)['optimizer']['state'].values()))['step'].item())

    def test_balanced_assignment(self):
        old,new = evaluation_rows(8)
        self.assertEqual(new.tolist(),[1,3,5,7,8,10,12,14])
        self.assertEqual(old.tolist(),[0,2,4,6,9,11,13,15])
        self.assertEqual(sorted(torch.cat((old,new)).tolist()),list(range(16)))
        # Distinct model outputs must reach precisely their assigned native rows.
        output = torch.empty(16)
        output[old],output[new] = 10.,20.
        self.assertEqual(output.reshape(8,2)[:4].tolist(),[[10.,20.]]*4)
        self.assertEqual(output.reshape(8,2)[4:].tolist(),[[20.,10.]]*4)

    def test_checkpoint_roundtrip(self):
        from klent.checkpoint import save
        model = torch.nn.Linear(2,2)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1,2)).sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = save(directory,model,optimizer,self.options,dict(cycle=5))
            saved = torch.load(path,weights_only=True)
            other = torch.nn.Linear(2,2)
            other.load_state_dict(saved['model'])
            torch.testing.assert_close(other.weight,model.weight)
            restored = torch.optim.AdamW(other.parameters())
            restored.load_state_dict(saved['optimizer'])
            self.assertTrue(restored.state)
            self.assertEqual(saved['summary']['cycle'],5)
            self.assertFalse(path.with_suffix('.pt.tmp').exists())

    def test_validation(self):
        validate(self.options)
        for override in ({'train_minibatch':3},{'test_games':3},{'m':1},{'lambda':1.1},{'alpha':0,'beta':0}):
            with self.assertRaises(ValueError):
                validate(dict(self.options,**override))


if __name__ == '__main__':
    unittest.main()
