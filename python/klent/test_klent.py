"""Native integration and objective regression tests; no GPU required."""
import contextlib
import io
import json
import unittest
import tempfile
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import torch
from config import settings
from klent.native import Arena, Batch
from klent.train import losses, opponent_policy_loss, run, validate, evaluation_rows, ModelHistory, mark_class_targets, mark_class_loss, discounted_mark_targets, discounted_mark_loss, immediate_score_loss, discounted_score_loss, restore_optimizer

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

    def test_opponent_policy_loss_swaps_paired_targets_and_ignores_padding(self):
        target = torch.zeros(4,160)
        target[0,2],target[1,4] = 1,1
        logits = torch.zeros(4,10,16,requires_grad=True)
        with torch.no_grad():
            logits[0,0,4] = 6
            logits[1,0,2] = 6
        loss = opponent_policy_loss(logits,target,torch.tensor([1.,1.,0.,0.]))
        own_targets = target.reshape(-1,2,160).flip(1).reshape(-1,160)
        wrong = opponent_policy_loss(logits,own_targets,torch.tensor([1.,1.,0.,0.]))
        self.assertLess(loss.item(),wrong.item())
        loss.backward()
        self.assertLess(logits.grad[0,0,4].item(),0)
        self.assertLess(logits.grad[1,0,2].item(),0)
        self.assertEqual(logits.grad[2:].count_nonzero().item(),0)

    def test_mark_class_targets_and_masked_loss(self):
        indices = torch.arange(80)//8*16+2*(torch.arange(80)%8)+(torch.arange(80)//8%2)
        classes = torch.full((2,80),-1,dtype=torch.int8)
        classes[0,0],classes[0,1],classes[0,2] = 0,2,5
        target = mark_class_targets(classes,indices)
        self.assertEqual(target.shape,(2,6,10,16))
        self.assertEqual(target[0,0,0,0].item(),1)
        self.assertEqual(target[0,2,0,2].item(),1)
        self.assertEqual(target[0,5,0,4].item(),1)
        self.assertEqual(target.sum().item(),3)
        logits = torch.zeros_like(target,requires_grad=True)
        loss = mark_class_loss(logits,target,torch.tensor([1.,0.]))
        loss.backward()
        self.assertAlmostEqual(loss.item(),torch.tensor(6.).log().item())
        self.assertEqual(logits.grad[1].count_nonzero().item(),0)
        self.assertEqual(logits.grad[0,:,0,1].count_nonzero().item(),0)

    def test_immediate_score_loss_both_players_and_padding(self):
        logits = torch.zeros(3,2,81,requires_grad=True)
        scores = torch.tensor([[0,80],[15,20],[80,80]])
        valid = torch.tensor([1.,1.,0.])
        loss = immediate_score_loss(logits,scores,valid)
        self.assertAlmostEqual(loss.item(),torch.tensor(81.).log().item(),places=5)
        loss.backward()
        self.assertEqual(logits.grad[2].count_nonzero().item(),0)
        self.assertLess(logits.grad[0,0,0].item(),0)
        self.assertLess(logits.grad[0,1,80].item(),0)

    def test_discounted_score_loss_uses_soft_targets_and_masks_padding(self):
        logits = torch.zeros(2,2,81,requires_grad=True)
        target = torch.zeros(2,2,81,dtype=torch.bfloat16)
        target[0,0,2],target[0,0,5] = 0.25,0.75
        target[0,1,7] = 1
        loss = discounted_score_loss(logits,target,torch.tensor([1.,0.]))
        self.assertAlmostEqual(loss.item(),torch.tensor(81.).log().item(),places=5)
        loss.backward()
        self.assertEqual(logits.grad[1].count_nonzero().item(),0)
        self.assertLess(logits.grad[0,0,5].item(),logits.grad[0,0,2].item())

    def test_discounted_mark_head_uses_eight_soft_classes_on_playable_squares(self):
        indices = torch.arange(80)//8*16+2*(torch.arange(80)%8)+(torch.arange(80)//8%2)
        distributions = torch.zeros(2,80,8,dtype=torch.bfloat16)
        distributions[0,:,7] = 1
        distributions[0,0,7] = 0.25
        distributions[0,0,6] = 0.75
        target = discounted_mark_targets(distributions,indices)
        self.assertEqual(target.shape,(2,8,10,16))
        self.assertEqual(target[0,6,0,0].item(),0.75)
        self.assertEqual(target[0,7,0,0].item(),0.25)
        self.assertEqual(target[0,:,0,1].count_nonzero().item(),0)
        logits = torch.zeros_like(target,requires_grad=True)
        loss = discounted_mark_loss(logits,target,torch.tensor([1.,0.]))
        self.assertAlmostEqual(loss.item(),torch.tensor(8.).log().item(),places=5)
        loss.backward()
        self.assertEqual(logits.grad[1].count_nonzero().item(),0)
        self.assertEqual(logits.grad[0,:,0,1].count_nonzero().item(),0)
        self.assertLess(logits.grad[0,6,0,0].item(),logits.grad[0,7,0,0].item())

    def test_native_batch_reproducibility_and_perspectives(self):
        with Arena(LIBRARY,self.options) as a, Arena(LIBRARY,self.options) as other:
            pi = torch.zeros(4,80)
            for _ in range(160):
                ab = a.inputs()
                bb = other.inputs()
                torch.testing.assert_close(ab,bb)
                full = a.step(pi,pi)
                self.assertEqual(full,other.step(pi,pi))
                if full:
                    break
            self.assertTrue(full)
            count = a.stats().positions
            batch = Batch(30)
            seen = 0
            for start in range(0,count,15):
                size = a.batch(start,batch)
                seen += size
                b,s,target,actions,returns,classes,discounted,discounted_marks = batch.tensors
                torch.testing.assert_close(b[0:2*size:2,3],b[1:2*size:2,4])
                torch.testing.assert_close(s[:2*size:2],s[1:2*size:2].flip(1))
                torch.testing.assert_close(target[:2*size].float().sum(1),torch.ones(2*size),atol=.004,rtol=0)
                self.assertTrue((target[:2*size].gather(1,actions[:2*size,None])>0).all())
                self.assertEqual(b[2*size:].count_nonzero().item(),0)
                self.assertEqual(target[2*size:].count_nonzero().item(),0)
                self.assertTrue((classes[2*size:] == -1).all())
                self.assertEqual(discounted[2*size:].count_nonzero().item(),0)
                self.assertEqual(discounted_marks[2*size:].count_nonzero().item(),0)
                torch.testing.assert_close(discounted[:2*size].float().sum(-1),
                    torch.ones(2*size,2),atol=.025,rtol=0)
                torch.testing.assert_close(discounted[0:2*size:2,0],discounted[1:2*size:2,1])
                torch.testing.assert_close(discounted[0:2*size:2,1],discounted[1:2*size:2,0])
                torch.testing.assert_close(discounted_marks[:2*size].float().sum(-1),
                    torch.ones(2*size,80),atol=.025,rtol=0)
                for class_id in range(6):
                    torch.testing.assert_close(discounted_marks[0:2*size:2,:,class_id],
                        discounted_marks[1:2*size:2,:,(class_id+3)%6])
                torch.testing.assert_close(discounted_marks[0:2*size:2,:,6:],
                    discounted_marks[1:2*size:2,:,6:])
                self.assertTrue(((classes[:2*size] >= -1) & (classes[:2*size] <= 5)).all())
                swapped = classes[1:2*size:2]
                own = classes[:2*size:2]
                torch.testing.assert_close(swapped,torch.where(own < 0,own,(own+3)%6))
                for row in range(2*size):
                    # Class 2 contributes two points, class 1 one, class 0 none.
                    own_score = sum(int(c) for c in classes[row].tolist() if 0 <= c < 3)
                    opponent_score = sum(int(c)-3 for c in classes[row].tolist() if 3 <= c < 6)
                    self.assertEqual((own_score,opponent_score),tuple(int(x) for x in s[row]))
            self.assertEqual(seen,count)
            a.reset(); a.clear()
            self.assertEqual(a.stats().positions,0)

    def test_two_complete_cycles(self):
        for name in ('katago',):
            key = name+'_model'
            small = dict(settings[key],filters=8,blocks=1,se_hidden=4,
                         policy_filters=4,action_value_filters=4)
            options = dict(self.options,model=name)
            report = io.StringIO()
            with patch.dict(settings,{key:small}),contextlib.redirect_stdout(report), torch.backends.mkldnn.flags(enabled=False):
                summaries = run(LIBRARY,options,cycles=2,device='cpu',compile_model=False)
            self.assertEqual(len(summaries),2)
            self.assertEqual(report.getvalue().count(' complete\n'),2)
            for label in ('Setup', 'Total', 'CPU', 'Model self play', 'Model training', 'Strength test', 'Other'):
                expected = 1 if label == 'Setup' else 2
                self.assertEqual(sum(line.strip().startswith(label+' ') for line in report.getvalue().splitlines()),expected)
            for label, weight in (('Policy',self.options['policy_loss_weight']),
                                  ('Opponent policy',self.options['opponent_policy_weight']),
                                  ('Action value',self.options['q_loss_weight']),
                                  ('Mark',self.options['mark_class_loss_weight']),
                                  ('Future mark',self.options['discounted_mark_weight']),
                                  ('Score',self.options['immediate_score_weight']),
                                  ('Future score',self.options['discounted_score_weight'])):
                lines = [line for line in report.getvalue().splitlines()
                         if line.startswith(f'    {label} ')]
                self.assertEqual(len(lines),2)
                self.assertTrue(all(f'× {weight} =' in line for line in lines))
            for removed in ('  Shuffle ', '  Scoring head processing ', '  Checkpoint ',
                            '  Metrics ', '  Reporting ', '  Cycle total '):
                self.assertNotIn(removed,report.getvalue())
            for block in report.getvalue().split('\nCycle ')[1:]:
                timing_lines = block.splitlines()
                labels = ('Total','CPU','Model self play','Model training','Strength test','Other')
                timing_rows = [next(i for i,line in enumerate(timing_lines)
                                    if line.strip().startswith(label+' ')) for label in labels]
                self.assertEqual(timing_rows,sorted(timing_rows))
                times = [float(timing_lines[i].split()[-2]) for i in timing_rows]
                self.assertAlmostEqual(sum(times[1:]),times[0],delta=0.04)
            for summary in summaries:
                self.assertEqual(sum(summary[k] for k in ('wins','draws','losses')),4)
                self.assertGreater(summary['states'],0)
                self.assertGreater(summary['loss'],0)
                self.assertGreaterEqual(summary['shuffle_seconds'],0)
                self.assertGreaterEqual(summary['scoring_head_processing_seconds'],0)
                self.assertGreaterEqual(summary['cpu_seconds'],
                                        summary['shuffle_seconds']+summary['scoring_head_processing_seconds'])
                self.assertGreaterEqual(summary['mark_class_loss'],0)
                self.assertGreater(summary['immediate_score_loss'],0)
                self.assertGreater(summary['discounted_score_loss'],0)
                self.assertGreater(summary['discounted_mark_loss'],0)
                self.assertGreater(summary['opponent_policy_loss'],0)
                self.assertAlmostEqual(summary['loss'],
                    summary['policy_loss_weight']*summary['policy_loss']+
                    summary['opponent_policy_weight']*summary['opponent_policy_loss']+
                    summary['q_loss_weight']*summary['q_loss']+
                    summary['mark_class_loss_weight']*summary['mark_class_loss']+
                    summary['immediate_score_weight']*summary['immediate_score_loss']+
                    summary['discounted_score_weight']*summary['discounted_score_loss']+
                    summary['discounted_mark_weight']*summary['discounted_mark_loss'],places=4)

    def test_history_is_independent_bounded_and_correctly_aged(self):
        model = torch.nn.Linear(1,1)
        history = ModelHistory()
        with torch.no_grad():
            model.weight.fill_(0)
        for cycle in range(36):
            history.remember_previous(model, cycle)
            with torch.no_grad():
                model.weight.fill_(cycle+1)
            history.remember_anchor(model, cycle+1)
            self.assertEqual([c for c,_ in history.anchors],
                             [c for c in (8,16,24) if c<=cycle+1])
            self.assertEqual(history.previous[1]['weight'].item(),cycle)
        bootstrapped = ModelHistory(start_cycle=5)
        for cycle in range(6, 30):
            bootstrapped.remember_anchor(model, cycle)
        self.assertEqual([c for c,_ in bootstrapped.anchors],[13,21,29])
        self.assertEqual([w['weight'].item() for _,w in history.anchors],[8,16,24])

    def test_33_cycles_evaluate_previous_and_fixed_anchors(self):
        small = dict(settings['katago_model'],filters=8,blocks=1,se_hidden=4,
                     policy_filters=4,action_value_filters=4)
        with patch.dict(settings,{'katago_model':small}),contextlib.redirect_stdout(io.StringIO()), torch.backends.mkldnn.flags(enabled=False):
            summaries = run(LIBRARY,dict(self.options,model='katago'),
                            cycles=33,device='cpu',compile_model=False)
        for cycle,summary in enumerate(summaries,1):
            expected = [('previous',cycle-1)]
            expected += [('anchor',c) for c in (8,16,24) if c<cycle-1]
            self.assertEqual([(r['kind'],r['opponent_cycle']) for r in summary['evaluations']],expected)
            for result in summary['evaluations']:
                self.assertEqual(result['opponent_cycle'],cycle-result['age'])
                self.assertEqual(sum(result[k] for k in ('wins','draws','losses')),4)

    def test_resume_restores_architecture_but_uses_current_training_settings(self):
        from klent.checkpoint import save
        small = dict(settings['katago_model'],filters=8,blocks=1,se_hidden=4,
                     policy_filters=4,action_value_filters=4)
        with tempfile.TemporaryDirectory() as directory, patch.dict(settings,{'katago_model':small}), contextlib.redirect_stdout(io.StringIO()), torch.backends.mkldnn.flags(enabled=False):
            def checkpoint(model,optimizer,options,summary,anchors):
                save(directory,model,optimizer,options,summary,anchors)
            run(LIBRARY,dict(self.options,model='katago'),cycles=1,device='cpu',
                compile_model=False,checkpoint=checkpoint)
            path = Path(directory)/'cycle-000001.pt'
            bootstrap = run(LIBRARY,dict(self.options,model='katago'),cycles=1,
                            device='cpu',compile_model=False,resume=path,
                            checkpoint=lambda model,optimizer,options,summary,anchors:
                                save(Path(directory)/'bootstrap',model,optimizer,options,summary,anchors))
            self.assertEqual([(r['kind'],r['opponent_cycle'],r['is_anchor'])
                              for r in bootstrap[0]['evaluations']], [('previous',1,False)])
            bootstrapped = torch.load(Path(directory)/'bootstrap/cycle-000002.pt',weights_only=True)
            self.assertNotIn('anchor_cycles',bootstrapped)
            self.assertNotIn('anchors',bootstrapped)
            self.assertFalse((Path(directory)/'bootstrap/anchors').exists())
            bootstrapped['anchor_cycles'] = [0]
            torch.save(bootstrapped,Path(directory)/'bootstrap/cycle-000002.pt')
            continued = run(LIBRARY,dict(self.options,model='katago'),cycles=1,
                            device='cpu',compile_model=False,
                            resume=Path(directory)/'bootstrap/cycle-000002.pt')
            self.assertEqual([(r['kind'],r['opponent_cycle'])
                              for r in continued[0]['evaluations']],
                             [('previous',2)])
            resume_state = torch.load(path,weights_only=True)
            resume_state['anchors'] = [(0, resume_state['model'])]
            torch.save(resume_state,path)
            # Change architecture defaults and runtime settings independently.
            settings['katago_model'] = dict(small, filters=16, blocks=2)
            current = dict(self.options,model='katago',cycles=1,n=3,m=480,
                           train_minibatch=40,test_games=6,lr=0.001,
                           weight_decay=0.02,alpha=0.04,beta=0.2,
                           **{'lambda':0.8,'discounted_score_lambda':0.7,
                              'discounted_mark_lambda':0.6,'seed':7})
            results = run(LIBRARY,current,
                          device='cpu',compile_model=False,resume=path,
                          checkpoint=checkpoint,log_dir=Path(directory)/'logs')
            self.assertEqual(results[0]['cycle'],2)
            self.assertEqual(results[0]['evaluations'][0]['opponent_cycle'],1)
            self.assertEqual([(r['kind'],r['opponent_cycle']) for r in results[0]['evaluations']],
                             [('previous',1)])
            saved = torch.load(Path(directory)/'cycle-000002.pt',weights_only=True)
            self.assertNotIn('anchor_cycles',saved)
            self.assertNotIn('anchors',saved)
            self.assertFalse((Path(directory)/'anchors').exists())
            self.assertEqual(saved['options'], dict(current, model='katago'))
            for group in saved['optimizer']['param_groups']:
                self.assertEqual(group['lr'], current['lr'])
                self.assertEqual(group['weight_decay'], current['weight_decay'])
            self.assertEqual(results[0]['wins']+results[0]['draws']+results[0]['losses'], 6)
            self.assertEqual(results[0]['batches'], (results[0]['states']+19)//20)
            self.assertTrue(all(v.dtype == torch.float32 for v in saved['model'].values()))
            for state in saved['optimizer']['state'].values():
                self.assertEqual(state['exp_avg'].dtype, torch.float32)
                self.assertEqual(state['exp_avg_sq'].dtype, torch.float32)
            metrics = [json.loads(line) for line in (Path(directory)/'logs/metrics.jsonl').read_text().splitlines()]
            self.assertEqual(metrics, results)
            self.assertEqual(metrics[0]['win_rate'], metrics[0]['wins']/6)
            self.assertEqual(metrics[0]['score_rate'], (metrics[0]['wins']+0.5*metrics[0]['draws'])/6)
            metadata = json.loads((Path(directory)/'logs/run.json').read_text())
            self.assertEqual(metadata['options'], dict(current, model='katago'))
            self.assertEqual(metadata['initial_cycle'], 1)
            self.assertEqual(saved['model_config'],small)
            self.assertGreater(next(iter(saved['optimizer']['state'].values()))['step'].item(),
                               next(iter(torch.load(path,weights_only=True)['optimizer']['state'].values()))['step'].item())

    def test_resume_rejects_bf16_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'legacy.pt'
            torch.save(dict(format_version=1, model={'weight': torch.ones(1, dtype=torch.bfloat16)}), path)
            with self.assertRaisesRegex(ValueError, 'FP32 checkpoint'):
                run(LIBRARY, self.options, cycles=1, device='cpu', compile_model=False, resume=path)

    def test_resume_adds_head_to_legacy_checkpoint(self):
        from models.katago import KataGoNet
        small = dict(settings['katago_model'],filters=8,blocks=1,se_hidden=4,
                     policy_filters=4,action_value_filters=4)
        with tempfile.TemporaryDirectory() as directory, patch.dict(settings,{'katago_model':small}), contextlib.redirect_stdout(io.StringIO()), torch.backends.mkldnn.flags(enabled=False):
            for has_mark_head, has_immediate_head in ((False,False),(True,False),(True,True)):
                with self.subTest(has_mark_head=has_mark_head,has_immediate_head=has_immediate_head):
                    legacy = KataGoNet()
                    if not has_immediate_head:
                        legacy.score_embed = torch.nn.Sequential(torch.nn.Linear(2,8),torch.nn.SiLU(),torch.nn.Linear(8,8))
                        modules = list(legacy._modules.items())
                        legacy._modules = OrderedDict(modules[:2]+[modules[-1]]+modules[2:-1])
                    def retained(name):
                        return not (name.startswith('discounted_score_head.') or
                                    name.startswith('discounted_mark_head.') or
                                    name.startswith('opponent_policy_head.') or
                                    (not has_immediate_head and name.startswith('immediate_score_head.')) or
                                    (not has_mark_head and name.startswith('mark_class_head.')))
                    optimizer = torch.optim.AdamW(p for name,p in legacy.named_parameters()
                                                   if retained(name))
                    board = torch.zeros(2,5,10,16)
                    scores = torch.zeros(2,2)
                    pi,q,mark,immediate,_,_,_ = legacy(board)
                    loss = pi.square().mean()+q.square().mean()
                    if not has_immediate_head:
                        loss = loss+legacy.score_embed(scores).square().mean()
                    else:
                        loss = loss+immediate.square().mean()
                    if has_mark_head:
                        loss = loss+mark.square().mean()
                    loss.backward()
                    optimizer.step()
                    path = Path(directory)/f'legacy-{has_mark_head}-{has_immediate_head}.pt'
                    old_weights = {k:v for k,v in legacy.state_dict().items()
                                   if retained(k)}
                    current = KataGoNet()
                    current.load_state_dict(old_weights,strict=False)
                    migrated = torch.optim.AdamW(current.parameters())
                    restore_optimizer(migrated,optimizer.state_dict(),old_weights,current)
                    old_tower = dict(legacy.named_parameters())['tower.0.reduce.weight']
                    new_tower = dict(current.named_parameters())['tower.0.reduce.weight']
                    torch.testing.assert_close(migrated.state[new_tower]['exp_avg'],optimizer.state[old_tower]['exp_avg'])
                    torch.save(dict(format_version=1,model=old_weights,optimizer=optimizer.state_dict(),
                                    options=dict(self.options,model='katago'),model_config=small,
                                    summary=dict(cycle=1),torch_rng=torch.get_rng_state(),cuda_rng=[]),path)
                    results = run(LIBRARY,dict(self.options,model='katago'),cycles=1,
                                  device='cpu',compile_model=False,resume=path)
                    self.assertEqual(results[0]['cycle'],2)
                    self.assertGreater(results[0]['mark_class_loss'],0)
                    self.assertGreater(results[0]['immediate_score_loss'],0)
                    self.assertGreater(results[0]['discounted_score_loss'],0)
                    self.assertGreater(results[0]['discounted_mark_loss'],0)
                    self.assertGreater(results[0]['opponent_policy_loss'],0)

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
            path = save(directory,model,optimizer,self.options,dict(cycle=5),
                        [(2,{k:v.detach().clone() for k,v in model.state_dict().items()})])
            saved = torch.load(path,weights_only=True)
            self.assertNotIn('anchors',saved)
            self.assertNotIn('anchor_cycles',saved)
            self.assertTrue((Path(directory)/'anchors/anchor-000002.pt').is_file())
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
        for override in ({'train_minibatch':3},{'test_games':3},{'m':1},{'lambda':1.1},{'alpha':0,'beta':0},
                         {'exploration_fraction':-0.1},{'exploration_fraction':1.1},
                         {'policy_loss_weight':-1},{'q_loss_weight':float('nan')},
                         {'mark_class_loss_weight':float('inf')},{'immediate_score_weight':-1},
                         {'discounted_score_weight':-1},{'discounted_score_lambda':1.1},
                         {'discounted_mark_weight':-1},{'discounted_mark_lambda':1.1},
                         {'opponent_policy_weight':-1},{'model':'resnet'}):
            with self.assertRaises(ValueError):
                validate(dict(self.options,**override))


if __name__ == '__main__':
    unittest.main()
