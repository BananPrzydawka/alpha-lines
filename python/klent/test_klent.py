"""Native integration and objective regression tests; no GPU required."""
import contextlib
import io
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from config import settings
from klent.native import Arena, Batch
from klent.train import losses, recalculated_policy, run, validate, evaluation_rows, historical_cycles, historical_rows, ModelHistory, load_reference_model, restore_optimizer
from models.katago import KataGoNet

LIBRARY = Path(__file__).resolve().parents[2]/'rust/target/release/libalpha_lines_game.so'

class KlentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference_dir = tempfile.TemporaryDirectory()
        cls.reference_path = Path(cls.reference_dir.name)/'349.pt'
        reference_config = dict(settings['katago_model'],filters=8,blocks=1,se_hidden=4,
                                policy_filters=4,action_value_filters=4)
        reference = KataGoNet(reference_config)
        torch.save(dict(format_version=1, options={'model':'katago'},
                        model_config=reference_config, model=reference.state_dict(),
                        summary={'cycle':349}),cls.reference_path)

    @classmethod
    def tearDownClass(cls):
        cls.reference_dir.cleanup()

    def setUp(self):
        torch.set_num_threads(1)
        self.options = dict(settings['klent'],n=8,m=1280,train_minibatch=30,test_games=4,
                            reference_checkpoint=str(self.reference_path))

    def test_reference_uses_its_saved_architecture(self):
        reference, cycle = load_reference_model(self.reference_path, 'cpu')
        self.assertEqual(cycle,349)
        self.assertEqual(reference.conv_input.out_channels,8)
        self.assertFalse(any(parameter.requires_grad for parameter in reference.parameters()))

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

    def test_recalculated_policy_respects_opening_legality_and_detaches(self):
        boards = torch.zeros(4,5,10,16)
        playable = torch.arange(80)//8*16+2*(torch.arange(80)%8)+(torch.arange(80)//8%2)
        boards[:3,1].reshape(3,160)[:,playable] = 1
        boards[2,1,0,0] = 0  # A later board permits both halves.
        logits = torch.zeros(4,10,16,requires_grad=True)
        q = torch.zeros(4,10,16,requires_grad=True)
        with torch.no_grad():
            q[0,0,0],q[0,0,2] = 1,2
        valid = torch.tensor([1.,1.,1.,0.])
        target = recalculated_policy(logits,q,boards,valid,0.03,0.1)
        self.assertFalse(target.requires_grad)
        torch.testing.assert_close(target.sum(1),valid)
        self.assertEqual(target[0].reshape(10,16)[:,8:].count_nonzero().item(),0)
        self.assertEqual(target[1].reshape(10,16)[:,:8].count_nonzero().item(),0)
        self.assertGreater(target[2].reshape(10,16)[:,8:].sum().item(),0)
        expected_ratio = torch.exp(torch.tensor((1.-2.)/0.13))
        torch.testing.assert_close(target[0,0]/target[0,2],expected_ratio)
        self.assertEqual(target[3].count_nonzero().item(),0)

    def test_recalculated_policy_runs_a_training_cycle(self):
        small = dict(settings['katago_model'],filters=8,blocks=1,se_hidden=4,
                     policy_filters=4,action_value_filters=4)
        options = dict(self.options,n=8,m=640,train_minibatch=40,test_games=2,
                       policy_recalculation=True)
        with patch.dict(settings,{'katago_model':small}), \
                contextlib.redirect_stdout(io.StringIO()), \
                torch.backends.mkldnn.flags(enabled=False):
            summary = run(LIBRARY,options,cycles=1,device='cpu',compile_model=False)[0]
        self.assertGreater(summary['states'],0)
        self.assertTrue(torch.isfinite(torch.tensor(summary['policy_loss'])))

    def test_native_batch_reproducibility_and_perspectives(self):
        with Arena(LIBRARY,self.options) as a, Arena(LIBRARY,self.options) as other:
            pi = torch.zeros(16,80)
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
                b,target,actions,returns = batch.tensors
                torch.testing.assert_close(b[0:2*size:2,3],b[1:2*size:2,4])
                torch.testing.assert_close(target[:2*size].float().sum(1),torch.ones(2*size),atol=.004,rtol=0)
                self.assertTrue((target[:2*size].gather(1,actions[:2*size,None])>0).all())
                self.assertEqual(b[2*size:].count_nonzero().item(),0)
                self.assertEqual(target[2*size:].count_nonzero().item(),0)
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
            self.assertEqual([s['training_opponent_cycles'] for s in summaries],[[0,0],[1,0]])
            self.assertEqual(report.getvalue().count(' complete\n'),2)
            for label in ('Setup', 'Total', 'CPU', 'Model self play', 'Model training', 'Strength test', 'Other'):
                expected = 1 if label == 'Setup' else 2
                self.assertEqual(sum(line.strip().startswith(label+' ') for line in report.getvalue().splitlines()),expected)
            for label, weight in (('Policy',self.options['policy_loss_weight']),
                                  ('Action value',self.options['q_loss_weight'])):
                lines = [line for line in report.getvalue().splitlines()
                         if line.startswith(f'    {label} ')]
                self.assertEqual(len(lines),2)
                self.assertTrue(all(f'× {weight} =' in line for line in lines))
            for removed in ('  Shuffle ', '  Scoring head processing ',
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
                self.assertGreaterEqual(summary['cpu_seconds'],
                                        summary['shuffle_seconds'])
                self.assertAlmostEqual(summary['loss'],
                    summary['policy_loss_weight']*summary['policy_loss']+
                    summary['q_loss_weight']*summary['q_loss'],places=4)


    def test_history_is_independent_bounded_and_correctly_aged(self):
        model = torch.nn.Linear(1,1)
        history = ModelHistory()
        with torch.no_grad():
            model.weight.fill_(1)
        history.initialize_anchors(model,1)
        history.remember_previous(model,1)
        with torch.no_grad():
            model.weight.fill_(2)
        results = [dict(anchor_index=i,score_rate=score)
                   for i,score in enumerate((0.70,0.81,0.91),1)]
        updates = history.update_anchors(model,2,results)
        self.assertEqual([u['anchor'] for u in updates],[2,3])
        self.assertEqual([c for c,_ in history.anchors],[1,2,2])
        self.assertEqual([w['weight'].item() for _,w in history.anchors],[1,2,2])
        self.assertEqual(history.previous[1]['weight'].item(),1)
        self.assertEqual([(kind,cycle,index) for kind,cycle,_,index in history.opponents()],
                         [('previous',1,None),('anchor',1,1),('anchor',2,2),('anchor',2,3)])
        with torch.no_grad():
            model.weight.fill_(3)
        results = [dict(anchor_index=i,score_rate=score)
                   for i,score in enumerate((0.71,0.80,0.90),1)]
        updates = history.update_anchors(model,3,results)
        self.assertEqual([u['anchor'] for u in updates],[1])
        self.assertEqual([c for c,_ in history.anchors],[3,2,2])

    def test_three_cycles_evaluate_previous_moving_anchors_and_reference(self):
        small = dict(settings['katago_model'],filters=8,blocks=1,se_hidden=4,
                     policy_filters=4,action_value_filters=4)
        with patch.dict(settings,{'katago_model':small}),contextlib.redirect_stdout(io.StringIO()), torch.backends.mkldnn.flags(enabled=False):
            summaries = run(LIBRARY,dict(self.options,model='katago'),
                            cycles=3,device='cpu',compile_model=False)
        for cycle,summary in enumerate(summaries,1):
            expected = ['previous'] + (['anchor']*3 if cycle > 1 else []) + ['reference']
            self.assertEqual([r['kind'] for r in summary['evaluations']],expected)
            self.assertEqual(summary['evaluations'][0]['opponent_cycle'],cycle-1)
            self.assertEqual(summary['evaluations'][-1]['opponent_cycle'],349)
            if cycle == 1:
                self.assertEqual(summary['anchor_cycles'],[1,1,1])
            if cycle == 2:
                self.assertTrue(all(r['shared_match'] for r in summary['evaluations'][1:4]))
            self.assertEqual(len(summary['anchor_cycles']),3)
            for result in summary['evaluations']:
                if result['kind'] != 'reference':
                    self.assertEqual(result['opponent_cycle'],cycle-result['age'])
                else:
                    self.assertIsNone(result['age'])
                self.assertEqual(sum(result[k] for k in ('wins','draws','losses')),4)

    def test_new_architectures_train_checkpoint_and_resume(self):
        from klent.checkpoint import save
        conv = dict(settings['resnet_model'],filters=8,blocks=1,se_hidden=4,
                    policy_filters=4,action_value_filters=4)
        configurations = {
            'resnet': conv,
            'katago_tf': dict(conv,attention_heads=2,mlp_ratio=2),
            'maia': dict(dim=16,layers=1,head_dim=8,mlp_ratio=2,gab_dim=4,
                         maia_big_version=True),
        }
        for name, configuration in configurations.items():
            with self.subTest(model=name), tempfile.TemporaryDirectory() as directory, \
                    patch.dict(settings,{f'{name}_model':configuration}), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    torch.backends.mkldnn.flags(enabled=False):
                options = dict(self.options,model=name,n=8,m=640,train_minibatch=40,test_games=2)
                first_dir = Path(directory)/'first'
                first = run(LIBRARY,options,cycles=1,device='cpu',compile_model=False,
                            checkpoint=lambda model,optimizer,o,summary,anchors:
                                save(first_dir,model,optimizer,o,summary,anchors),
                            history_dir=first_dir)[0]
                path = first_dir/'cycle-000001.pt'
                saved = torch.load(path,weights_only=True)
                self.assertEqual(saved['model_config'],configuration)
                self.assertEqual(saved['options']['model'],name)
                self.assertEqual([r['kind'] for r in first['evaluations']],['previous','reference'])
                second = run(LIBRARY,options,cycles=1,device='cpu',compile_model=False,
                             resume=path)[0]
                self.assertEqual(second['cycle'],2)
                self.assertEqual([r['kind'] for r in second['evaluations']],
                                 ['previous','anchor','anchor','anchor','reference'])
                if name == 'resnet':
                    (first_dir/'cycle-000000-weights.pt').unlink()
                    with self.assertRaisesRegex(FileNotFoundError,'Missing historical opponent'):
                        run(LIBRARY,options,cycles=1,device='cpu',compile_model=False,resume=path)

    def test_resume_restores_architecture_but_uses_current_training_settings(self):
        from klent.checkpoint import save
        small = dict(settings['katago_model'],filters=8,blocks=1,se_hidden=4,
                     policy_filters=4,action_value_filters=4)
        with tempfile.TemporaryDirectory() as directory, patch.dict(settings,{'katago_model':small}), contextlib.redirect_stdout(io.StringIO()), torch.backends.mkldnn.flags(enabled=False):
            def checkpoint(model,optimizer,options,summary,anchors):
                save(directory,model,optimizer,options,summary,anchors)
            run(LIBRARY,dict(self.options,model='katago'),cycles=1,device='cpu',
                compile_model=False,checkpoint=checkpoint,history_dir=directory)
            path = Path(directory)/'cycle-000001.pt'
            bootstrap = run(LIBRARY,dict(self.options,model='katago'),cycles=1,
                            device='cpu',compile_model=False,resume=path,
                            checkpoint=lambda model,optimizer,options,summary,anchors:
                                save(Path(directory)/'bootstrap',model,optimizer,options,summary,anchors),
                            history_dir=Path(directory)/'bootstrap')
            self.assertEqual([(r['kind'],r['opponent_cycle'])
                              for r in bootstrap[0]['evaluations']],
                             [('previous',1)]+[('anchor',1)]*3+[('reference',349)])
            bootstrapped = torch.load(Path(directory)/'bootstrap/cycle-000002.pt',weights_only=True)
            self.assertEqual(bootstrapped['anchor_system'],'score_gated_v1')
            self.assertEqual(len(bootstrapped['anchors']),3)
            continued = run(LIBRARY,dict(self.options,model='katago'),cycles=1,
                            device='cpu',compile_model=False,
                            resume=Path(directory)/'bootstrap/cycle-000002.pt')
            self.assertEqual([r['kind'] for r in continued[0]['evaluations']],
                             ['previous','anchor','anchor','anchor','reference'])
            self.assertEqual([r['opponent_cycle'] for r in continued[0]['evaluations'][1:4]],
                             [cycle for cycle,_ in bootstrapped['anchors']])
            # Change architecture defaults and runtime settings independently.
            settings['katago_model'] = dict(small, filters=16, blocks=2)
            current = dict(self.options,model='katago',cycles=1,n=8,m=640,
                           train_minibatch=40,test_games=6,lr=0.001,
                           weight_decay=0.02,alpha=0.04,beta=0.2,
                           **{'lambda':0.8,'seed':7})
            results = run(LIBRARY,current,
                          device='cpu',compile_model=False,resume=path,
                          checkpoint=checkpoint,log_dir=Path(directory)/'logs')
            self.assertEqual(results[0]['cycle'],2)
            self.assertEqual(results[0]['evaluations'][0]['opponent_cycle'],1)
            self.assertEqual([(r['kind'],r['opponent_cycle']) for r in results[0]['evaluations']],
                             [('previous',1)]+[('anchor',1)]*3+[('reference',349)])
            saved = torch.load(Path(directory)/'cycle-000002.pt',weights_only=True)
            self.assertEqual(saved['anchor_system'],'score_gated_v1')
            self.assertEqual(len(saved['anchors']),3)
            self.assertEqual([cycle for cycle,_ in saved['anchors']],results[0]['anchor_cycles'])
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

    def test_resume_drops_legacy_auxiliary_heads(self):
        from models.katago import KataGoNet
        small = dict(settings['katago_model'],filters=8,blocks=1,se_hidden=4,
                     policy_filters=4,action_value_filters=4)
        with tempfile.TemporaryDirectory() as directory, patch.dict(settings,{'katago_model':small}), \
                contextlib.redirect_stdout(io.StringIO()), torch.backends.mkldnn.flags(enabled=False):
            legacy = KataGoNet()
            legacy.mark_class_head = torch.nn.Conv2d(8,6,1)
            optimizer = torch.optim.AdamW(legacy.parameters())
            pi,q = legacy(torch.zeros(2,5,10,16))
            (pi.square().mean()+q.square().mean()).backward()
            optimizer.step()
            path = Path(directory)/'legacy.pt'
            torch.save(dict(format_version=1,model=legacy.state_dict(),optimizer=optimizer.state_dict(),
                            options=dict(self.options,model='katago'),model_config=small,
                            summary=dict(cycle=0),torch_rng=torch.get_rng_state(),cuda_rng=[]),path)
            result = run(LIBRARY,dict(self.options,model='katago'),cycles=1,
                         device='cpu',compile_model=False,resume=path)
            self.assertEqual(result[0]['cycle'],1)

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

    def test_geometric_training_opponents_and_side_assignment(self):
        expected = {1:(0,0),2:(1,0),3:(2,1),4:(2,0),5:(3,1),
                    6:(4,2),7:(5,3),8:(4,0),48:(32,16),
                    104:(72,40),204:(172,140)}
        self.assertEqual({cycle:historical_cycles(cycle) for cycle in expected},expected)
        self.assertEqual(historical_rows(1024,2).tolist(),
                         [2*i+(i>=640) for i in range(512,768)])
        self.assertEqual(historical_rows(1024,3).tolist(),
                         [2*i+(i>=896) for i in range(768,1024)])

    def test_checkpoint_roundtrip(self):
        from klent.checkpoint import save
        model = torch.nn.Linear(2,2)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1,2)).sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            anchors = [(2,{k:v.detach().clone() for k,v in model.state_dict().items()})]*3
            path = save(directory,model,optimizer,self.options,dict(cycle=5),anchors)
            saved = torch.load(path,weights_only=True)
            self.assertEqual(saved['anchor_system'],'score_gated_v1')
            self.assertEqual([cycle for cycle,_ in saved['anchors']],[2,2,2])
            torch.testing.assert_close(saved['anchors'][0][1]['weight'],model.weight)
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
        for override in ({'train_minibatch':3},{'test_games':3},{'m':1},{'n':9},{'lambda':1.1},{'alpha':0,'beta':0},
                         {'exploration_fraction':-0.1},{'exploration_fraction':1.1},
                         {'policy_loss_weight':-1},{'q_loss_weight':float('nan')},
                         {'model':'unknown'},
                         {'reference_checkpoint':''},{'policy_recalculation':1},
                         {'anchor_thresholds':[0.7,0.8]},
                         {'anchor_thresholds':[0.7,float('nan'),0.9]},
                         {'anchor_thresholds':[0.7,0.7,0.9]}):
            with self.assertRaises(ValueError):
                validate(dict(self.options,**override))


if __name__ == '__main__':
    unittest.main()
