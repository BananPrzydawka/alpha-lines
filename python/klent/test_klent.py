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
from klent.gradient_compare import compare as compare_gradients
from klent.train import losses, run, validate, evaluation_rows, ModelHistory, load_reference_model, restore_optimizer
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
        self.options = dict(settings['klent'],n=2,m=320,train_minibatch=30,test_games=4,
                            reference_checkpoints=[str(self.reference_path)],
                            anchor_interval=2,test_interval=1,opponent_interval=2)

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

    def test_optimizer_steps_once_per_minibatch_and_can_resume(self):
        from klent.checkpoint import save
        small = dict(settings['katago_model'],filters=8,blocks=1,se_hidden=4,
                     policy_filters=4,action_value_filters=4)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(settings,{'katago_model':small}), \
                contextlib.redirect_stdout(io.StringIO()), \
                torch.backends.mkldnn.flags(enabled=False):
            first = run(LIBRARY,dict(self.options,train_minibatch=38),cycles=1,
                        device='cpu',compile_model=False,
                        checkpoint=lambda model,optimizer,options,summary,anchors,opponent:
                            save(directory,model,optimizer,options,summary,anchors,opponent))[0]
            self.assertGreater(first['batches'],1)
            self.assertEqual(first['optimizer_steps'],first['batches'])
            initial_untrained = torch.load(Path(directory)/'cycle-000000.pt',weights_only=True)
            self.assertEqual(initial_untrained['summary']['cycle'],0)
            self.assertEqual(initial_untrained['format_version'],2)
            self.assertEqual(initial_untrained['summary']['anchor_cycles'],[0])
            self.assertNotIn('anchors',initial_untrained)
            initial_path = Path(directory)/'cycle-000001.pt'
            initial = torch.load(initial_path,weights_only=True)
            initial_step = next(iter(initial['optimizer']['state'].values()))['step'].item()
            self.assertEqual(initial_step,first['batches'])
            second = run(LIBRARY,dict(self.options,train_minibatch=38),cycles=1,
                         device='cpu',compile_model=False,resume=initial_path,
                         checkpoint=lambda model,optimizer,options,summary,anchors,opponent:
                             save(directory,model,optimizer,options,summary,anchors,opponent))[0]
            self.assertGreater(second['batches'],1)
            self.assertEqual(second['optimizer_steps'],second['batches'])
            resumed = torch.load(Path(directory)/'cycle-000002.pt',weights_only=True)
            self.assertEqual(next(iter(resumed['optimizer']['state'].values()))['step'].item(),
                             initial_step+second['batches'])

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
                b,target,actions,returns = batch.tensors
                torch.testing.assert_close(b[0:2*size:2,3],b[1:2*size:2,4])
                torch.testing.assert_close(target[:2*size].float().sum(1),torch.ones(2*size),atol=.004,rtol=0)
                self.assertTrue((target[:2*size].gather(1,actions[:2*size,None])>0).all())
                self.assertEqual(b[2*size:].count_nonzero().item(),0)
                self.assertEqual(target[2*size:].count_nonzero().item(),0)
            self.assertEqual(seen,count)
            a.reset(); a.clear()
            self.assertEqual(a.stats().positions,0)

    def test_fixed_opponent_records_balanced_players_and_current_targets(self):
        options = dict(self.options,n=2,m=320)
        with Arena(LIBRARY,options,fixed_opponent=True) as a, Arena(LIBRARY,options,fixed_opponent=True) as b:
            actions = torch.zeros(4,80)
            targets = torch.arange(80,dtype=torch.float32).repeat(4,1) / 10
            q = torch.zeros_like(actions)
            for _ in range(320):
                self.assertEqual(a.step(actions,q,target_logits=targets,target_q=q),
                                 b.step(actions,q))
                if a.stats().positions > 160:
                    break
            self.assertGreater(a.stats().positions,0)
            batch_a,batch_b = Batch(2),Batch(2)
            a.batch(0,batch_a); b.batch(0,batch_b)
            torch.testing.assert_close(batch_a.tensors[2],batch_b.tensors[2])
            self.assertGreater((batch_a.tensors[1]-batch_b.tensors[1]).abs().sum().item(),0)
            a.shuffle()
            players = set()
            batch = Batch(40)
            for start in range(0,a.stats().positions,20):
                size = a.batch(start,batch)
                players.update(batch.players[:size].tolist())
            self.assertEqual(players,{0,1})

    def test_fixed_opponent_trains_with_either_perspective_setting(self):
        small = dict(settings['katago_model'],filters=8,blocks=1,se_hidden=4,
                     policy_filters=4,action_value_filters=4)
        with patch.dict(settings,{'katago_model':small}), contextlib.redirect_stdout(io.StringIO()), \
                torch.backends.mkldnn.flags(enabled=False):
            for both_sides in (False,True):
                with self.subTest(both_sides=both_sides):
                    result = run(LIBRARY,dict(self.options,both_sides=both_sides),cycles=1,
                                 device='cpu',compile_model=False,fixed_opponent=True)[0]
                    self.assertGreater(result['states'],0)
                    self.assertEqual([row['kind'] for row in result['evaluations']],
                                     ['previous','anchor','reference'])

    def test_fixed_opponent_advances_on_interval_and_resumes(self):
        from klent.checkpoint import save
        small = dict(settings['katago_model'],filters=8,blocks=1,se_hidden=4,
                     policy_filters=4,action_value_filters=4)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(settings,{'katago_model':small}), \
                contextlib.redirect_stdout(io.StringIO()), \
                torch.backends.mkldnn.flags(enabled=False):
            options = dict(self.options,reference_checkpoints=[],opponent_interval=2,
                           anchor_interval=3,fixed_opponent=True)
            def checkpoint(model,optimizer,options,summary,anchors,opponent):
                save(directory,model,optimizer,options,summary,anchors,opponent)
            results = run(LIBRARY,options,cycles=3,device='cpu',compile_model=False,
                          checkpoint=checkpoint)
            self.assertEqual([row['training_opponent_cycle'] for row in results],[0,0,2])
            self.assertEqual([row['next_training_opponent_cycle'] for row in results],[0,2,2])
            saved = torch.load(Path(directory)/'cycle-000003.pt',weights_only=True)
            self.assertNotIn('training_opponent',saved)
            self.assertNotIn('anchors',saved)
            self.assertEqual(saved['summary']['anchor_cycles'],[0,3])
            resumed = run(LIBRARY,options,cycles=1,device='cpu',compile_model=False,
                          resume=Path(directory)/'cycle-000003.pt')[0]
            self.assertEqual(resumed['training_opponent_cycle'],3)
            self.assertEqual(resumed['next_training_opponent_cycle'],4)
            self.assertEqual(resumed['anchor_cycles'],[3])

    def test_compile_setting_controls_model_wrapping(self):
        small = dict(settings['katago_model'],filters=8,blocks=1,se_hidden=4,
                     policy_filters=4,action_value_filters=4)
        with patch.dict(settings,{'katago_model':small}), contextlib.redirect_stdout(io.StringIO()), \
                torch.backends.mkldnn.flags(enabled=False), \
                patch('torch.compile',side_effect=lambda model, **_: model) as compile_mock:
            for enabled, autotune in ((False,False),(True,False),(True,True)):
                compile_mock.reset_mock()
                run(LIBRARY,dict(self.options,compile_model=enabled,
                                 compile_max_autotune=autotune),cycles=1,
                    device='cpu')
                self.assertEqual(compile_mock.call_count,3 if enabled else 0)
                if enabled:
                    expected = dict(dynamic=False)
                    if autotune:
                        expected['mode'] = 'max-autotune'
                    self.assertTrue(all(call.kwargs == expected for call in compile_mock.call_args_list))

    def test_cycle_boundary_time_limit(self):
        small = dict(settings['katago_model'],filters=8,blocks=1,se_hidden=4,
                     policy_filters=4,action_value_filters=4)
        with patch.dict(settings,{'katago_model':small}), contextlib.redirect_stdout(io.StringIO()), \
                torch.backends.mkldnn.flags(enabled=False):
            results = run(LIBRARY,self.options,cycles=3,device='cpu',compile_model=False,
                          max_seconds=0.001)
        self.assertEqual(len(results),1)

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
            for label in ('Policy', 'Action value'):
                lines = [line for line in report.getvalue().splitlines()
                         if line.startswith(f'    {label} ')]
                self.assertEqual(len(lines),2)
                self.assertTrue(all('×' not in line for line in lines))
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
                    summary['policy_loss']+summary['q_loss'],places=4)


    def test_history_keeps_cycle_zero_and_every_interval_anchor(self):
        model = torch.nn.Linear(1,1)
        history = ModelHistory(2)
        with torch.no_grad():
            model.weight.fill_(1)
        history.initialize_anchors(model,0)
        history.remember_previous(model,0)
        with torch.no_grad():
            model.weight.fill_(2)
        self.assertIsNone(history.maybe_add_anchor(model,1))
        self.assertEqual(history.maybe_add_anchor(model,2),2)
        self.assertIsNone(history.maybe_add_anchor(model,3))
        self.assertEqual([c for c,_ in history.anchors],[0,2])
        self.assertEqual([w['weight'].item() for _,w in history.anchors],[1,2])
        self.assertEqual(history.previous[1]['weight'].item(),1)
        self.assertEqual([(kind,cycle) for kind,cycle,_ in history.opponents()],
                         [('previous',0),('anchor',0),('anchor',2)])
        with torch.no_grad():
            model.weight.fill_(3)
        self.assertEqual(history.maybe_add_anchor(model,4),4)
        self.assertEqual([c for c,_ in history.anchors],[0,2,4])

    def test_interval_evaluations_keep_previous_all_anchors_and_reference(self):
        small = dict(settings['katago_model'],filters=8,blocks=1,se_hidden=4,
                     policy_filters=4,action_value_filters=4)
        with patch.dict(settings,{'katago_model':small}),contextlib.redirect_stdout(io.StringIO()), torch.backends.mkldnn.flags(enabled=False):
            summaries = run(LIBRARY,dict(self.options,model='katago',test_interval=2),
                            cycles=4,device='cpu',compile_model=False)
        self.assertEqual([s['anchor_cycles'] for s in summaries],
                         [[0],[0,2],[0,2],[0,2,4]])
        self.assertEqual([len(s['evaluations']) for s in summaries],[0,3,0,4])
        self.assertEqual([(r['kind'],r['opponent_cycle']) for r in summaries[1]['evaluations']],
                         [('previous',1),('anchor',0),('reference',349)])
        self.assertEqual([(r['kind'],r['opponent_cycle']) for r in summaries[3]['evaluations']],
                         [('previous',3),('anchor',0),('anchor',2),('reference',349)])
        self.assertIsNone(summaries[0]['score_rate'])
        self.assertEqual(summaries[0]['strength_test_seconds'],0)

    def test_multiple_references_are_optional_and_reported_separately(self):
        small = dict(settings['katago_model'],filters=8,blocks=1,se_hidden=4,
                     policy_filters=4,action_value_filters=4)
        with patch.dict(settings,{'katago_model':small}),contextlib.redirect_stdout(io.StringIO()), \
                torch.backends.mkldnn.flags(enabled=False):
            no_reference = run(LIBRARY,dict(self.options,reference_checkpoints=[]),
                               cycles=1,device='cpu',compile_model=False)[0]
            references = run(LIBRARY,dict(self.options,reference_checkpoints=[]),
                             references=[str(self.reference_path)]*2,
                             cycles=1,device='cpu',compile_model=False)[0]
        self.assertEqual([row['kind'] for row in no_reference['evaluations']],
                         ['previous','anchor'])
        self.assertEqual([row['kind'] for row in references['evaluations']],
                         ['previous','anchor','reference','reference'])
        self.assertEqual(no_reference['states'],references['states'])
        self.assertAlmostEqual(no_reference['loss'],references['loss'],places=6)
        self.assertTrue(all(row['reference_path'] == str(self.reference_path)
                            for row in references['evaluations'][-2:]))

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
                options = dict(self.options,model=name,n=2,m=160,train_minibatch=40,test_games=2)
                first_dir = Path(directory)/'first'
                first = run(LIBRARY,options,cycles=1,device='cpu',compile_model=False,
                            checkpoint=lambda model,optimizer,o,summary,anchors,opponent:
                                save(first_dir,model,optimizer,o,summary,anchors,opponent))[0]
                path = first_dir/'cycle-000001.pt'
                saved = torch.load(path,weights_only=True)
                self.assertEqual(saved['model_config'],configuration)
                self.assertEqual(saved['options']['model'],name)
                self.assertEqual([r['kind'] for r in first['evaluations']],['previous','anchor','reference'])
                second = run(LIBRARY,options,cycles=1,device='cpu',compile_model=False,
                             resume=path)[0]
                self.assertEqual(second['cycle'],2)
                self.assertEqual([r['kind'] for r in second['evaluations']],
                                 ['previous','anchor','reference'])

    def test_resume_restores_architecture_but_uses_current_training_settings(self):
        from klent.checkpoint import save
        small = dict(settings['katago_model'],filters=8,blocks=1,se_hidden=4,
                     policy_filters=4,action_value_filters=4)
        with tempfile.TemporaryDirectory() as directory, patch.dict(settings,{'katago_model':small}), contextlib.redirect_stdout(io.StringIO()), torch.backends.mkldnn.flags(enabled=False):
            def checkpoint(model,optimizer,options,summary,anchors,opponent):
                save(directory,model,optimizer,options,summary,anchors,opponent)
            run(LIBRARY,dict(self.options,model='katago'),cycles=1,device='cpu',
                compile_model=False,checkpoint=checkpoint)
            path = Path(directory)/'cycle-000001.pt'
            bootstrap = run(LIBRARY,dict(self.options,model='katago'),cycles=1,
                            device='cpu',compile_model=False,resume=path,
                            checkpoint=lambda model,optimizer,options,summary,anchors,opponent:
                                save(Path(directory)/'bootstrap',model,optimizer,options,summary,anchors,opponent))
            self.assertEqual([(r['kind'],r['opponent_cycle'])
                              for r in bootstrap[0]['evaluations']],
                             [('previous',1),('anchor',1),('reference',349)])
            bootstrapped = torch.load(Path(directory)/'bootstrap/cycle-000002.pt',weights_only=True)
            self.assertEqual(bootstrapped['format_version'],2)
            self.assertEqual(bootstrapped['summary']['anchor_cycles'],[1,2])
            self.assertNotIn('anchors',bootstrapped)
            continued = run(LIBRARY,dict(self.options,model='katago'),cycles=1,
                            device='cpu',compile_model=False,
                            resume=Path(directory)/'bootstrap/cycle-000002.pt')
            self.assertEqual([r['kind'] for r in continued[0]['evaluations']],
                             ['previous','anchor','reference'])
            self.assertEqual(continued[0]['evaluations'][1]['opponent_cycle'],2)
            # Change architecture defaults and runtime settings independently.
            settings['katago_model'] = dict(small, filters=16, blocks=2)
            current = dict(self.options,model='katago',cycles=1,n=3,m=480,
                           train_minibatch=40,test_games=6,lr=0.001,
                           weight_decay=0.02,alpha=0.04,beta=0.2,
                           **{'lambda':0.8,'seed':7})
            results = run(LIBRARY,current,
                          device='cpu',compile_model=False,resume=path,
                          checkpoint=checkpoint,log_dir=Path(directory)/'logs')
            self.assertEqual(results[0]['cycle'],2)
            self.assertEqual(results[0]['evaluations'][0]['opponent_cycle'],1)
            self.assertEqual([(r['kind'],r['opponent_cycle']) for r in results[0]['evaluations']],
                             [('previous',1),('anchor',1),('reference',349)])
            saved = torch.load(Path(directory)/'cycle-000002.pt',weights_only=True)
            self.assertEqual(saved['format_version'],2)
            self.assertNotIn('anchors',saved)
            self.assertEqual(saved['summary']['anchor_cycles'],results[0]['anchor_cycles'])
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
                            summary=dict(cycle=1),torch_rng=torch.get_rng_state(),cuda_rng=[]),path)
            result = run(LIBRARY,dict(self.options,model='katago'),cycles=1,
                         device='cpu',compile_model=False,resume=path)
            self.assertEqual(result[0]['cycle'],2)
            self.assertEqual(result[0]['anchor_cycles'],[1,2])

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

    def test_gradient_comparison_uses_core_weights_and_no_optimizer_steps(self):
        small = dict(settings['katago_model'], filters=8, blocks=1, se_hidden=4,
                     policy_filters=4, action_value_filters=4)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(settings, {'katago_model':small}), \
                torch.backends.mkldnn.flags(enabled=False):
            model = KataGoNet(small)
            model.opponent_policy_head = torch.nn.Conv2d(8,1,1)
            path = Path(directory)/'legacy.pt'
            torch.save(dict(format_version=1, options={'model':'katago'},
                            model_config=small, model=model.state_dict(),
                            summary={'cycle':55}), path)
            result = compare_gradients(LIBRARY, path,
                options=dict(self.options,train_minibatch=20), chunks=2, device='cpu')
            self.assertEqual(result['cycle'],55)
            self.assertEqual(result['auxiliary_tensors_ignored'],2)
            self.assertEqual([row['perspectives'] for row in result['results']],[20,40])
            self.assertTrue(all(torch.isfinite(torch.tensor(row['mean_cosine']))
                                for row in result['results']))
            torch.testing.assert_close(torch.load(path,weights_only=True)['model']['conv_input.weight'],
                                       model.conv_input.weight)

    def test_checkpoint_roundtrip(self):
        from klent.checkpoint import save
        model = torch.nn.Linear(2,2)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1,2)).sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            weights = {k:v.detach().clone() for k,v in model.state_dict().items()}
            anchors = [(0,weights),(2,weights)]
            path = save(directory,model,optimizer,self.options,dict(cycle=5),anchors)
            saved = torch.load(path,weights_only=True)
            self.assertEqual(saved['format_version'],2)
            self.assertNotIn('anchors',saved)
            self.assertNotIn('training_opponent',saved)
            other = torch.nn.Linear(2,2)
            other.load_state_dict(saved['model'])
            torch.testing.assert_close(other.weight,model.weight)
            restored = torch.optim.AdamW(other.parameters())
            restored.load_state_dict(saved['optimizer'])
            self.assertTrue(restored.state)
            self.assertEqual(saved['summary']['cycle'],5)
            self.assertFalse(path.with_suffix('.pt.tmp').exists())

    def test_async_checkpoint_freezes_weights_before_next_update(self):
        from klent.checkpoint import AsyncWriter, write
        from threading import Event
        model = torch.nn.Linear(2,2)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1,2)).sum().backward()
        optimizer.step()
        expected = {key: value.detach().clone() for key,value in model.state_dict().items()}
        anchors = [(1,expected)]*3
        started, release = Event(),Event()
        def delayed_write(directory, snapshot):
            started.set()
            self.assertTrue(release.wait(5))
            return write(directory, snapshot)
        commits = []
        with tempfile.TemporaryDirectory() as directory:
            with patch('klent.checkpoint.write',side_effect=delayed_write):
                with AsyncWriter(directory,lambda: commits.append(True)) as writer:
                    writer.save(model,optimizer,self.options,dict(cycle=1),anchors)
                    self.assertTrue(started.wait(5))
                    with torch.no_grad():
                        model.weight.add_(10)
                    release.set()
            saved = torch.load(Path(directory)/'cycle-000001.pt',weights_only=True)
            torch.testing.assert_close(saved['model']['weight'],expected['weight'])
            self.assertFalse(torch.equal(saved['model']['weight'],model.weight))
            self.assertEqual(commits,[True])

    def test_validation(self):
        validate(self.options)
        for override in ({'train_minibatch':3},{'test_games':3},{'m':1},{'lambda':1.1},{'alpha':0,'beta':0},
                         {'exploration_fraction':-0.1},{'exploration_fraction':1.1},
                         {'model':'unknown'}, {'compile_model':0}, {'compile_max_autotune':1},
                         {'reference_checkpoints':['']},
                         {'reference_checkpoints':'foo'},
                         {'anchor_interval':0}, {'test_interval':0},
                         {'opponent_interval':0}, {'fixed_opponent':1}):
            with self.assertRaises(ValueError):
                validate(dict(self.options,**override))


if __name__ == '__main__':
    unittest.main()
