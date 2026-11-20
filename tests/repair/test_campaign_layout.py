import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('layout', Path(__file__).parents[2] / 'scripts/campaign_layout.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class LayoutTests(unittest.TestCase):
    def test_metadata_relocates_without_changing_score_or_identity(self):
        row = {'selected': {'output_folder': '/old/node/output'}, 'cv_score': .847, 'run_id': 'original'}
        new = m.relocate_paths(row, Path('/old'), Path('/new'))
        self.assertEqual(new['selected']['output_folder'], '/new/node/output')
        self.assertEqual(new['cv_score'], row['cv_score'])
        self.assertEqual(new['run_id'], row['run_id'])

    def test_timeout_without_artifact_remains_method_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(m.is_diagnostic(Path(d), {'status':'no_artifact', 'termination_reason':'timeout'}, '25'))

    def test_quota_and_list_parser_crashes_are_diagnostic(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for text in ['GenerateRequestsPerDayPerProjectPerModel-FreeTier', "expected string or bytes-like object, got 'list'"]:
                (p/'search.log').write_text(text)
                self.assertTrue(m.is_diagnostic(p, {'status':'no_artifact','termination_reason':'process_error'}, '31'))

    def test_unknown_crash_is_not_silently_removed(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(m.is_diagnostic(Path(d), {'status':'no_artifact','termination_reason':'process_error'}, '31'))

    def test_preflight_conflict_does_not_move_files(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a',Path(d)/'b';a.mkdir();b.mkdir()
            (a/'model.pkl').write_bytes(b'first');(b/'model.pkl').write_bytes(b'second')
            with self.assertRaises(ValueError): m.validate_merge(a,b,set())
            self.assertEqual((a/'model.pkl').read_bytes(),b'first')
            self.assertEqual((b/'model.pkl').read_bytes(),b'second')

    def test_merge_preserves_all_model_bytes_and_removes_suffix_folder(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'source-textfix',Path(d)/'target';a.mkdir();b.mkdir()
            (a/'node').mkdir();(a/'node/model.pkl').write_bytes(b'pickle')
            (b/'old').mkdir()
            m.validate_merge(a,b,set());m.merge_tree(a,b)
            self.assertFalse(a.exists());self.assertEqual((b/'node/model.pkl').read_bytes(),b'pickle')

    def test_nonfinal_attempt_is_diagnostic(self):
        self.assertTrue(m.is_diagnostic(Path('/unused'), {'final':False}, '25'))

    def test_runtime_reconstruction_matches_recorded_correction(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'source'/m.REL).parent.mkdir(parents=True)
            source=b'class AssistantChatGemini:\n    def describe(self):\n        return {}\n'
            expected=source.decode().replace('    def describe(self):\n',m.METHOD+'    def describe(self):\n').encode()
            (p/'source'/m.REL).write_bytes(source)
            m.write(p/'runtime.json',{'source_hashes':{m.REL:m.digest(source)}})
            m.write(p/'runtime-gemini-textfix.json',{'source_hashes':{m.REL:m.digest(expected)}})
            self.assertEqual(m.runtime_files(p)[3], expected)

    def test_finals_from_original_identity_are_skipped_without_forged_id(self):
        import yaml
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);prefix=root/'.mlzero-local';prefix.mkdir()
            source=b'original adapter'
            state={'prefix':str(prefix),'source_hashes':{m.REL:m.digest(source)}}
            m.write(prefix/'runtime.json',state)
            (prefix/'source'/m.REL).parent.mkdir(parents=True)
            (prefix/'campaign-adapters').mkdir()
            (prefix/'campaign-adapters/gemini31.py').write_bytes(source)
            cfg={'llm':{'model':'gemini-3.1-flash-lite'},'mlzero':{'runtime_manifest':str(prefix/'runtime.json')},'local_runtime_identity':'test', 'outputs':{'artifact_root':str(root/'artifacts')}, 'campaign_id':m.N31,'dataset_entries':[{'key':'adult'}],'repetitions':[1,2]}
            cp=root/'cfg.yml';cp.write_text(yaml.safe_dump(cfg))
            rp=root/'artifacts/campaigns'/m.N31/'runs/adult/rep-1/oldhash/attempt-1/record.json'
            record={'dataset':'adult','repetition':1,'final':True,'run_id':'oldhash','status':'completed','cv_score':.847}
            m.write(rp,record)
            calls=[]
            m.write(root/'artifacts/campaign-layout.json',{})
            pkg=types.ModuleType('ageml_mlzero_benchmark');pkg.canonical_hash=lambda _: 'test'
            native=types.ModuleType('ageml_mlzero_benchmark.hamlet_cv')
            native.doctor=lambda *args:None;native.validate_inputs=lambda *args:None
            native.report=lambda *args:'report';native.run=lambda *args:calls.append(args)
            runtime=types.ModuleType('ageml_mlzero_benchmark.local_runtime')
            runtime.snapshot=lambda *args:state;runtime.verify=lambda *args:state
            from contextlib import nullcontext
            with patch.object(m,'ROOT',root),patch.object(m,'checkout_idle',nullcontext),patch.dict(sys.modules,{'ageml_mlzero_benchmark':pkg,'ageml_mlzero_benchmark.hamlet_cv':native,'ageml_mlzero_benchmark.local_runtime':runtime}),patch.dict('os.environ',{'GEMINI_API_KEY':'fake','AGEML_BENCH_PILOT':'0'}):
                m.launch([str(cp)])
            self.assertEqual([(a[2],a[3]) for a in calls],[(['adult'],2)])
            self.assertEqual(m.read(rp),record)


if __name__=='__main__':unittest.main()

class MigrationIntegration(unittest.TestCase):
    def test_full_migration_preserves_winner_and_removes_bad_records(self):
        import yaml
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);prefix=root/'.mlzero-local';prefix.mkdir();(root/'configs').mkdir()
            plain=b'class AssistantChatGemini:\n    def describe(self):\n        return {}\n'
            corrected=plain.decode().replace('    def describe(self):\n',m.METHOD+'    def describe(self):\n').encode()
            (root/'scripts').mkdir()
            (root/'scripts/cv_export_corrected.py').write_text('REVISION = 1\n')
            (root/'ageml_mlzero_benchmark').mkdir()
            (root/'ageml_mlzero_benchmark/cv_export.py').write_text('original validator\n')
            old={'prefix':str(prefix),'source_hashes':{m.REL:m.digest(plain)},'freeze':['openai==3.51.0']}
            fixed={**old,'source_hashes':{m.REL:m.digest(corrected)}}
            (prefix/'source'/m.REL).parent.mkdir(parents=True);(prefix/'source'/m.REL).write_bytes(plain)
            m.write(prefix/'runtime.json',old);m.write(prefix/'runtime-gemini-textfix.json',fixed)
            cfg={'llm':{'model':'gemini-3.1-flash-lite'},'outputs':{'artifact_root':str(root/'artifacts')},'mlzero':{'runtime_manifest':str(prefix/'runtime-gemini-textfix.json')},'campaign_id':m.N31+'-textfix'}
            original_cfg={**cfg,'campaign_id':m.N31}
            (root/'configs'/m.C31).write_text(yaml.safe_dump(original_cfg))
            (root/'configs'/m.C31.replace('.yml','-textfix.yml')).write_text(yaml.safe_dump(cfg))
            (root/'configs'/m.C25).write_text(yaml.safe_dump({'outputs':cfg['outputs']}))
            c25=root/'artifacts/campaigns'/m.N25
            c31=root/'artifacts/campaigns'/m.N31
            legacy=c31.with_name(m.N31+'-textfix')
            attempt=legacy/'runs/adult/rep-1/good/attempt-1';output=attempt/'mlzero/node_2/output';output.mkdir(parents=True)
            (output/'model_factory.py').write_bytes(b'original factory with historical path-textfix')
            abandoned=legacy/'runs/adult/rep-4/unfinished/attempt-1'
            abandoned.mkdir(parents=True);(abandoned/'search.log').write_text('interrupted')
            winner={'dataset':'adult','repetition':1,'status':'completed','cv_score':.847443,'run_id':'good','selected':{'output_folder':str(output)},'campaign_id':cfg['campaign_id']}
            m.write(attempt/'record.json',winner)
            m.write(c31/'runs/adult/rep-1/bad/attempt-1/record.json',winner)  # earlier copied summary
            m.write(c25/'runs/adult/rep-1/old/attempt-1/record.json',{'dataset':'adult','repetition':1,'status':'no_artifact','termination_reason':'timeout'})
            prior_regression=c25/'runs/auto_mpg/rep-1/old/attempt-1'
            m.write(prior_regression/'record.json',{'dataset':'auto_mpg','repetition':1,'status':'completed','cv_score':2.5})
            quota=c25/'runs/ames_housing/rep-1/old/attempt-2'
            m.write(quota/'record.json',{'dataset':'ames_housing','repetition':1,'status':'no_artifact','termination_reason':'process_error'})
            (quota/'search.log').write_text('GenerateRequestsPerDayPerProjectPerModel-FreeTier')
            for folder in (c31,legacy):
                (folder/'report').mkdir();(folder/'report/results.csv').write_text(str(folder))
            pkg=types.ModuleType('ageml_mlzero_benchmark');pkg.canonical_hash=lambda _: 'hash'
            runtime=types.ModuleType('ageml_mlzero_benchmark.local_runtime');runtime.snapshot=lambda _:old
            from contextlib import nullcontext
            with patch.object(m,'ROOT',root),patch.object(m,'checkout_idle',nullcontext),patch.dict(sys.modules,{'ageml_mlzero_benchmark':pkg,'ageml_mlzero_benchmark.local_runtime':runtime}):
                m.organize()
            self.assertFalse(legacy.exists());self.assertEqual(len(m.records(c31)),1)
            self.assertFalse((c31/'runs/adult/rep-4/unfinished/attempt-1').exists())
            self.assertTrue((root/'artifacts/diagnostics/campaign-layout/gemini31/interrupted/runs/adult/rep-4/unfinished/attempt-1/search.log').exists())
            row=m.read(m.records(c31)[0]);self.assertEqual(row['cv_score'],.847443)
            self.assertTrue(Path(row['selected']['output_folder']).is_dir())
            self.assertEqual((Path(row['selected']['output_folder'])/'model_factory.py').read_bytes(), b'original factory with historical path-textfix')
            self.assertFalse(quota.exists());self.assertEqual(len(m.records(c25)),1)
            self.assertFalse(prior_regression.exists())
            self.assertTrue((root/'artifacts/diagnostics/campaign-layout/gemini25/regression-before-numeric-fix/original/auto_mpg/rep-1/old/attempt-1/record.json').exists())
            # Applying the fix twice must not discard new regression attempts.
            new_reg=c25/'runs/ames_housing/rep-1/new/attempt-1/record.json'
            m.write(new_reg,{'dataset':'ames_housing','repetition':1,'status':'completed','cv_score':25000})
            with patch.object(m,'ROOT',root),patch.object(m,'checkout_idle',nullcontext),patch.dict(sys.modules,{'ageml_mlzero_benchmark':pkg,'ageml_mlzero_benchmark.local_runtime':runtime}):
                m.organize()
            self.assertTrue(new_reg.exists())
            canonical=yaml.safe_load((root/'configs'/m.C31).read_text())
            self.assertEqual(canonical['campaign_id'],m.N31)
            self.assertFalse((root/'configs'/m.C31.replace('.yml','-textfix.yml')).exists())
