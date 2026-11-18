import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest import mock
import numpy as np
from .test_stage2 import sample

spec=importlib.util.spec_from_file_location('p0_sources',Path(__file__).resolve().parents[1]/'tools/kaggle_audit_p0_sources.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def feature_sample(i):
    s=sample(str(i))
    s.scores[:2]=[.080078125,.14]; s.node_features[:,4]=s.scores
    roi=np.array([[.123456,1.34567]]*len(s.scores),np.float32)
    s.node_features=np.concatenate([s.node_features,roi],axis=1)
    s.roi_features=roi.astype(np.float16).astype(np.float32)
    return s


class SourceAuditTests(unittest.TestCase):
    def test_uploaded_reference_archive_and_extracted_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); expected={'arguments':{'smoke':False},'marker':'frozen fixture'}
            payload=json.dumps(expected)
            archive=root/'p0_reports.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('nested/audit_summary.json',payload)
            self.assertEqual(module.read_reference(archive),expected)
            self.assertEqual(module.read_reference(root),expected)
            # Uploaded archive can be renamed without an extension.
            extensionless=root/'report_archive'
            archive.rename(extensionless)
            self.assertEqual(module.read_reference(extensionless),expected)
            extracted=root/'p0_reports'/'nested'; extracted.mkdir(parents=True)
            summary=extracted/'audit_summary.json'; summary.write_text(payload,encoding='utf-8-sig')
            for path in [summary,str(summary),root,root/'p0_reports',archive]:
                self.assertEqual(module.read_reference(path),expected)
                self.assertEqual(module.resolve_reference_path(path),summary)
            with self.assertRaises(FileNotFoundError):
                module.read_reference(root/'missing.zip')
            (root/'audit_summary.json').write_text(payload,encoding='utf-8')
            with self.assertRaises(ValueError):
                module.read_reference(root)

    def test_bundled_reference_does_not_need_zip_asset(self):
        with mock.patch.object(module.zipfile,'ZipFile',side_effect=AssertionError('No external ZIP expected')):
            ref=module.read_reference()
        self.assertEqual(len(ref['evaluated_ids']['validation']),149)
        self.assertEqual(ref['train']['cache_sha256'],'4d2138cf0fba10afbeea3c1cf3ccbae072f0ebb647f84c7c1281b819e97acf47')
        self.assertEqual(ref['test']['cache_sha256'],'0e97ef57d22475e3dc34fe7e9692d6891760bc0795c20ab6d38274a27213e571')
        ref['evaluated_ids']['validation'].clear()
        self.assertEqual(len(module.read_reference()['evaluated_ids']['validation']),149)

    def test_threshold_and_quantization_checks(self):
        s=feature_sample(1)
        out=module.inspect_samples([s],.15)
        self.assertEqual(out['counts']['below_declared_tolerant'],2)
        self.assertEqual(out['counts']['score_column_mismatch'],0)
        self.assertEqual(out['counts']['roi_quantized_mismatch'],0)
        s.node_features[0,4]+=.01
        s.roi_features[1,0]+=.1
        out=module.inspect_samples([s],.15)
        self.assertEqual(out['counts']['score_column_mismatch'],1)
        self.assertEqual(out['counts']['roi_quantized_mismatch'],1)

    def test_full_source_audit_frozen_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ref=dict(arguments=dict(smoke=False),evaluated_ids=dict(validation=['2']))
            argv=[]
            for split,ids in [('train',[1,2]),('test',[3])]:
                path=root/f'{split}.pt'
                module.audit.runner.OfflineDatasetIO.save_samples([feature_sample(i) for i in ids],str(path))
                ref[split]=dict(cache_sha256=module.audit.runner.sha(path),manifest=dict(conf_thresh=.15))
                argv+=['--'+split+'-cache',str(path)]
            rp=root/'audit_summary.json'; rp.write_text(json.dumps(ref),encoding='utf-8')
            module.main(argv+['--p0-report',str(root),'--output',str(root/'out')])
            report=json.loads((root/'out/source_audit.json').read_text())
            self.assertEqual(report['status'],'METADATA_CONTRADICTION')
            self.assertTrue(all(report['cache_files_unchanged'].values()))
            self.assertTrue((root/'out/p0_sources_reports.zip').is_file())
            self.assertAlmostEqual(report['roi_comparisons']['test']['candidate_weighted']['cosine'],1.)
            # Default CLI path must work without any external reference file.
            ref['source_archive_sha256']='fixture'
            with mock.patch.object(module,'BUNDLED_REFERENCE',ref):
                module.main(argv+['--output',str(root/'bundled')])
            bundled=json.loads((root/'bundled/source_audit.json').read_text())
            self.assertEqual(bundled['reference_mode'],'bundled_p0_20261116')
            self.assertEqual(bundled['content_statistics'],report['content_statistics'])
            ref['train']['cache_sha256']='incorrect'
            rp.write_text(json.dumps(ref),encoding='utf-8')
            with self.assertRaises(SystemExit):
                module.main(argv+['--p0-report',str(rp),'--output',str(root/'bad')])
            self.assertEqual(json.loads((root/'bad/source_audit.json').read_text())['status'],'ERROR')


if __name__=='__main__': unittest.main()
