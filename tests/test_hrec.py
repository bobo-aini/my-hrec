"""Mechanism tests: hierarchy, masking, memory consistency and gradient paths."""
import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
import torch
from hrec.config import load_config
from hrec.demo import create_demo
from hrec.data import FeatureDataset,collate_features,apply_pattern,validate_protocol
from hrec.geometry import PoincareBall
from hrec.hierarchy import annotation_path,tree_distance,EmotionTree,sentiment_level
from hrec.retrieval import HyperbolicRetriever,module_digest
from hrec.model import HREC,EvidenceCompletion
from hrec.engine import load_bank
from hrec.prepare import convert_benchmark,import_memory


class GeometryTests(unittest.TestCase):
    def test_inverse_and_nonunit_curvature(self):
        for c in (.3,1.,3.):
            ball = PoincareBall(c)
            v = torch.tensor([[.02,-.12],[0.,0.]],dtype=torch.float64)
            torch.testing.assert_close(ball.log_map_zero(ball.exp_map_zero(v)),v,atol=1e-10,rtol=1e-10)
            self.assertTrue((ball.project(torch.ones(2,3)*100).norm(dim=-1) < 1/c**.5).all())

    def test_distance_origin_symmetry_and_zero(self):
        ball = PoincareBall()
        x = torch.tensor([[.1,.2]],dtype=torch.float64)
        y = torch.tensor([[-.2,.1]],dtype=torch.float64)
        torch.testing.assert_close(ball.distance(x,y),ball.distance(y,x))
        torch.testing.assert_close(ball.distance(torch.zeros_like(x),x),2*torch.atanh(x.norm(dim=-1)))
        self.assertLess(ball.distance(x,x).item(),1e-12)

    def test_finite_origin_gradient(self):
        v = torch.zeros(2,3,requires_grad=True)
        ball = PoincareBall()
        loss = ball.exp_map_zero(v).sum() + ball.distance(ball.exp_map_zero(v),torch.ones(2,3)*.1).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(v.grad).all())


class HierarchyTests(unittest.TestCase):
    def test_surprise_is_undetermined_and_synonyms_merge(self):
        p = annotation_path({"id":"x","label":"surprised"})
        self.assertEqual(p,("root","undetermined","surprise"))
        self.assertEqual(annotation_path({"id":"x","label":"happiness"}),annotation_path({"id":"y","label":"joy"}))

    def test_sentiment_normalization(self):
        self.assertEqual(sentiment_level(1.,"simsv2"),3)
        self.assertEqual(sentiment_level(-.5,"mosei"),-1)
        self.assertEqual(sentiment_level(0.,"mosei"),0)

    def test_shortest_path_for_coarse_and_fine(self):
        self.assertEqual(tree_distance(("root","negative","anger"),("root","negative","sadness")),2)
        self.assertEqual(tree_distance(("root","negative"),("root","negative","anger","frustration")),2)

    def test_canonical_paths_merge_equivalent_nodes(self):
        self.assertEqual(annotation_path({"id":"x","hierarchy":["positive","joy","happiness"]}),
                         annotation_path({"id":"x","label":"joy"}))
        self.assertEqual(annotation_path({"id":"x","label":"平静"}), ("root","neutral"))

    def test_stratified_candidates_and_shortage_fill(self):
        records = []
        for label in ("anger","sadness","joy"):
            records += [{"id":f"{label}_{i}","label":label} for i in range(40)]
        tree = EmotionTree(records)
        selected,_ = tree.candidates({"id":"q","label":"anger"},96)
        self.assertEqual(len(set(selected.tolist())),96)
        self.assertEqual([(selected//40 == j).sum().item() for j in range(3)],[32,32,32])
        small = EmotionTree(records[:4]+records[40:80]+records[80:])
        selected,_ = small.candidates({"id":"q","label":"anger"},60)
        self.assertEqual(len(set(selected.tolist())),60)


class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.directory = tempfile.TemporaryDirectory()
        cls.cfg = load_config(create_demo(cls.directory.name))
        cls.train = FeatureDataset(cls.cfg["train_manifest"],cls.cfg)
        cls.memory = FeatureDataset(cls.cfg["memory_manifest"],cls.cfg)
        cls.retriever = HyperbolicRetriever(cls.cfg).eval()
        batch = collate_features([cls.memory[i] for i in range(len(cls.memory))])
        with torch.no_grad():
            _,summary,tangent = cls.retriever.encode(batch)
        cls.bank = {"coordinates":cls.retriever.ball.exp_map_zero(tangent),
                    "evidence":{"audio":summary["audio"],"visual":summary["visual"]},
                    "records":cls.memory.records,"retriever_fingerprint":module_digest(cls.retriever)}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def batch(self):
        return collate_features([self.train[0],self.train[1]])

    def test_text_excluded_from_nonverbal_retrieval(self):
        batch = self.batch()
        reference = self.retriever.retrieve(batch,self.bank)
        batch["features"]["text"] = torch.randn_like(batch["features"]["text"])*100
        changed = self.retriever.retrieve(batch,self.bank)
        torch.testing.assert_close(reference["distances"],changed["distances"])
        self.assertTrue((changed["alpha"][:,2]==0).all())

    def test_text_fallback_is_only_active_query(self):
        batch = apply_pattern(self.batch(),"text_only")
        first = self.retriever.retrieve(batch,self.bank)
        batch["features"]["audio"] += 50
        batch["features"]["visual"] -= 50
        second = self.retriever.retrieve(batch,self.bank)
        torch.testing.assert_close(first["distances"],second["distances"])
        torch.testing.assert_close(first["alpha"],torch.tensor([[0.,0.,1.],[0.,0.,1.]]))

    def test_topk_chunking_equals_full_distance_sort(self):
        batch = self.batch()
        out = self.retriever.retrieve(batch,self.bank)
        _,s,z = self.retriever.encode(batch)
        q,a = self.retriever.query(s,z,batch["observed"])
        distances = self.retriever.candidate_distances(q,a,self.retriever.ball.log_map_zero(self.bank["coordinates"]))
        expected,indices = distances.topk(self.cfg["top_k"],largest=False)
        torch.testing.assert_close(out["distances"],expected,atol=1e-6,rtol=1e-5)
        self.assertTrue(torch.equal(out["indices"],indices))

    def test_completion_attention_retains_geodesic_prior(self):
        module = EvidenceCompletion(self.cfg).eval()
        with torch.no_grad():
            module.query.weight.zero_()
        weights = torch.tensor([[.1,.2,.3,.4]])
        _,_,_,attention = module(torch.randn(1,2,16),torch.ones(1,2,dtype=torch.bool),
                                  torch.tensor([True]),torch.randn(1,4,16),weights.log())
        torch.testing.assert_close(attention[0,0],weights[0])

    def test_labels_do_not_change_prediction(self):
        model = HREC(self.cfg,copy.deepcopy(self.retriever),self.bank).eval()
        batch = self.batch()
        with torch.no_grad():
            a = model(batch)["prediction"]
            for row in batch["records"]:
                row["label"] = "QUERY_SECRET_SENTINEL"
            b = model(batch)["prediction"]
        torch.testing.assert_close(a,b)

    def test_stage_two_gradients_and_frozen_parameters(self):
        model = HREC(self.cfg,copy.deepcopy(self.retriever),self.bank).train()
        before_ret = module_digest(model.retriever)
        before_llm = module_digest(model.backbone)
        before_completion = module_digest(model.completion)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        result = model(self.batch())
        result["prediction"].square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.completion.parameters()))
        self.assertTrue(all(p.grad is None for p in model.retriever.parameters()))
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        optimizer.step()
        self.assertEqual(before_ret,module_digest(model.retriever))
        self.assertEqual(before_llm,module_digest(model.backbone))
        self.assertNotEqual(before_completion,module_digest(model.completion))

    def test_missing_inputs_and_gate_normalization(self):
        model = HREC(self.cfg,copy.deepcopy(self.retriever),self.bank).eval()
        for pattern in ("all","no_text","no_audio","no_visual","text_only"):
            result = model(apply_pattern(self.batch(),pattern))
            self.assertTrue(torch.isfinite(result["prediction"]).all())
            torch.testing.assert_close(result["beta"].sum(-1),torch.ones(2))
            self.assertEqual(result["descriptor"].shape,(2,5))
            self.assertTrue(((result["descriptor"][:,1] >= 0)&(result["descriptor"][:,1] <= 1+1e-6)).all())

    def test_last_nonpadding_state_matches_individual_forward(self):
        model = HREC(self.cfg,copy.deepcopy(self.retriever),self.bank).eval()
        items = [self.train[0],self.train[1]]
        items[0]["record"] = {**items[0]["record"],"text":"Short"}
        with torch.no_grad():
            batched = model(collate_features(items))["prediction"]
            singles = torch.cat([model(collate_features([item]))["prediction"] for item in items])
        torch.testing.assert_close(batched,singles,atol=1e-5,rtol=1e-5)

    def test_source_overlap_rejected(self):
        valid = FeatureDataset(self.cfg["valid_manifest"],self.cfg)
        valid.records[0]["source_id"] = self.train.records[0]["source_id"]
        with self.assertRaisesRegex(ValueError,"Overlap"):
            validate_protocol({"train":self.train,"valid":valid})

    def test_stale_memory_rejected(self):
        cfg = {**self.cfg,"run_dir":str(Path(self.directory.name)/"stale")}
        Path(cfg["run_dir"]).mkdir(exist_ok=True)
        torch.save(self.bank,Path(cfg["run_dir"])/"memory.pt")
        with self.assertRaisesRegex(ValueError,"Stale"):
            load_bank(cfg,{"fingerprint":"different"})


class PreparationTests(unittest.TestCase):
    def test_processed_classification_import_and_numeric_mapping(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            row = {"id":"clip1","source_id":"corpus:video1","label":4,
                   "features":{"audio":np.zeros((4,8)),"video":np.ones((3,10)),
                               "text":"Observed sentence", "audio_len":2,"video_len":3}}
            with (path/"input.pkl").open("wb") as stream:
                pickle.dump([row],stream)
            (path/"labels.json").write_text(json.dumps({"4":"joy"}))
            convert_benchmark(path/"input.pkl","meld","train",path/"out.jsonl","test_features",label_map=path/"labels.json")
            result = json.loads((path/"out.jsonl").read_text())
            self.assertEqual(result["label"],"joy")
            self.assertEqual(np.load(result["audio"]).shape,(2,8))
            self.assertEqual(result["text"],"Observed sentence")

    def test_memory_import_rejects_missing_paired_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            row = {"id":"x","source_id":"video1","split":"train","text":"text",
                   "caption":"caption","label":"joy","hierarchy":["positive","joy"]}
            (path/"source.jsonl").write_text(json.dumps(row)+"\n")
            with self.assertRaisesRegex(FileNotFoundError,"Missing paired audio"):
                import_memory(path/"source.jsonl",path,path,path/"output.jsonl","test_features")


if __name__ == "__main__":
    unittest.main()
