from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from diagnostics.week2_partition import build_manifest, paired_sensitivity, parse_filenames


def records(n=20):
    data = []
    for pid in range(1,n+1):
        for cam in (1,2,3)[:1 + (pid % 3)]:
            data.extend(f"{pid:04}_c{cam}s2_{f:06}_00.jpg" for f in range(1+pid%2))
    return data


def plan(names,seed=20261122):
    return build_manifest(names,calibration_identities=3,evaluation_identities=4,
                          min_train_identities=3,seed=seed,exposed_ids=[1,2,4])


class TestWeek2(unittest.TestCase):
    def test_disjoint_cross_camera_query_and_prior_dev_exclusions(self):
        local, audit = plan(records())
        tr,c,e = [set(local["identity_sets"][x]) for x in ("train","calibration","evaluation")]
        self.assertEqual((len(tr),len(c),len(e)),(13,3,4))
        self.assertFalse((tr&c)|(tr&e)|(c&e))
        self.assertTrue({1,2,4} <= tr)
        for role in ("calibration","evaluation"):
            q,g = [local["records"][f"{role}_{x}"] for x in ("query","gallery")]
            self.assertEqual(len(q),len({r["identity"] for r in q}))
            for row in q:
                self.assertTrue(any(r["identity"]==row["identity"] and r["camera"]!=row["camera"]
                                    for r in g))
            self.assertEqual(audit[role]["queries_with_cross_camera_positive"],len(q))

    def test_order_invariance_seed_sensitivity_and_hashes(self):
        names = records()
        l1,a1=plan(names)
        random.Random(31).shuffle(names)
        l2,a2=plan(names)
        l3,a3=plan(names,seed=8)
        self.assertEqual((l1,a1),(l2,a2))
        self.assertEqual(a1["source_names_sha256"],a3["source_names_sha256"])
        self.assertNotEqual(a1["partition_manifest_sha256"],a3["partition_manifest_sha256"])
        payload=json.dumps(l1,sort_keys=True,ensure_ascii=True,separators=(",",":")).encode()
        self.assertEqual(a1["partition_manifest_sha256"],sha256(payload).hexdigest())
        self.assertNotIn(".jpg",json.dumps(a1))
        self.assertFalse(a1["model_ancestry_verified"])
        self.assertFalse(a1["scientific_independence_verified"])

    def test_input_errors_and_negatives(self):
        rows, skipped=parse_filenames(["-1_c1_0.jpg","0000_c1_0.jpg","0001_c1_0.jpg","README.txt"])
        self.assertEqual(len(rows),1)
        self.assertEqual(skipped,{"reserved_nonpositive_identity_images":2,"non_image_entries":1})
        with self.assertRaises(ValueError):
            parse_filenames(["../../0001_c1_0.jpg"])
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            parse_filenames(["0001_c1_0.jpg","0001_c1_0.JPG"])
        with self.assertRaisesRegex(ValueError,"syntax"):
            parse_filenames(["arbitrary.JPG"])
        with self.assertRaisesRegex(ValueError,"insufficient|Too few"):
            plan(records(7))
        with self.assertRaisesRegex(ValueError,"absent"):
            build_manifest(records(),calibration_identities=1,evaluation_identities=1,
                           min_train_identities=1,exposed_ids=[999])

    def test_sensitivity(self):
        a=paired_sensitivity(150)
        self.assertEqual(a["target_delta_pp"],2)
        ns=[]
        for x in a["scenarios"]:
            ns.append(x["approx_identity_units_needed"])
            self.assertEqual(x["one_net_query_pp"],2/3)
            self.assertEqual(x["min_all_same_direction_discordances_exact_sign"],6)
            d=x["design_effect_sensitivity"]
            self.assertEqual(d[2]["effective_identity_units"],75)
            self.assertEqual(d[2]["approx_raw_identity_units_needed"],ns[-1]*2)
        self.assertEqual(ns,sorted(ns))

    def test_real_cli_shape_without_decodable_images(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)
            train=base/"Market-1501"/"bounding_box_train"
            train.mkdir(parents=True)
            for n in records():
                (train/n).write_bytes(b"deliberately not image content")
            receipt=base/"prior.json"
            receipt.write_text('{"excluded_identity_ids":[1,2,4]}')
            target=base/"frozen"
            cmd=[sys.executable,"-m","diagnostics.week2_partition","--train-dir",str(train),
                 "--output-dir",str(target),"--calibration-identities","3","--evaluation-identities","4",
                 "--prior-dev-ids",str(receipt),"--expect-prior-dev-count","3","--min-train-identities","3"]
            first=subprocess.run(cmd,check=True,capture_output=True,text=True)
            self.assertNotIn(".jpg",first.stdout)
            self.assertEqual(json.loads(first.stdout),json.loads((target/"aggregate_audit.json").read_text()))
            tr=json.loads((target/"local_train_only.json").read_text())
            self.assertEqual(set(tr["records"]),{"model_train"})
            self.assertEqual(len(list(target.iterdir())),5)
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)

if __name__ == "__main__":
    unittest.main()
