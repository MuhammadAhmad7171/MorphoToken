import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "scripts" / "morphotoken.py"
ABLATION = ROOT / "scripts" / "morphology_ablation.py"


class RepositoryIntegrityTests(unittest.TestCase):
    def test_sources_parse(self):
        for path in (PRIMARY, ABLATION):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_primary_contains_expected_m3_kernels(self):
        text = PRIMARY.read_text(encoding="utf-8")
        self.assertIn("for k in (3, 5, 7):", text)
        self.assertIn("torch.cat([x] + grads", text)

    def test_ablation_exposes_all_morphology_variants(self):
        text = ABLATION.read_text(encoding="utf-8")
        for variant in ("M0", "M1", "M2", "M3", "M4", "M5", "M6"):
            self.assertIn(f'"{variant}"', text)

    def test_ablation_reproduction_script_has_paper_weights(self):
        text = (ROOT / "scripts" / "reproduce_morphology_ablation.sh").read_text(encoding="utf-8")
        self.assertIn("--class-weights 1,1,1,1.1", text)
        self.assertIn("--morphology-ablation-seeds 2026,3407,5891", text)

    def test_tta_documentation_matches_implementation(self):
        for path in (PRIMARY, ABLATION):
            text = path.read_text(encoding="utf-8")
            self.assertIn("Average original and horizontal-flip logits before softmax", text)
            self.assertIn('logits = 0.5 * (logits + model(torch.flip(x, dims=(-1,)))["logits"])', text)


if __name__ == "__main__":
    unittest.main()
