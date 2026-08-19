import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY_ROOT / "spai-parameterized-radius-fine-tuning.ipynb"


class TestKaggleNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.code_cells = [
            "".join(cell["source"])
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        cls.source = "\n\n".join(cls.code_cells)

    def test_code_cells_are_plain_valid_python(self) -> None:
        for index, source in enumerate(self.code_cells):
            compile(source, f"{NOTEBOOK_PATH.name}:cell-{index}", "exec")
            magic_lines = [
                line for line in source.splitlines()
                if line.lstrip().startswith(("!", "%"))
            ]
            self.assertEqual(magic_lines, [])

    def test_external_commands_fail_fast(self) -> None:
        self.assertIn("subprocess.run(command, check=True)", self.source)
        self.assertIn('run_command("git", "clone"', self.source)
        self.assertIn('run_module("gdown"', self.source)
        self.assertIn('run_module("spai", *fixed_train_args)', self.source)
        self.assertIn('run_module("spai", *learnable_train_args)', self.source)
        self.assertEqual(self.source.count('"spai", "test"'), 2)

    def test_each_execution_uses_an_isolated_run_tag(self) -> None:
        self.assertIn('strftime("smoke_%Y%m%dT%H%M%S%fZ")', self.source)
        self.assertIn('f"{RUN_TAG}_fixed"', self.source)
        self.assertIn('f"{RUN_TAG}_learnable"', self.source)

    def test_training_can_resume_from_full_checkpoints(self) -> None:
        self.assertIn("FIXED_RESUME_CHECKPOINT = None", self.source)
        self.assertIn("LEARNABLE_RESUME_CHECKPOINT = None", self.source)
        self.assertEqual(
            self.source.count('extend(("--resume",'),
            2,
        )

    def test_kaggle_metadata_enables_required_services(self) -> None:
        kaggle_metadata = self.notebook["metadata"]["kaggle"]
        self.assertTrue(kaggle_metadata["isGpuEnabled"])
        self.assertTrue(kaggle_metadata["isInternetEnabled"])


if __name__ == "__main__":
    unittest.main()
