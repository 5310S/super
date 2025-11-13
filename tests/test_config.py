import unittest
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency documented
    yaml = None

from supervisor import apply_config_defaults, build_arg_parser, load_config_data


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path("tests_tmp")
        self.tmp.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        for child in self.tmp.glob("*"):
            if child.is_file():
                child.unlink()
        self.tmp.rmdir()

    def _write(self, name: str, content: str) -> Path:
        path = self.tmp / name
        path.write_text(content, encoding="utf-8")
        return path

    @unittest.skipIf(yaml is None, "PyYAML not installed")
    def test_yaml_config_merging(self) -> None:
        config = self._write(
            "config.yaml",
            """
objective: Ship feature
max_turns: 5
builder_tool:
  - pytest -q
""".strip(),
        )
        parser = build_arg_parser()
        data = load_config_data(config)
        apply_config_defaults(parser, data)
        args = parser.parse_args([])
        self.assertEqual(args.objective, "Ship feature")
        self.assertEqual(args.max_turns, 5)
        self.assertEqual(args.builder_tool, ["pytest -q"])

    def test_json_config_merging(self) -> None:
        config = self._write(
            "config.json",
            '{"context_status_lines": 10, "auto_commit_final": true}',
        )
        parser = build_arg_parser()
        data = load_config_data(config)
        apply_config_defaults(parser, data)
        args = parser.parse_args([])
        self.assertEqual(args.context_status_lines, 10)
        self.assertTrue(args.auto_commit_final)


if __name__ == "__main__":
    unittest.main()
