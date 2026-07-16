from __future__ import annotations

import argparse

import pytest

from barellm import cli


# --- Parser tests -----------------------------------------------------------


class TestBuildParser:
    def test_returns_argument_parser(self) -> None:
        parser = cli.build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_no_command_sets_none(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args([])
        assert args.command is None

    @pytest.mark.parametrize(
        ("argv", "expected_command"),
        [
            (["pull", "Qwen/Qwen3-0.6B"], "pull"),
            (["ls"], "ls"),
            (["rm", "Qwen/Qwen3-0.6B"], "rm"),
            (["run", "hi"], "run"),
            (["serve"], "serve"),
        ],
    )
    def test_subcommands_registered(
        self, argv: list[str], expected_command: str
    ) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(argv)
        assert args.command == expected_command


class TestPullParser:
    def test_defaults_model(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["pull", "Qwen/Qwen3-0.6B"])
        assert args.model == "Qwen/Qwen3-0.6B"

    def test_custom_model(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["pull", "meta-llama/Llama-3-8B"])
        assert args.model == "meta-llama/Llama-3-8B"


class TestRmParser:
    def test_takes_model_positional(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["rm", "Qwen/Qwen3-0.6B"])
        assert args.model == "Qwen/Qwen3-0.6B"


class TestRunParser:
    def test_defaults(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["run", "hello"])
        assert args.prompt == "hello"
        assert args.max_new_tokens == 2048
        assert args.temperature == 0.7
        assert args.top_p == 0.9
        assert args.model == "Qwen/Qwen3-0.6B"
        assert args.local_files_only is False

    def test_custom_flags(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "run",
                "hello",
                "--max_new_tokens",
                "100",
                "--temperature",
                "0.1",
                "--top_p",
                "0.5",
                "--model",
                "custom/model",
                "--local_files_only",
            ]
        )
        assert args.max_new_tokens == 100
        assert args.temperature == 0.1
        assert args.top_p == 0.5
        assert args.model == "custom/model"
        assert args.local_files_only is True

    def test_prompt_is_required(self) -> None:
        parser = cli.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run"])


# --- Dispatch tests ---------------------------------------------------------


class TestMainDispatch:
    def test_no_command_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sys.argv", ["barellm"])
            cli.main()
        out = capsys.readouterr().out
        assert "LLM inference engine" in out

    def test_serve_not_implemented(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sys.argv", ["barellm", "serve"])
        cli.main()
        out = capsys.readouterr().out
        assert "not implemented" in out

    def test_pull_routes_to_handler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[str] = []
        monkeypatch.setattr(cli, "COMMANDS", {"pull": lambda a: called.append(a.model)})
        monkeypatch.setattr("sys.argv", ["barellm", "pull", "some/model"])
        cli.main()
        assert called == ["some/model"]

    def test_run_routes_to_handler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[str] = []
        monkeypatch.setattr(cli, "COMMANDS", {"run": lambda a: seen.append(a.prompt)})
        monkeypatch.setattr("sys.argv", ["barellm", "run", "ping"])
        cli.main()
        assert seen == ["ping"]

    def test_unknown_command_falls_through(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(cli, "COMMANDS", {})
        monkeypatch.setattr("sys.argv", ["barellm", "serve"])
        cli.main()
        out = capsys.readouterr().out
        assert "not implemented" in out


# --- Handler tests (hub/generate mocked) ------------------------------------


class TestCmdPull:
    def test_calls_download_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import barellm.hub as hub

        calls: list[str] = []
        monkeypatch.setattr(hub, "download_model", lambda m: calls.append(m))

        args = argparse.Namespace(model="Qwen/Qwen3-0.6B")
        cli.cmd_pull(args)
        assert calls == ["Qwen/Qwen3-0.6B"]


class TestCmdLs:
    def test_lists_models(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import barellm.hub as hub

        monkeypatch.setattr(hub, "list_models", lambda: ["Qwen/Qwen3-0.6B", "other/x"])
        cli.cmd_ls(argparse.Namespace())
        out = capsys.readouterr().out
        assert "Downloaded models:" in out
        assert "Qwen/Qwen3-0.6B" in out
        assert "other/x" in out

    def test_no_models_message(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import barellm.hub as hub

        monkeypatch.setattr(hub, "list_models", lambda: [])
        cli.cmd_ls(argparse.Namespace())
        out = capsys.readouterr().out
        assert "No models downloaded." in out


class TestCmdRm:
    def test_removed_successfully(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import barellm.hub as hub

        monkeypatch.setattr(hub, "remove_model", lambda m: True)
        cli.cmd_rm(argparse.Namespace(model="Qwen/Qwen3-0.6B"))
        out = capsys.readouterr().out
        assert "removed successfully" in out

    def test_not_found(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import barellm.hub as hub

        monkeypatch.setattr(hub, "remove_model", lambda m: False)
        cli.cmd_rm(argparse.Namespace(model="missing/model"))
        out = capsys.readouterr().out
        assert "not found" in out


class TestCmdRun:
    def test_passes_all_args_to_generate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import barellm.generate as generate_mod

        captured: dict[str, object] = {}

        def fake_generate(prompt, max_new_tokens, temperature, top_p, local_files_only):
            captured.update(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                local_files_only=local_files_only,
            )
            return "generated-output"

        monkeypatch.setattr(generate_mod, "generate", fake_generate)

        args = argparse.Namespace(
            prompt="hi",
            max_new_tokens=64,
            temperature=0.2,
            top_p=0.8,
            local_files_only=True,
        )
        cli.cmd_run(args)

        assert captured["prompt"] == "hi"
        assert captured["max_new_tokens"] == 64
        assert captured["temperature"] == 0.2
        assert captured["top_p"] == 0.8
        assert captured["local_files_only"] is True

    def test_prints_generated_output(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import barellm.generate as generate_mod

        monkeypatch.setattr(generate_mod, "generate", lambda *a, **k: "hello world")
        args = argparse.Namespace(
            prompt="x",
            max_new_tokens=1,
            temperature=0.0,
            top_p=1.0,
            local_files_only=False,
        )
        cli.cmd_run(args)
        out = capsys.readouterr().out
        assert "hello world" in out


# --- Sanity: add_run_parser signature --------------------------------------


class TestAddRunParser:
    def test_attaches_run_subcommand(self) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        cli.add_run_parser(subparsers)
        args = parser.parse_args(["run", "hello"])
        assert args.command == "run"
        assert args.prompt == "hello"
        assert args.max_new_tokens == 2048  # default still applies via add_run_parser
