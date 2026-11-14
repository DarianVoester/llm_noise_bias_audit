"""BYOM registry: TOML parsing, validation, URL policy, append/remove."""

import pytest

from src.config import CUSTOM_MODELS_EXAMPLE_FILE
from src.model_registry import (
    CustomModelSpec,
    append_custom_model,
    custom_cloud_spec,
    load_custom_registry,
    remove_custom_model,
)
from src.model_spec import ModelSource

VALID = """
[[model]]
id = "llamacpp-ft"
display_name = "My fine-tune"
model = "my-finetune"
api_base = "http://127.0.0.1:8080/v1"
supports_json_mode = false
max_concurrency = 1
[model.default_params]
top_p = 0.9

[[model]]
id = "ollama-x"
provider = "ollama"
model = "my-model:latest"
api_base = "http://127.0.0.1:11434"
"""


def test_example_file_is_valid_toml_with_no_active_entries():
    specs, errors = load_custom_registry(CUSTOM_MODELS_EXAMPLE_FILE)
    assert errors == [] and specs == []  # everything is commented out


def test_example_file_entries_are_valid_when_uncommented(tmp_path):
    text = CUSTOM_MODELS_EXAMPLE_FILE.read_text()
    import re

    setting = re.compile(r"^# (\[\[?model[\].]|[a-z_]+\s+= )")  # commented-out TOML only
    uncommented = "\n".join(
        line[2:] if setting.match(line) else line
        for line in text.splitlines()
        if not line.startswith("# allow_public_endpoint")
    )
    p = tmp_path / "models.toml"
    p.write_text(uncommented)
    specs, errors = load_custom_registry(p)
    assert errors == [], errors
    assert len(specs) == 5


def test_load_valid(tmp_path):
    p = tmp_path / "models.toml"
    p.write_text(VALID)
    specs, errors = load_custom_registry(p)
    assert errors == [] and [s.id for s in specs] == ["llamacpp-ft", "ollama-x"]
    ms = specs[0].to_model_spec()
    assert ms.id == "custom/llamacpp-ft" and ms.model_route == "openai/my-finetune"
    assert ms.source == ModelSource.CUSTOM and ms.default_params == {"top_p": 0.9}
    assert specs[1].to_model_spec().model_route == "ollama_chat/my-model:latest"


def test_missing_file_is_fine(tmp_path):
    assert load_custom_registry(tmp_path / "nope.toml") == ([], [])


@pytest.mark.parametrize(
    "entry,fragment",
    [
        ('id = "Bad ID"\nmodel = "m"\napi_base = "http://127.0.0.1:1/v1"', "id"),
        ('id = "x"\nmodel = "m"\napi_base = "https://api.evil.com/v1"', "not local"),
        ('id = "x"\nmodel = "m"\napi_base = "http://169.254.169.254/"', "link-local"),
        ('id = "x"\nmodel = "m; rm -rf /"\napi_base = "http://127.0.0.1:1/v1"', "model"),
        ('id = "x"\nmodel = "m"\napi_base = "http://127.0.0.1:1/v1"\napi_key_env = "sk-abc123"',
         "api_key_env"),
        ('id = "x"\nmodel = "m"\napi_base = "http://127.0.0.1:1/v1"\n'
         '[model.default_params]\napi_key = "x"', "default_params"),
        ('id = "x"\napi_base = "http://127.0.0.1:1/v1"', "model"),
    ],
)
def test_invalid_entries_reported_and_skipped(tmp_path, entry, fragment):
    p = tmp_path / "models.toml"
    p.write_text(VALID + "\n[[model]]\n" + entry + "\n")
    specs, errors = load_custom_registry(p)
    assert len(specs) == 2  # the valid ones still load
    assert len(errors) == 1 and fragment in str(errors[0])


def test_toml_syntax_error(tmp_path):
    p = tmp_path / "models.toml"
    p.write_text("[[model]\nid = ")
    specs, errors = load_custom_registry(p)
    assert specs == [] and "syntax" in str(errors[0])


def test_duplicate_ids(tmp_path):
    p = tmp_path / "models.toml"
    p.write_text(VALID + '\n[[model]]\nid = "ollama-x"\nmodel = "m"\n'
                         'api_base = "http://127.0.0.1:1/v1"\n')
    specs, errors = load_custom_registry(p)
    assert len(specs) == 2 and "duplicate" in str(errors[0])


def test_public_endpoint_opt_in(tmp_path):
    p = tmp_path / "models.toml"
    p.write_text('[[model]]\nid = "pub"\nmodel = "m"\napi_base = "https://my.server.example/v1"\n'
                 "allow_public_endpoint = true\n")
    specs, errors = load_custom_registry(p)
    assert errors == [] and specs[0].id == "pub"


def test_append_and_remove_roundtrip(tmp_path):
    p = tmp_path / "models.toml"
    spec = CustomModelSpec(id="new-one", model="qwen", api_base="http://localhost:1234/v1",
                           notes='has "quotes" and \\ backslash', default_params={"top_k": 20})
    append_custom_model(spec, p)
    append_custom_model(CustomModelSpec(id="second", model="m",
                                        api_base="http://127.0.0.1:8000/v1"), p)
    specs, errors = load_custom_registry(p)
    assert errors == [] and [s.id for s in specs] == ["new-one", "second"]
    assert specs[0].notes == 'has "quotes" and \\ backslash'
    assert specs[0].default_params == {"top_k": 20}
    with pytest.raises(ValueError):
        append_custom_model(spec, p)
    assert remove_custom_model("new-one", p) is True
    assert [s.id for s in load_custom_registry(p)[0]] == ["second"]
    assert remove_custom_model("missing", p) is False


@pytest.mark.parametrize("env", ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GITHUB_TOKEN",
                                 "AWS_SECRET_ACCESS_KEY"])
def test_keys_of_other_services_are_not_sent_to_custom_endpoints(env):
    with pytest.raises(ValueError, match="another service"):
        CustomModelSpec(id="x", model="m", api_base="http://127.0.0.1:8080/v1", api_key_env=env)


def test_key_needs_https_on_public_endpoints():
    with pytest.raises(ValueError, match="https"):
        CustomModelSpec(id="x", model="m", api_base="http://my.server.example/v1",
                        api_key_env="MY_SERVER_KEY", allow_public_endpoint=True)
    ok = dict(id="x", model="m", api_key_env="MY_SERVER_KEY")
    assert CustomModelSpec(api_base="https://my.server.example/v1", allow_public_endpoint=True,
                           **ok)
    assert CustomModelSpec(api_base="http://127.0.0.1:8080/v1", **ok)  # local: http is fine
    assert CustomModelSpec(id="y", model="m", api_base="http://my.server.example/v1",
                           allow_public_endpoint=True)  # no key, nothing to protect


def test_control_characters_do_not_break_the_registry(tmp_path):
    p = tmp_path / "models.toml"
    spec = CustomModelSpec(id="ctl", model="m", api_base="http://127.0.0.1:8080/v1",
                           notes="tab\there, cr\rthere, bell\x07", default_params={"a b": 1})
    append_custom_model(spec, p)
    specs, errors = load_custom_registry(p)
    assert errors == [] and specs[0].notes == "tab\there, cr\rthere, bell\x07"
    assert specs[0].default_params == {"a b": 1}
    with pytest.raises(ValueError):  # urlparse would silently drop the line break
        append_custom_model(CustomModelSpec(id="bad", model="m",
                                            api_base="http://127.0.0.1:8080/v1\r\nx"), p)
    trimmed = CustomModelSpec(id="trim", model="m", api_base=" http://127.0.0.1:8080/v1\r\n")
    assert trimmed.api_base == "http://127.0.0.1:8080/v1"


def test_remove_keeps_file_and_leaves_no_temp_file(tmp_path):
    p = tmp_path / "models.toml"
    append_custom_model(CustomModelSpec(id="only", model="m",
                                        api_base="http://127.0.0.1:8000/v1"), p)
    assert remove_custom_model("only", p) is True
    assert p.is_file() and load_custom_registry(p) == ([], [])
    assert [f.name for f in tmp_path.iterdir()] == ["models.toml"]


def test_custom_cloud_spec():
    s = custom_cloud_spec("openrouter/qwen/qwen3-8b")
    assert s.provider == "openrouter" and s.source == ModelSource.CLOUD
    with pytest.raises(ValueError):
        custom_cloud_spec("no-provider")
    with pytest.raises(ValueError):
        custom_cloud_spec("unknownprov/model")
