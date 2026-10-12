import json
import logging
import io

from src.utils.config import load_config, tracking_uri
from src.utils.logging_utils import JsonFormatter
from src.utils.prompts import build_messages, list_versions, load_prompt


def test_prompt_versions_exist_and_differ():
    cfg = load_config()
    assert list_versions(cfg) == ["v1", "v2"]
    v1, text1 = load_prompt(cfg, "v1")
    v2, text2 = load_prompt(cfg, "v2")
    assert (v1, v2) == ("v1", "v2") and text1 != text2 and "AuroraSync" in text1


def test_unknown_prompt_version_raises():
    cfg = load_config()
    try:
        load_prompt(cfg, "v99")
    except FileNotFoundError as exc:
        assert "v99" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_build_messages_roles_and_strip():
    msgs = build_messages("sys", "  hello \n")
    assert msgs == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]


def test_config_has_required_sections():
    cfg = load_config()
    for key in ("model", "prompt", "data", "lora", "train", "mlflow", "eval", "gate", "monitoring", "retrain"):
        assert key in cfg
    assert cfg["model"]["base_name"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert set(cfg["mlflow"]["aliases"]) == {"candidate", "production"}


def test_tracking_uri_is_absolute_for_relative_sqlite():
    uri = tracking_uri(load_config())
    assert uri.startswith("sqlite:////") or uri.startswith("http")


def test_json_log_format():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("test_json_log")
    logger.handlers, logger.propagate = [handler], False
    logger.setLevel(logging.INFO)
    logger.info("hello %s", "world", extra={"fields": {"request_id": "abc", "msg": "clash"}})
    payload = json.loads(stream.getvalue())
    assert payload["msg"] == "hello world" and payload["level"] == "INFO"
    assert payload["request_id"] == "abc" and payload["field_msg"] == "clash" and "ts" in payload
