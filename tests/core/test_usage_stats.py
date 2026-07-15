from free_claude_code.core.usage_stats import ModelUsageStats, UsageStatsTracker


def test_record_success_accumulates_requests_and_tokens(tmp_path):
    tracker = UsageStatsTracker(str(tmp_path / "usage_stats.json"))

    tracker.record_success("open_router", "openai/gpt-oss-20b:free", input_tokens=100)
    tracker.record_success("open_router", "openai/gpt-oss-20b:free", input_tokens=50)

    stats = tracker.snapshot()["models"]["open_router/openai/gpt-oss-20b:free"]
    assert stats["requests"] == 2
    assert stats["input_tokens"] == 150
    assert stats["errors"] == 0
    assert stats["last_used_at"]


def test_record_error_increments_errors_without_touching_tokens(tmp_path):
    tracker = UsageStatsTracker(str(tmp_path / "usage_stats.json"))

    tracker.record_success("gemini", "gemini-3.1-flash-lite", input_tokens=10)
    tracker.record_error("gemini", "gemini-3.1-flash-lite")
    tracker.record_error("gemini", "gemini-3.1-flash-lite")

    stats = tracker.snapshot()["models"]["gemini/gemini-3.1-flash-lite"]
    assert stats["requests"] == 1
    assert stats["errors"] == 2
    assert stats["input_tokens"] == 10


def test_distinct_models_are_tracked_independently(tmp_path):
    tracker = UsageStatsTracker(str(tmp_path / "usage_stats.json"))

    tracker.record_success("open_router", "model-a", input_tokens=10)
    tracker.record_success("gemini", "model-a", input_tokens=20)

    models = tracker.snapshot()["models"]
    assert models["open_router/model-a"]["input_tokens"] == 10
    assert models["gemini/model-a"]["input_tokens"] == 20


def test_load_restores_counters_persisted_by_a_previous_process(tmp_path):
    storage_path = str(tmp_path / "usage_stats.json")
    tracker = UsageStatsTracker(storage_path)
    tracker.record_success("open_router", "model-a", input_tokens=42)
    tracker._persistence.flush()

    restored = UsageStatsTracker(storage_path)
    restored.load()

    stats = restored.snapshot()["models"]["open_router/model-a"]
    assert stats["requests"] == 1
    assert stats["input_tokens"] == 42


def test_load_ignores_missing_or_malformed_file(tmp_path):
    tracker = UsageStatsTracker(str(tmp_path / "does-not-exist.json"))

    tracker.load()

    assert tracker.snapshot()["models"] == {}


def test_model_usage_stats_from_json_defaults_malformed_fields():
    stats = ModelUsageStats.from_json(
        {"requests": "not-an-int", "errors": 2, "input_tokens": None}
    )

    assert stats == ModelUsageStats(requests=0, errors=2, input_tokens=0)


def test_model_usage_stats_from_json_ignores_non_dict_input():
    assert ModelUsageStats.from_json("not-a-dict") == ModelUsageStats()
