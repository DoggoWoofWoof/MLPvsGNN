from scripts.modal_sa_mlp_confirmation import _remote_paths


def test_confirmation_remote_paths_are_posix_on_windows() -> None:
    result, metrics = _remote_paths("metaqa", "1234567890abcdef-rest")

    assert result == (
        "/root/message-passing-retrieval/storage/outputs/sa_mlp_confirmation/"
        "metaqa/1234567890abcdef/result.json"
    )
    assert metrics.endswith("/metaqa/1234567890abcdef/query_metrics.npz")
    assert "\\" not in result
    assert "\\" not in metrics
