from scripts.modal_sa_mlp_confirmation import CONFIG, _local_jobs, _remote_paths


def test_confirmation_remote_paths_are_posix_on_windows() -> None:
    result, metrics = _remote_paths("metaqa", "1234567890abcdef-rest")

    assert result == (
        "/root/message-passing-retrieval/storage/outputs/sa_mlp_confirmation/"
        "metaqa/1234567890abcdef/result.json"
    )
    assert metrics.endswith("/metaqa/1234567890abcdef/query_metrics.npz")
    assert "\\" not in result
    assert "\\" not in metrics


def test_only_legacy_datasets_enable_pre_hop_contract_compatibility() -> None:
    jobs = _local_jobs(list(CONFIG["datasets"]))
    compatible = {
        job["dataset"] for job in jobs if job["candidate_contract_compatibility"] is not None
    }

    assert compatible == {"2wiki_clean", "musique_clean"}
