from scripts.modal_sa_mlp import _remote_result_path


def test_remote_result_path_is_posix_even_on_windows() -> None:
    path = _remote_result_path("metaqa", "1234567890abcdef-rest")

    assert path == (
        "/root/message-passing-retrieval/storage/outputs/sa_mlp_screen/"
        "metaqa/1234567890abcdef/result.json"
    )
    assert "\\" not in path
