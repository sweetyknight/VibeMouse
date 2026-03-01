from __future__ import annotations

from pathlib import Path

import openvino as ov
from openvino import Dimension, PartialShape


def build_bounded_models(src_dir: Path, dst_dir: Path) -> None:
    core = ov.Core()
    dst_dir.mkdir(parents=True, exist_ok=True)

    audio = core.read_model(str(src_dir / "openvino_thinker_audio_model.xml"))
    audio.reshape({"padded_feature": PartialShape([1, 128, Dimension(80, 2048)])})
    ov.save_model(audio, str(dst_dir / "openvino_thinker_audio_model.xml"))

    audio_encoder = core.read_model(
        str(src_dir / "openvino_thinker_audio_encoder_model.xml")
    )
    audio_encoder.reshape(
        {
            "hidden_states": PartialShape([Dimension(64, 4096), 896]),
            "cu_seqlens": PartialShape([Dimension(2, 512)]),
        }
    )
    ov.save_model(
        audio_encoder, str(dst_dir / "openvino_thinker_audio_encoder_model.xml")
    )

    embedding = core.read_model(str(src_dir / "openvino_thinker_embedding_model.xml"))
    embedding.reshape({"input": PartialShape([1, Dimension(1, 2048)])})
    ov.save_model(embedding, str(dst_dir / "openvino_thinker_embedding_model.xml"))

    language = core.read_model(str(src_dir / "openvino_thinker_language_model.xml"))
    language.reshape(
        {
            "attention_mask": PartialShape([1, Dimension(1, 2048)]),
            "position_ids": PartialShape([3, 1, Dimension(1, 2048)]),
            "inputs_embeds": PartialShape([1, Dimension(1, 2048), 1024]),
            "beam_idx": PartialShape([1]),
        }
    )
    ov.save_model(language, str(dst_dir / "openvino_thinker_language_model.xml"))


def check_npu_compile(model_dir: Path) -> list[tuple[str, bool, str]]:
    core = ov.Core()
    files = [
        "openvino_thinker_audio_model.xml",
        "openvino_thinker_audio_encoder_model.xml",
        "openvino_thinker_embedding_model.xml",
        "openvino_thinker_language_model.xml",
    ]
    result: list[tuple[str, bool, str]] = []
    for name in files:
        path = str(model_dir / name)
        try:
            compiled = core.compile_model(path, "NPU")
            exec_devs = str(compiled.get_property("EXECUTION_DEVICES"))
            result.append((name, True, exec_devs))
        except Exception as error:
            result.append((name, False, str(error).splitlines()[0]))
    return result


def main() -> None:
    src = Path("tmp/qwen3_ov_model/thinker")
    dst = Path("tmp/qwen3_ov_model_static/thinker")
    build_bounded_models(src, dst)
    print("Bounded models saved to", dst)
    for name, ok, message in check_npu_compile(dst):
        print(name, "NPU_COMPILE_OK" if ok else "NPU_COMPILE_FAIL", message)


if __name__ == "__main__":
    main()
