import pytest
import torch

from unirl.sde.runtime import ensure_sample_sigmas
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Audio, Audios, Images, Texts, Video, Videos
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams


def _joint_part(sample_ids: list[str], *, sample_rate: int = 48_000) -> Part:
    videos = Videos.from_list([Video(frames=torch.zeros(1, 3, 2, 2, dtype=torch.float32)) for _ in sample_ids])
    audios = Audios.from_list([Audio(waveform=torch.zeros(16, dtype=torch.float32)) for _ in sample_ids])
    return Part(
        sample_ids=sample_ids,
        primitives={"video": videos, "audio": audios},
        primitive_metadata={"audio": {"sample_rate": sample_rate}},
    )


def test_part_rejects_wrong_modality_key_and_batch() -> None:
    with pytest.raises(ValueError, match="canonical modality key"):
        Part(sample_ids=["p0"], primitives={"image": Texts(texts=["prompt"])})

    with pytest.raises(ValueError, match="batch 1 != sample_ids count 2"):
        Part(sample_ids=["p0", "p1"], primitives={"text": Texts(texts=["prompt"])})

    with pytest.raises(ValueError, match="batch 1 != sample_ids count 0"):
        Part(primitives={"text": Texts(texts=["prompt"])})


def test_part_rejects_invalid_primitive_metadata() -> None:
    with pytest.raises(ValueError, match="absent primitive modalities"):
        Part(sample_ids=["p0"], primitive_metadata={"audio": {"sample_rate": 48_000}})

    audio = Audios.from_list([Audio(waveform=torch.zeros(16))])
    with pytest.raises(ValueError, match="positive int"):
        Part(
            sample_ids=["p0"],
            primitives={"audio": audio},
            primitive_metadata={"audio": {"sample_rate": 0}},
        )

    with pytest.raises(TypeError, match="values must be dictionaries"):
        Part(
            sample_ids=["p0"],
            primitives={"audio": audio},
            primitive_metadata={"audio": 48_000},
        )


def test_joint_primitives_slice_concat_and_metadata_validation() -> None:
    part = _joint_part(["p0", "p1"])
    left, right = part.chunk(2)
    merged = Part.concat([left, right])

    assert set(merged.primitives) == {"video", "audio"}
    assert len(merged.primitives["video"]) == 2
    assert len(merged.primitives["audio"]) == 2
    assert merged.primitive_metadata == {"audio": {"sample_rate": 48_000}}

    with pytest.raises(ValueError, match="primitive_metadata must be identical"):
        Part.concat([left, _joint_part(["p2"], sample_rate=44_100)])


def test_conditioning_flattens_joint_map_in_canonical_order() -> None:
    audio = Audios.from_list([Audio(waveform=torch.zeros(8))])
    root = Part.input(
        ["p0"],
        primitives={"audio": audio, "text": Texts(texts=["prompt"])},
    )
    sample = Sample.request(root).fork(1, sampling_params=ARSamplingParams())

    conditioning = sample.conditioning()
    assert isinstance(conditioning[0], Texts)
    assert isinstance(conditioning[1], Audios)
    assert not hasattr(root, "primitive")


def test_has_image_input_reads_modality_map() -> None:
    pixels = torch.zeros(1, 3, 2, 2)
    root = Part.input(["p0"], primitives={"text": Texts(texts=["prompt"])})
    image = root.input_child({"image": Images(pixels=pixels)})
    sample = Sample.request(root, image).fork(1, sampling_params=ARSamplingParams())

    assert sample.has_image_input()


def test_noise_recipe_respects_driver_xt_opt_out() -> None:
    root = Part.input(["p0"], primitives={"text": Texts(texts=["prompt"])})
    frontier = root.fork(
        1,
        sampling_params=DiffusionSamplingParams(
            seed=7,
            init_noise_latent_shape=[4, 8, 8],
            disable_driver_xt=True,
        ),
    )

    recipe = NoiseRecipe.from_sample(Sample(parts=[root, frontier]))

    assert recipe.noise_group_ids == []
    assert recipe.latent_shape is None
    assert recipe.initial_latents is None
    assert recipe.resolve() is None


def test_ensure_sample_sigmas_pins_diffusion_part_and_ignores_ar_only() -> None:
    class _Policy:
        def compute_sigma(self, *, num_inference_steps: int, height: int, width: int) -> torch.Tensor:
            assert (num_inference_steps, height, width) == (2, 16, 24)
            return torch.tensor([1.0, 0.5, 0.0])

    root = Part.input(["p0"], primitives={"text": Texts(texts=["prompt"])})
    diffusion = Sample.request(root).fork(
        1,
        sampling_params=DiffusionSamplingParams(num_inference_steps=2, height=16, width=24),
    )
    ensure_sample_sigmas(diffusion, _Policy())
    assert torch.equal(diffusion.parts[-1].sampling_params.sigmas, torch.tensor([1.0, 0.5, 0.0]))

    ar_only = Sample.request(root).fork(1, sampling_params=ARSamplingParams())
    ensure_sample_sigmas(ar_only, _Policy())
