# video/vbench

[VBench](https://github.com/Vchitect/VBench) prompt suites (Apache-2.0, vendored in
`data/`): `all_dimension.txt` (946) and `all_category.txt` (800). Generation and scoring
both use the official toolkit — the runner does not drive video models.

Official protocol: 5 videos per prompt named `<full prompt text>-<i>.mp4` (i = 0..4,
random recorded seeds); the `temporal_flickering` dimension needs 25 videos/prompt plus
its `static_filter.py` pre-pass. Score with the `vbench` pip package
(`vbench evaluate --videos_path ... --dimension ...`; model weights auto-download).

Generate with your UniRL-trained video checkpoint via its recipe pipeline (wan21/22,
hunyuan_video15, ltx2) or plain diffusers, honoring the naming convention above.
