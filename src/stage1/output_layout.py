from __future__ import annotations


_STAGE1_ARTIFACT_DIRS = {
    "boiler": "boilerCCS",
    "boiler_ccs": "boilerCCS",
    "te": "TE",
    "tennessee_eastman": "TE",
}


def stage1_artifact_dir(platform: str) -> str:
    value = str(platform).strip().lower()
    try:
        return _STAGE1_ARTIFACT_DIRS[value]
    except KeyError as exc:
        raise ValueError(f"unsupported stage1 artifact platform: {platform}") from exc
