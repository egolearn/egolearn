"""Check explicit video ordering and prepared-frame integrity."""

import pytest
from PIL import Image

from egolearn.data.sources.walkingtours import WalkingToursStream


def test_walkingtours_preserves_video_order_and_frame_timestamps(tmp_path):
    for session in ("zurich", "london"):
        folder = tmp_path / session
        folder.mkdir()
        for index in (3, 1, 2):
            Image.new("RGB", (32, 32), (index, 0, 0)).save(folder / f"frame_{index}.png")
    stream = WalkingToursStream(str(tmp_path), sessions=["zurich", "london"])
    sessions = list(stream.sessions())
    assert [session.sid for session in sessions] == ["zurich", "london"]
    assert stream.total_observations() == 6
    channel = sessions[0].channels["video"]
    observations = list(channel)
    assert [obs.t for obs in observations] == pytest.approx([0, 1 / 3.75, 2 / 3.75])
    assert stream._frames("zurich") == ["frame_1.png", "frame_2.png", "frame_3.png"]
    assert stream.open_frame(observations[2].ref, 0).getpixel((0, 0)) == (3, 0, 0)


@pytest.mark.parametrize("names", [[], ["frame_1.png", "frame_3.png"], ["image.png"]])
def test_walkingtours_rejects_empty_or_incomplete_sessions(tmp_path, names):
    folder = tmp_path / "london"
    folder.mkdir()
    for name in names:
        Image.new("RGB", (32, 32)).save(folder / name)
    stream = WalkingToursStream(str(tmp_path), sessions=["london"])
    # A failed scan must not leave an unchecked cached file list.
    for _ in range(2):
        with pytest.raises(ValueError, match=r"no frames|contiguous"):
            stream.total_observations()


def test_walkingtours_rejects_missing_and_duplicate_sessions(tmp_path):
    with pytest.raises(ValueError, match="unique sessions"):
        WalkingToursStream(str(tmp_path), sessions=["london", "london"])
    with pytest.raises(FileNotFoundError):
        WalkingToursStream(str(tmp_path), sessions=["london"]).total_observations()
