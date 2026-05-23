import json
import torch
import pytest

from data.dataset import TrajectoryDataset, collate_trajectories

VOCAB = {"<PAD>": 0, "<UNK>": 1, "g_100_200": 2, "g_101_201": 3, "g_102_202": 4}

SAMPLE_TRAJS = [
    {
        "user_id": "u_abc",
        "date": "2012-01-01",
        "tokens": [
            {"geocell": "g_100_200", "hour": 8, "dow": 0},
            {"geocell": "g_101_201", "hour": 12, "dow": 0},
            {"geocell": "g_102_202", "hour": 18, "dow": 0},
            {"geocell": "g_100_200", "hour": 20, "dow": 0},
        ],
    },
    {
        "user_id": "u_def",
        "date": "2012-01-01",
        "tokens": [
            {"geocell": "g_100_200", "hour": 9, "dow": 1},
            {"geocell": "g_101_201", "hour": 14, "dow": 1},
        ],
    },
]


@pytest.fixture
def dataset(tmp_path):
    jsonl_path = tmp_path / "test.jsonl"
    with open(jsonl_path, "w") as f:
        for t in SAMPLE_TRAJS:
            f.write(json.dumps(t) + "\n")
    return TrajectoryDataset(str(jsonl_path), VOCAB, max_len=64, ctx_frac=0.75)


def test_dataset_length(dataset):
    assert len(dataset) == 2


def test_dataset_item_keys(dataset):
    item = dataset[0]
    assert set(item.keys()) == {"cell_ids", "hours", "dows", "pad_mask", "ctx_mask", "tgt_mask"}


def test_dataset_shapes(dataset):
    item = dataset[0]
    T = 64
    assert item["cell_ids"].shape == (T,)
    assert item["hours"].shape == (T,)
    assert item["dows"].shape == (T,)
    assert item["pad_mask"].shape == (T,)
    assert item["ctx_mask"].shape == (T,)
    assert item["tgt_mask"].shape == (T,)


def test_masks_disjoint(dataset):
    item = dataset[0]
    assert (item["ctx_mask"] & item["tgt_mask"]).sum() == 0
    assert torch.equal(item["ctx_mask"] | item["tgt_mask"], item["pad_mask"])


def test_collate_batch_shapes(dataset):
    items = [dataset[0], dataset[1]]
    batch = collate_trajectories(items)
    B = 2
    T = 64
    assert batch["cell_ids"].shape == (B, T)
    assert batch["pad_mask"].shape == (B, T)
    assert batch["ctx_mask"].dtype == torch.bool


def test_unknown_cell_maps_to_unk(tmp_path):
    trajs = [
        {
            "user_id": "u_x",
            "date": "2012-01-01",
            "tokens": [
                {"geocell": "g_999_999", "hour": 8, "dow": 0},
                {"geocell": "g_100_200", "hour": 12, "dow": 0},
            ],
        }
    ]
    jsonl_path = tmp_path / "unk.jsonl"
    with open(jsonl_path, "w") as f:
        for t in trajs:
            f.write(json.dumps(t) + "\n")
    ds = TrajectoryDataset(str(jsonl_path), VOCAB, max_len=64)
    item = ds[0]
    assert item["cell_ids"][0].item() == VOCAB["<UNK>"]
