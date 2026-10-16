# pic-pick

Groups JPEGs of the same scene and has [Jev](https://docs.typesafe.ai) score each photo to pick the best shot of each group.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- `TYPESAFE_API_KEY` set to a TypeSafe API key

## Usage

```sh
# 1. Group similar photos
uv run pic_pick.py group <photo-folder>

# 2. (Optional) edit runs/<folder>-<hash>/groups.json to split or merge groups

# 3. Score each photo and write report.json
uv run pic_pick.py rank <photo-folder>
```

If `groups.json` does not exist yet, `rank` runs the grouping first.

Options:

| Option | Default | Description |
|---|---|---|
| `--gap` | 60 | A photo may join a group that has a photo taken within this many seconds of it |
| `--dhash` | 10 | Max dHash distance between any two photos in a group (0–64, lower is stricter) |
| `--jobs` | 4 | Parallel workers, also the max concurrent Jev calls |
| `--model` | jev-latest | Jev model used by `rank` |

## How ranking works

Each photo in a group of two or more gets one Jev request with four Scores (`focus`, `exposure`, `composition`, `moment`) and three yes/no flags (`closed_eyes`, `cropped_limbs`, `bad_expression`). The code combines them with `WEIGHTS`, the photo's sharpness relative to the group, and `FLAG_PENALTY`, then sorts the group by that total. Tune these constants in `pic_pick.py`; changing them does not call Jev again.

## Output

`runs/<folder>-<hash>/report.json`. Each group has:

- `best`: the top-scoring photo
- `ranking`: the whole group, best first
- `totals`: the combined score of each photo
- `flags`: flags with probability above 0.5
- `needs_review`: the top two totals are within `REVIEW_GAP`; `alt_best` is the runner-up, so check it yourself
- `scores`: technical scores (sharpness, over/underexposed ratio, slow-shutter/high-ISO risks) and the raw Jev answers under `jev`

Each photo is decoded once into a thumbnail (longest side 1568 px) at `runs/<folder>-<hash>/thumbs/<sha>.jpg`; the technical scores, Jev, and any viewer use it.

Jev answers are cached per photo in `runs/`, so a rerun only calls Jev for new photos. To start over, delete the folder's `runs/` subdirectory.
