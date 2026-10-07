# pic-pick

Groups JPEGs of the same scene and has an LLM rank each group to pick the best shot.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- `pi` CLI with access to the model given by `--model` (default `gpt-6-luna`)

## Usage

```sh
# 1. Group similar photos
uv run pic_pick.py group <photo-folder>

# 2. (Optional) edit runs/<folder>-<hash>/groups.json to split or merge groups

# 3. Rank each group and write report.json
uv run pic_pick.py rank <photo-folder>
```

If `groups.json` does not exist yet, `rank` runs the grouping first.

Options:

| Option | Default | Description |
|---|---|---|
| `--gap` | 2 | Max seconds between photos in the same group |
| `--dhash` | 10 | Similarity threshold for grouping (0–64, lower is stricter) |
| `--jobs` | 4 | Parallel workers, also the max concurrent `pi` calls |
| `--model` | gpt-6-luna | Model used by `rank` and `learn` |

## Output

`runs/<folder>-<hash>/report.json`. Each group has:

- `best`: the AI's pick
- `ranking`: the whole group, best first
- `flags`: issues found (`closed_eyes`, `cropped_limbs`, `bad_expression`)
- `needs_review`: the final ran twice and the results disagreed; `final_alt_best` is the other pick, so check it yourself
- `scores`: technical scores (sharpness, over/underexposed ratio, slow-shutter/high-ISO risks)

Results are cached in `runs/`, so a rerun only does the unfinished work. To start over, delete the folder's `runs/` subdirectory.

## Teaching it your taste

```sh
# When you disagree with the AI, record your pick (group number = id in report.json)
uv run pic_pick.py override <photo-folder> <group> <file> --note "why"

# After collecting some overrides, have the LLM summarize your preferences
uv run pic_pick.py learn
```

`learn` only prints suggested notes. Review them and save them as `taste.md` next to `pic_pick.py`; `rank` uses that file from then on. Changing `taste.md` invalidates cached match results, so the next `rank` calls the LLM again.
