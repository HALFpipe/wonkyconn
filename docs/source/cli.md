# Command-Line Interfaces

`wonkyconn` ships with two complementary command-line experiences: the traditional
non-interactive CLI and an interactive wizard. This guide walks through both
modes, highlights how to save/load configuration presets, and outlines the
essential arguments you need to supply.

## Prerequisites

Before launching either interface make sure you have:

- A BIDS-derivatives directory produced by fMRIPrep or Halfpipe.
- A participants TSV with the required `participant_id`, `gender`, and `age`
  columns.
- At least one atlas file (`_dseg.nii[.gz]`) and the label you would like to use
  for it (for example, `Schaefer2018Combined`).

## Standard CLI (non-interactive)

The core invocation pattern is:

```bash
wonkyconn --phenotypes <participants.tsv> \
          --atlas <LABEL> <atlas_dseg.nii.gz> \
          [optional flags] \
          <bids_dir> <output_dir> group
```

Key points:

- `bids_dir`, `output_dir`, and `group` are **positional arguments**; the
  analysis level is currently fixed to `group`.
- Use `--atlas` once for each atlas you want to analyse; the label must match
  the tags present in the connectivity matrices.
- `--phenotypes` must point to the TSV containing all subjects referenced in
  the connectivity matrices; rows are matched by `participant_id`.
- Optional flags such as `--verbosity 0|1|2|3` and `--debug` toggle logging and
  drop you into `pdb` on crashes.

Example (using the bundled test data paths):

```bash
wonkyconn \
  --phenotypes wonkyconn/data/test_data2/participants.tsv \
  --atlas Schaefer2018Combined wonkyconn/data/test_data2/atlas/atlas-Schaefer2018Combined_dseg.nii \
  wonkyconn/data/test_data2/derivatives/halfpipe \
  output/test-run \
  group
```
eg.
wonkyconn \
  --phenotypes /flash/PaoU/seann/wonkyconn/wonkyconn/data/test_data2/participants.tsv \
  --atlas Schaefer2018Combined /flash/PaoU/seann/wonkyconn/wonkyconn/data/test_data2/atlas/atlas-Schaefer2018Combined_dseg.nii \
  /flash/PaoU/seann/wonkyconn/wonkyconn/data/test_data2/derivatives/halfpipe \
  /flash/PaoU/seann/wonkyconn/output/test2 \
  group


### Saving and reusing configurations

Any non-interactive run can emit a JSON configuration for future reuse:

```bash
wonkyconn ... --save-config my-run.json
```

Later, supply that file via `--config` to populate the same arguments without
retyping them:

```bash
wonkyconn --config my-run.json
```

You can still override individual options (for example, a different output
directory or atlas) on the command line when using `--config`.

## Wizard CLI (interactive)

Launch the wizard with:

```bash
wonkyconn --wizard
```

After optionally selecting a light/dark theme, the wizard will prompt you for:

1. BIDS derivatives directory.
2. Output directory.
3. Phenotypes TSV (auto-detected beneath the BIDS tree when possible).
4. Atlas file (with on-the-fly label inference).
5. Advanced options (group-by tags and verbosity) if you opt in.
6. Whether to enable the debugger on crash.

A summary of your selections is shown before the run starts. Choosing “no” at
this stage cancels execution without side effects.

### Working with configuration presets

The wizard honours the same `--config` file used by the non-interactive CLI;
any values found in the JSON preset are treated as defaults when the prompts are
shown. Likewise, provide `--save-config` to persist the final selections:

```bash
wonkyconn --wizard --config my-run.json --save-config updated.json
```

This is particularly helpful when you want the interactive confirmation but also
wish to reuse the configuration later in a scripted environment.

## Troubleshooting tips

- If the wizard cannot auto-detect an atlas, ensure the atlas file lives inside
  the BIDS directory tree or provide an absolute path when prompted.
- `ValueError: No groups found` usually means the atlas label did not match any
  connectivity matrices under the selected derivatives directory. Double-check
  the provided label.
- Run with `--verbosity 3` (or answer the corresponding wizard prompt) to get
  detailed debug logging written to the terminal.

With these commands you can comfortably switch between scripted automation and
an exploratory wizard-driven workflow while sharing the same configuration
presets.
