# projects/

One directory per paper or release. The library (`src/egolearn/`) holds everything
reusable; a project directory holds only what is specific to that paper.

## Conventions

- `projects/<short-name>/` contains: `README.md` (title, authors, venue/status),
  `configs/` composing on the packaged tree, thin entry-point scripts, and launch
  scripts. No library code.
- Projects import from `egolearn` only — **never from another project**. When two
  projects need the same code, promote it into `src/egolearn/` via a normal PR.
- Project configs reach the packaged base via:

  ```yaml
  hydra:
    searchpath:
      - pkg://egolearn.configs
  ```

## Unpublished work

Unpublished projects live on `proj/<name>` branches pushed **only to the private
`-dev` mirror** (see RELEASING.md). On public `main`, this directory contains
released projects only.
