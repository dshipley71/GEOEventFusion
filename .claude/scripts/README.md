# .claude/scripts/

Claude Code hook scripts for GEOEventFusion.

## setup.sh

**Trigger:** SessionStart hook (automatic on every Claude Code session start)  
**Purpose:** Installs Python dependencies and prepares the output directory structure in Anthropic-managed cloud environments.

### Behaviour

| Environment | Action |
|---|---|
| Claude Code web (`CLAUDE_CODE_REMOTE=true`) | Full `pip install` of all production + dev dependencies, then import verification |
| Local CLI | Prints a reminder to use `pip install -e '.[dev]'`; does nothing else |

### Making it executable

If you clone the repo on a Unix/macOS system and the execute bit is lost:

```bash
chmod +x .claude/scripts/setup.sh
```

The `settings.json` hook command includes an explicit `chmod +x` before execution,
so the script will run correctly in the cloud environment even if the execute bit
was not committed.

### Updating dependencies

If you add a new package to `pyproject.toml`, also add it to the `pip install`
block in `setup.sh` so cloud sessions pick it up automatically.

## Adding new scripts

Any script added to this directory and registered in `.claude/settings.json` will
run automatically as part of the specified hook event. See the
[Claude Code hooks documentation](https://code.claude.com/docs/en/hooks) for
supported hook events (`SessionStart`, `PreToolUse`, `PostToolUse`, `TaskCompleted`).
