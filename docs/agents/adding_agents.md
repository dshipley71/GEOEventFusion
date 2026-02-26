# GEOEventFusion — How to Add a New Agent

Follow this checklist whenever you want to add a new agent to the pipeline.

---

## Step 1: Define the Output Model

Add a new typed dataclass in `geoeventfusion/models/`:

```python
# geoeventfusion/models/my_feature.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List

@dataclass
class MyFeatureAgentResult:
    """Complete output from the MyFeatureAgent."""
    items: List[Any] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    status: str = "OK"   # "OK", "PARTIAL", "CRITICAL"
```

Export it from `geoeventfusion/models/__init__.py`:

```python
from geoeventfusion.models.my_feature import MyFeatureAgentResult
__all__ = [..., "MyFeatureAgentResult"]
```

---

## Step 2: Add the Result Field to PipelineContext

In `geoeventfusion/models/pipeline.py`, add a field to `PipelineContext`:

```python
@dataclass
class PipelineContext:
    ...
    my_feature_result: Optional[Any] = None   # MyFeatureAgentResult
```

---

## Step 3: Implement the Agent

Create `geoeventfusion/agents/my_feature_agent.py`:

```python
from __future__ import annotations
import logging
from typing import Any
from geoeventfusion.agents.base import BaseAgent
from geoeventfusion.models.my_feature import MyFeatureAgentResult

logger = logging.getLogger(__name__)

class MyFeatureAgent(BaseAgent):
    """One-line description of what this agent does."""

    name = "MyFeatureAgent"
    version = "1.0.0"

    def run(self, context: Any) -> MyFeatureAgentResult:
        """Execute the agent.

        Args:
            context: PipelineContext with config and upstream results.

        Returns:
            MyFeatureAgentResult with ...
        """
        cfg = context.config
        result = MyFeatureAgentResult()

        # 1. Check required upstream inputs
        if not context.gdelt_result:
            logger.info("MyFeatureAgent: no upstream data — returning empty result")
            result.warnings.append("No upstream data available")
            return result

        # 2. Do the work
        try:
            ...
        except Exception as exc:
            logger.error("MyFeatureAgent failed: %s", exc)
            result.errors.append(str(exc))
            result.status = "CRITICAL"

        return result
```

**Rules:**
- Inherit from `BaseAgent`
- Accept `context: Any` (typed as `PipelineContext` at runtime)
- Return only the typed result dataclass — never return a raw `dict`
- Log warnings for recoverable failures, errors for unrecoverable ones
- Downstream agents must tolerate `None` or empty results

---

## Step 4: Register in the Pipeline

In `geoeventfusion/pipeline.py`, add the agent to the execution order:

```python
from geoeventfusion.agents.my_feature_agent import MyFeatureAgent

def run_pipeline(config: PipelineConfig) -> PipelineContext:
    ...
    # Phase N: My Feature
    record = context.log_phase_start("MyFeatureAgent")
    agent = MyFeatureAgent(config=config)
    context.my_feature_result = agent.run(context)
    context.log_phase_end(record, status=context.my_feature_result.status)
    ...
```

---

## Step 5: Update AGENTS.md

Add a new section to `AGENTS.md` following the existing format:

```markdown
## 2.N MyFeatureAgent

**Module:** `geoeventfusion/agents/my_feature_agent.py`
**Purpose:** One-line description.

### Inputs
| Field | Type | Description |
|---|---|---|
| `context.gdelt_result` | `GDELTAgentResult` | ... |

### Outputs — `MyFeatureAgentResult`
...

### Failure Handling
...
```

---

## Step 6: Update skills.md

Add a new section to `skills.md` describing the new capability.

---

## Step 7: Update DIRECTORY_STRUCTURE.md

Add the new file to the canonical directory tree.

---

## Step 8: Write Unit Tests

Create `tests/unit/test_my_feature_agent.py` with at minimum:

1. **Happy-path test** — valid fixture data → expected output
2. **Empty-input test** — `None` or empty upstream result → graceful return
3. **Malformed-input test** — bad JSON or missing fields → no exception raised

```python
class TestMyFeatureAgentHappyPath:
    def test_run_returns_typed_result(self, test_pipeline_config, ...):
        ...

class TestMyFeatureAgentEmpty:
    def test_run_no_upstream_data_returns_result(self, test_pipeline_config):
        agent = MyFeatureAgent(config=test_pipeline_config)
        ctx = MagicMock()
        ctx.config = test_pipeline_config
        ctx.gdelt_result = None
        result = agent.run(ctx)
        assert isinstance(result, MyFeatureAgentResult)

class TestMyFeatureAgentMalformed:
    def test_run_bad_data_does_not_raise(self, test_pipeline_config):
        ...
```

---

## Step 9: Update requirements.txt

If your agent introduces new dependencies, add them to `requirements.txt`:

```
my-new-library>=1.0.0
```

---

## Checklist Summary

- [ ] Output model defined in `geoeventfusion/models/`
- [ ] Model exported from `geoeventfusion/models/__init__.py`
- [ ] Result field added to `PipelineContext`
- [ ] Agent implemented in `geoeventfusion/agents/<name>_agent.py`
- [ ] Agent registered in `geoeventfusion/pipeline.py`
- [ ] `AGENTS.md` updated with agent contract
- [ ] `skills.md` updated with new capabilities
- [ ] `DIRECTORY_STRUCTURE.md` updated
- [ ] Unit tests in `tests/unit/test_<name>_agent.py`
- [ ] `requirements.txt` updated if needed
