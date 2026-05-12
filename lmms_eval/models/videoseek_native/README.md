# VideoSeek Native Backend

This package vendors the VideoSeek runtime used by `--model videoseek` into `lmms-eval`.

## Upstream Snapshot
- Source repo: `/Users/lazymieie/Desktop/ae/videoseek`
- Vendored commit: `443f8710bd1f9023f4a52ac95cb83610ad29c5ac`

## Layout
- `agent.py`: native VideoSeek agent execution loop
- `tools/`: `overview`, `skim`, `focus`, `answer`
- `core/`: `Action`, `Observation`, `Trajectory`
- `config/`: vendored `general.yaml` and `prompts.yaml`
- `utils.py`: LiteLLM calls, recorder hooks, subtitle parsing, text helpers

## Sync Process
1. Diff the upstream VideoSeek repo against this package.
2. Copy code changes into `lmms_eval/models/videoseek_native/`.
3. Preserve lmms-eval-specific changes:
   - relative imports
   - recorder hooks in `utils.py`
   - direct tool-call execution in `agent.py`
   - subprocess orchestration in `lmms_eval/models/simple/videoseek.py`
4. Update the vendored commit constant in `__init__.py`.
