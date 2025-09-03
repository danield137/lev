# Migration Plan: EvalManifest Class System

## Objective
Migrate evaluation manifests from `kusto.json` (old flat schema) to `kusto.json.next` (new hierarchical, class-based schema) with no backward compatibility layer.

## Guiding Principles
- Zero backward-compat; delete old paths when each step is green
- Tight feedback loops: one-hour tasks, one PR each  
- Replicate current `dataset_loader.py` ergonomics: **single entry-point returns a list of EvalCase objects** usable by existing runner/tests

## Steps

### 1. Datamodel (scaffold) ≈ 1 h
1. Create `lev/manifest/eval_manifest.py` with dataclasses:
   - `LLMProfile` - individual LLM configuration
   - `LLMConfig` - collection of profiles with active selection. note that if no profiles are given, defaults to `profiles.json` file lookup in current directory (or, an optional, sepcific `profiles_file` field on the llm_config)
   - `MCPServer` - MCP server configuration
   - `MCPConfig` - collection of MCP servers
   - `ExecutionConfig` - execution parameters (solver, asker settings)
   - `Evaluator` - individual evaluator configuration
   - `EvalCase` - single evaluation item
   - `EvalManifest` - top-level manifest container

2. Provide sensible defaults so omitted fields deserialize cleanly
3. Unit test: `EvalManifest.from_file("kusto.json.next")` produces 3 EvalCase objects

### 2. Loader shim ≈ 1 h  
Replace `dataset_loader.py::load_dataset` body with:  
```python
return EvalManifest.from_file(path).data
```  
Keep signature identical (path→list[EvalCase]) so no callers change.

### 3. LLM profile wiring ≈ 1 h  
Update `llm_config_loader.py` & provider-factory code to take `EvalManifest.llm` rather than legacy `llm_config`.  
Migration mapping:  
- `manifest.llm.active_profile` ← `llm_config['active_profile']`
- `manifest.llm.profiles[name].model` ← `llm_config['profiles'][name]['model']`

### 4. MCP servers wiring ≈ 1 h  
Swap `runner.py` & `core/mcp.py` to ingest `manifest.mcp.servers`.  
Remove old `mcp_servers` reference.

### 5. Execution section ≈ 1 h  
Propagate `EvalCase.execution.*` into solver/asker factories. Map 1:1 to existing kwargs so behaviour unchanged.

### 6. Evaluators refactor ≈ 2 h  
1. Introduce `lev/scoring/evaluator_registry.py`
2. Convert hard-coded scoring logic to iterate over `EvalCase.evaluators`
3. Support two evaluator kinds: `function` (call local python) and `llm` (call provider)
4. Port existing "critique", "llm_judge" etc. as registry entries
5. Update affected tests (`tests/test_judge*.py`, `test_mcp_eval.py`)

### 7. Cleanup ≈ 30 min  
- Delete deprecated keys (`llm_config`, `mcp_servers`) & old loader helpers
- Update docs & sample JSON
- Run full test suite; commit

### 8. Stretch – CLI helper (optional)  
`lev.cli convert --in old.json --out new.json` for future datasets.

## Sequencing / PR Order
Datamodel → Loader shim → LLM → MCP → Execution → Evaluators → Cleanup

## Estimated Total ≈ 7 hours

All steps maintain current ergonomics: **`load_dataset(path)` still works**; everything new is internal.
