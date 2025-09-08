# LEV Architecture Refactor Plan

## Objective

Refactor the LEV codebase into three clean, decoupled layers:

1. **MCP Host** - Standalone question-answering with agents and tools
2. **Evaluation** - External loop with asker agent and evaluations  
3. **Infrastructure** - Shared utilities, config, and external integrations

## Proposed Solution

```
lev/
├── common/              # Shared primitives used across layers
│   ├── __init__.py      # Export Agent, ChatHistory, event_hooks
│   ├── agent.py         # Base Agent class with optional tool support
│   ├── chat_history.py  # Message log with pretty-print helpers
│   └── event_hooks.py   # Pre/post-prompt callback system
│
├── host/                # Standalone MCP Host layer
│   ├── __init__.py      # Export configure_host, McpHost
│   ├── mcp_host.py      # Main McpHost implementation with prompt() method
│   ├── mcp_client.py    # Async wrapper around MCP stdio servers
│   └── host_errors.py   # Host-specific exception classes
│
├── eval/                # Evaluation framework
│   ├── __init__.py      # Export Evaluator and factory helpers
│   ├── evaluator.py     # Main evaluation loop and orchestration
│   ├── asker_agent.py   # Agent for continuing conversations with host
│   ├── runner.py        # CLI entry point for running evaluations
│   ├── scoring.py       # Scoring coordination and result aggregation
│   └── scorers/         # Individual scoring implementations
│       ├── __init__.py  # Scorer discovery and registration
│       └── llm_judge.py # LLM-based scoring implementation
│
└── infra/               # Infrastructure and utilities
    ├── config/          # Configuration loading and validation
    │   ├── __init__.py  # Export config loaders
    │   ├── manifest.py  # Evaluation manifest loading
    │   └── settings.py  # General settings and validation
    ├── io/              # Input/output utilities
    │   ├── __init__.py  # Export pretty printing and sinks
    │   ├── pretty_print.py # Colored console output
    │   └── result_sink.py  # Write evaluation results to files
    ├── logging/         # Logging configuration
    │   ├── __init__.py  # Export logging setup
    │   └── setup.py     # Structured, colorized logging
    └── providers/       # LLM provider integrations
        ├── __init__.py  # Provider factory and registry
        ├── registry.py  # Provider discovery and management
        ├── openai.py    # OpenAI provider implementation
        ├── azure.py     # Azure OpenAI provider implementation
        └── lmstudio.py  # LM Studio provider implementation
```

## Key Design Principles

1. **Dependency Isolation**: Host layer is standalone, depends only on common
2. **Hook-based Scoring**: Scorers use event hooks rather than being embedded in flow
3. **Clear Ownership**: Each layer has distinct responsibilities
4. **Logical Grouping**: Related functionality grouped together (not generic names)

## Critical Component APIs

### MCP Host API
```python
# Usage pattern:
mcp_host = configure_host(agent: Agent, tools: McpRegistry, config: Config)
reply = await mcp_host.prompt("What is 2+2?")
```

### Evaluation API  
```python
# Usage pattern:
evaluator = configure_evaluator(mcp_host, config)
for eval_spec in evals:
    result = await evaluator.evaluate(eval_spec, scorers, config)
```

### Event Hook System
```python
# Scorers register for callbacks:
from lev.common.event_hooks import register_post_prompt

@register_post_prompt
def my_scorer(host, reply, ctx):
    # Score the reply and store in ctx
    ctx['scores']['my_metric'] = calculate_score(reply)
```

## Implementation Steps

1. Create new package structure with logical groupings
2. Move shared components (Agent, ChatHistory) to common
3. Extract and clean MCP Host functionality into host layer
4. Refactor evaluation loop to use new host API
5. Split infrastructure into logical modules (config, io, logging, providers)
6. Implement event hook system for scorers
7. Update all imports and tests
8. Verify functionality and performance

## Benefits

- **Modularity**: Each layer can be used independently
- **Testability**: Clear boundaries make unit testing easier
- **Extensibility**: Event hook system allows easy scorer addition
- **Maintainability**: Reduced coupling and clearer responsibilities
- **Logical Organization**: Related code grouped together, not by generic types
- **Reusability**: Host layer can be used outside evaluation context

## Migration Strategy

Since we're not maintaining backward compatibility:
1. Implement new structure in parallel
2. Migrate functionality incrementally  
3. Update tests to use new APIs
4. Remove old code once migration complete
