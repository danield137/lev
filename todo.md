1. ~~list_tables on sample database is very very long.~~ 
2. ~~json format is verbose and redundant - use columnar (test it works properly)~~
3. limit inner monologue to 0 - mimics "q&a"
4. results to csv
5. add a switch to disable judge mode
6. compact mode - only send tool calls to judge, without actual responses
7. propagate errors from mcp servers running locally
8. bug when multiple tool calls are made, the output repeats
9. handle mcp instructions (should be sent along with available tools)



graph issues:
---
1. By default, won't resort to graph semantics (graph-match) 
2. No id() function confuses the model.
3. No clear way to describe a graph (get schema / topology) or sample it
4. Even if forced to (say, by having a `kusto_graph_query` tool specifically), often produces wrong queries.


# context management policy
Introduce a policy to manage context usage, limit size and decide what to do when breaching.
Preferred is to just omit payload, add tool call signature instead.
(remove llm compressing, as it is eating up the same tokens I'm trying to limit)
```
    "context_usage_policy": {
      "max_size": 10000,
      "reaction": "omit_parts"
    }
```