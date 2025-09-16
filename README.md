# Lev

A project for LLM evaluation and testing.

## Getting Started

1. Make sure you have a venv, and install this project
2. Make sure you have a .env file (copy the sample one to `.env` and edit).
3. Make sure you have the right authentication configuration.
4. `python eval.py [your_file]` (e.g. kusto_relational_easy.evl)w

## Output example (running kusto_relational_easy.evl)

<img width="3127" height="654" alt="image" src="https://github.com/user-attachments/assets/8c7fcdc8-993b-4888-a50b-9adeaacaeaaa" />

## Overview

The goal of this project is to be able to properly evaluate the added value of specific tools to an LLM (Large Language Model) agent.

It consists of several components, including:

- A framework for running agents with tools (MCP), following the ReAct pattern (LLM retrospection after tool calling). Effectively, an MCP host.
- A framework for judging the quality of LLM outputs
- A set of utilities to load configuration files and other plumbing tasks.


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
