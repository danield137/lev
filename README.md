# Lev

A project for LLM evaluation and testing.

## Getting Started

1. Make sure you have a venv, and install this project
2. Make sure you have a .env file (copy the sample one to `.env` and edit)
3. `python eval.py [your_file]`

## Overview

The goal of this project is to be able to properly evaluate the added value of specific tools to an LLM (Large Language Model) agent.

It consists of several components, including:

- A framework for running agents with tools (MCP), following the ReAct pattern (LLM retrospection after tool calling). Effectively, an MCP host.
- A framework for judging the quality of LLM outputs
- A set of utilities to load configuration files and other plumbing tasks.


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
