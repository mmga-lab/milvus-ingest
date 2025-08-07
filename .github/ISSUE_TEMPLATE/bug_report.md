---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description
A clear and concise description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:
1. Command run: `milvus-ingest ...`
2. Schema used: (attach or paste schema file)
3. Expected behavior
4. Actual behavior

## Environment
- OS: [e.g. Ubuntu 22.04, macOS 13.0, Windows 11]
- Python version: [e.g. 3.11.5]
- Package version: [e.g. 0.1.0]
- Installation method: [pip, conda, docker, source]

## Schema File (if applicable)
```json
{
  "collection_name": "your_collection",
  "fields": [
    ...
  ]
}
```

## Command and Output
```bash
$ milvus-ingest generate --schema schema.json --total-rows 100
# Paste the full command and output here
```

## Error Message (if applicable)
```
Paste the full error message and stack trace here
```

## Additional Context
Add any other context about the problem here.

## Possible Solution
If you have any ideas about how to fix this issue, please describe them here.