# Contributing

Contributions that improve reproducibility, testing, documentation, or dataset-integrity checks are welcome.

Before opening a pull request:

1. keep training/evaluation behavior changes separate from documentation-only changes;
2. describe any change that can alter model outputs, data selection, checkpoint selection, or metrics;
3. run:

```bash
python -m compileall scripts
python -m unittest discover -s tests -v
```

Do not commit third-party datasets, patient information, local absolute paths, credentials, or large checkpoints to Git history.
