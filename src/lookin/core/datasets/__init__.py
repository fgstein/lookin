"""The datasets of a team: what is in the shared folders, where each CSV
came from, and what is in it.

There is no registry. The shared folders the user points Lookin at are the
reference -- they are read-only, maintained by hand or by scripts, and
everything Lookin learns about them (content hashes, statistics, the
MLflow run a file came from) lives in Lookin's own cache, never next to
the CSVs. A dataset is known by its file name and the hash of its content.

    config.py       which folders, which MLflow experiments hold datasets
    catalog.py      the CSVs of those folders, and which is derived from which
    fingerprint.py  the content hash, the same one the dataset scripts log
    profile.py      the statistics of one CSV, as plain numbers
    mlflow_link.py  the runs of the datasets experiments, and matching a file to one
    compare.py      two datasets side by side
    index.py        all of the above kept up to date, for the UI and the MCP tools
    agent.py        the MCP tools' answers
"""
