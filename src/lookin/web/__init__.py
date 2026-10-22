"""The browser interface of Lookin, served as plain HTML/CSS/JS.

Reached with `lookin web`. The analysis itself lives in `lookin.core`, which
knows nothing of HTML: the same functions answer the MCP endpoint
(`lookin.mcp_server`) and the exported reports.

Everything the browser needs ships in `static/` (see CLAUDE.md: no CDN, no
font or script fetched at runtime). Pages are rendered on the server, one
session per browser tab (see session.py), and the small runtime in
static/app.js swaps the fragments an action re-renders.
"""
