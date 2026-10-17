"""The browser interface of Lookin, served as plain HTML/CSS/JS.

A second front end next to the Flet one in `lookin.ui`: same core, same
library of workspaces on disk, same features -- reached with `lookin web`
instead of `lookin ui`, on a port of its own, so both can run at once.

Everything the browser needs ships in `static/` (see CLAUDE.md: no CDN, no
font or script fetched at runtime). Pages are rendered on the server, one
session per browser tab (see session.py), and the small runtime in
static/app.js swaps the fragments an action re-renders.
"""
