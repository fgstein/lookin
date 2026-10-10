"""Makes the bundled Flet web client work with zero network access.

Two separate things fetch from the internet by default and both need
fixing for `lookin ui` to run fully offline:

1. Flet checks that the installed `flet-web` package version matches
   `flet`'s version, and if not (or if it's simply missing), shells out to
   `pip install` to fetch it. Pinning `flet` and `flet-web` to the exact
   same version in pyproject.toml means that check always passes, so this
   one is handled entirely by the dependency pins -- nothing to do here.

2. The served page loads CanvasKit (Skia/WASM) from Google's CDN
   (gstatic.com/flutter-canvaskit) at page-load time. This is hardcoded in
   TWO places, not one:
     a) flutter_bootstrap.js's own loader, which decides which CanvasKit
        build to fetch during initial page load;
     b) a *second*, independent default baked into the compiled
        main.dart.js itself, read from the engine-init config at runtime
        (this is the one that bit us the first time around -- patching
        only (a) still leaves main.dart.js falling back to the CDN for
        this).
   This module patches the installed `flet_web` package's static files
   once (idempotently) to serve a locally bundled copy of CanvasKit for
   both.

3. Separately, main.dart.js unconditionally fetches the app's base UI font
   (Roboto Regular, Latin subset -- "roboto/v32/....woff2") from
   fonts.gstatic.com on every load; it is NOT an optional fallback, it's
   the font essentially all on-screen text is rendered with. It's fetched
   relative to `fontFallbackBaseUrl`, the same config key CanvasKit's
   *actual* (optional) fallback mechanism uses for missing glyphs (CJK,
   Arabic, etc. -- hundreds of Noto Sans files we don't need and are happy
   to have simply fail). Blanking that URL to kill the CDN reach (an
   earlier version of this patch did exactly that) also breaks the Roboto
   fetch, and CanvasKit silently renders no text at all when it can't load
   its base font -- every field in the UI *looks* empty even though it
   isn't. So instead of blanking the URL, we point it at a local `fonts/`
   directory and bundle the one Roboto file inside it; optional
   fallback-script fetches still 404 there instead of reaching Google,
   which is fine since we don't ship the other ~700 script fonts.

If patching fails or a pattern doesn't match (e.g. an incompatible future
flet_web release), we print a clear warning rather than silently doing
nothing -- an unpatched, network-reaching build should never look
identical to a successfully patched one.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

# Both patterns below double as "reverse the patch" regexes: they match
# *either* the untouched original or an already-patched result (from this
# version or any earlier one, since we control every value ever injected
# here), and normalize back to the pristine form. Patching therefore always
# starts from a known-clean base, which makes it idempotent *and*
# self-healing if a previous lookin version applied a since-fixed patch --
# without this, an already-patched file with stale/broken values would look
# "done" forever and never pick up a fix shipped in a later lookin release.
_PRISTINE_A = "config: {\n        renderer: webRenderer\n    }"
_RE_A = re.compile(
    r'config:\s*\{\s*renderer:\s*webRenderer'
    r'(?:,\s*canvasKitBaseUrl:\s*"[^"]*"\s*/\*\s*lookinOfflinePatch\s*\*/)?'
    r'\s*\}'
)

_PRISTINE_B = "engineInitializer.initializeEngine({useColorEmoji: useColorEmoji})"
_RE_B = re.compile(
    r'engineInitializer\.initializeEngine\(\{\s*useColorEmoji:\s*useColorEmoji'
    r'(?:,\s*canvasKitBaseUrl:\s*"[^"]*"\s*,\s*fontFallbackBaseUrl:\s*"[^"]*")?'
    r'\s*\}\)'
)


def _bundled_canvaskit_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "canvaskit"


def _bundled_fonts_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "fonts"


def _flet_web_dir() -> Path | None:
    try:
        import flet_web
    except ImportError:
        return None
    return Path(flet_web.__file__).resolve().parent / "web"


def ensure_offline_assets() -> None:
    web_dir = _flet_web_dir()
    if web_dir is None or not web_dir.is_dir():
        print("lookin: flet_web package not found -- skipping offline-asset patch.")
        return

    try:
        _ensure_local_canvaskit_files(web_dir)
        _ensure_local_font_files(web_dir)
        _patch_bootstrap(web_dir)
    except OSError as ex:
        print(f"lookin: could not patch bundled Flet assets for offline use ({ex}); "
              f"the app will still work if there's internet access.")


def _ensure_local_canvaskit_files(web_dir: Path) -> None:
    src = _bundled_canvaskit_dir()
    dst = web_dir / "canvaskit"
    marker = dst / "canvaskit.wasm"
    if marker.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _ensure_local_font_files(web_dir: Path) -> None:
    src = _bundled_fonts_dir()
    dst = web_dir / "fonts"
    marker = dst / "roboto" / "v32" / "KFOmCnqEu92Fr1Me4GZLCzYlKw.woff2"
    if marker.exists():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _patch_bootstrap(web_dir: Path) -> None:
    bootstrap = web_dir / "flutter_bootstrap.js"
    if not bootstrap.is_file():
        print("lookin: flutter_bootstrap.js not found -- skipping offline-asset patch.")
        return
    text = bootstrap.read_text(encoding="utf-8")

    # Undo any prior patch (this version's or an older one's) before
    # reapplying -- see the _RE_A / _RE_B comment above for why.
    pristine = _RE_A.sub(_PRISTINE_A, text)
    pristine = _RE_B.sub(_PRISTINE_B, pristine)

    # (a) the JS loader's own asset-fetch decision.
    new_a = (
        "config: {\n"
        "        renderer: webRenderer,\n"
        "        canvasKitBaseUrl: \"canvaskit/\" /* lookinOfflinePatch */\n"
        "    }"
    )

    # (b) main.dart.js's own independent default, read from the engine-init
    # config at runtime -- this is the one a first pass at this patch missed.
    # fontFallbackBaseUrl also has to point somewhere that actually serves
    # the base Roboto font (see module docstring point 3) -- blanking it
    # entirely, as an earlier version of this patch did, left the UI
    # rendering no text at all.
    new_b = (
        "engineInitializer.initializeEngine({"
        "useColorEmoji: useColorEmoji, "
        "canvasKitBaseUrl: \"canvaskit/\", "
        "fontFallbackBaseUrl: \"fonts/\""
        "})"
    )

    missing = [name for name, needle in (("(a) loader config", _PRISTINE_A), ("(b) engine-init config", _PRISTINE_B)) if needle not in pristine]
    if missing:
        print(f"lookin: offline-asset patch could not find expected text for {missing} in "
              f"flutter_bootstrap.js -- this flet-web build may differ from the one this was "
              f"written against. The app will still work if there's internet access, but "
              f"CanvasKit may be fetched from Google's CDN.")
        return

    patched = pristine.replace(_PRISTINE_A, new_a).replace(_PRISTINE_B, new_b)

    if patched == text:
        return  # already up to date

    # Package managers (uv in particular) commonly hardlink cached wheel
    # contents into site-packages to save space/time. Writing over the file
    # in place (open+truncate) would mutate that shared inode and silently
    # corrupt the *cache* too, not just this install -- so instead we remove
    # the old directory entry and create a brand new file, which never
    # touches whatever else the original inode was linked from.
    tmp = bootstrap.with_suffix(".js.lookin-tmp")
    tmp.write_text(patched, encoding="utf-8")
    bootstrap.unlink()
    tmp.rename(bootstrap)
    print("lookin: patched Flet web assets to load CanvasKit locally (offline-safe).")
