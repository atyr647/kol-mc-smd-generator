"""Pre-indexed case-insensitive asset resolver for Knight Online asset directories.

Resolution priority (newest first — pass roots in this order):
  1. additions/1886-Client/       ← newest, most complete
  2. ko-assets-1298/              ← USKO v1298
  3. ko-assets-master/game/       ← oldest fallback

Usage:
    resolver = AssetResolver([
        ".../additions/1886-Client",
        ".../ko-assets-1298",
        ".../ko-assets-master/game",
    ])
    path = resolver.resolve("Object/some_building/mesh.n3pmesh")
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AssetResolver:
    """Pre-indexed lookup across multiple asset roots.

    Indexes all files under the given roots at construction time.
    Higher-priority roots are listed first — their files win on collision.

    Two lookup strategies:
      1. Full relative path (normalised, case-insensitive)
      2. Basename-only fallback (for refs that omit the directory)
    """

    def __init__(self, asset_roots: list):  # list[str | Path]
        # Maps: normalised_key → absolute Path
        self._by_relpath: dict[str, Path] = {}
        self._by_basename: dict[str, Path] = {}
        self._basename_candidates: dict[str, list[Path]] = {}
        self._missing_log: list[str] = []

        # Walk roots in reverse so highest-priority root overwrites lower ones
        for root in reversed(asset_roots):
            root = Path(root)
            if not root.exists():
                logger.warning("Asset root not found: %s", root)
                continue
            count = 0
            for dirpath, _, filenames in os.walk(root):
                for fname in filenames:
                    full = Path(dirpath) / fname
                    try:
                        rel = full.relative_to(root)
                    except ValueError:
                        continue
                    norm_rel = self._normalize(str(rel))
                    norm_base = self._normalize(fname)
                    self._by_relpath[norm_rel] = full
                    self._by_basename[norm_base] = full
                    self._basename_candidates.setdefault(norm_base, []).append(full)
                    count += 1
            logger.info("Indexed %d files from %s", count, root)

        logger.info(
            "AssetResolver ready: %d unique relative paths, %d unique basenames",
            len(self._by_relpath),
            len(self._by_basename),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def resolve(self, ref: str) -> Optional[Path]:
        """Resolve an asset reference to an absolute Path, or None if not found.

        Tries full relative path first, then basename-only fallback.
        Missing refs are logged to missing_refs for later review.
        """
        norm = self._normalize(ref)

        result = self._by_relpath.get(norm)
        if result is not None:
            return result

        base = self._normalize(Path(norm).name)
        result = self._by_basename.get(base)
        if result is not None:
            return result

        self._missing_log.append(ref)
        return None

    def resolve_with_meta(self, ref: str) -> tuple[Optional[Path], dict]:
        """Resolve an asset reference with basic ambiguity diagnostics.

        Returns (path, meta) where meta includes:
          strategy: relpath | basename | missing
          candidate_count: number of candidates for the chosen strategy
          ambiguous: True when basename lookup had multiple candidates
        """
        norm = self._normalize(ref)

        result = self._by_relpath.get(norm)
        if result is not None:
            return result, {
                "strategy": "relpath",
                "candidate_count": 1,
                "ambiguous": False,
            }

        base = self._normalize(Path(norm).name)
        result = self._by_basename.get(base)
        if result is not None:
            cands = self._basename_candidates.get(base, [result])
            return result, {
                "strategy": "basename",
                "candidate_count": len(cands),
                "ambiguous": len(cands) > 1,
            }

        self._missing_log.append(ref)
        return None, {
            "strategy": "missing",
            "candidate_count": 0,
            "ambiguous": False,
        }

    @property
    def missing_refs(self) -> list[str]:
        """All refs that failed to resolve, in order encountered."""
        return list(self._missing_log)

    def clear_missing_log(self) -> None:
        self._missing_log.clear()

    def total_files(self) -> int:
        return len(self._by_relpath)

    # ── Internal ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(s: str) -> str:
        return s.replace("\\", "/").lower().lstrip("/")
