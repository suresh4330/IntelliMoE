"""
utils/prompt_manager.py
-----------------------
Centralised prompt management for all IntelliMoE experts.

Responsibilities:
  - Resolve prompt files relative to the project root (prompts/*.txt).
  - Support prompt versioning via the naming convention:
        prompts/{name}.txt          ← default / unversioned prompt
        prompts/{name}_v{N}.txt     ← versioned prompt (N = positive integer)
  - Cache every loaded prompt in memory so disk I/O happens at most once
    per (name, version) pair — even across multiple expert instances.
  - Return prompt text by expert name and optional version.
  - Raise clear, descriptive exceptions on missing or unreadable files.

Versioning convention
─────────────────────
  File on disk              get_prompt() call
  ─────────────────────     ──────────────────────────────────────
  prompts/coding.txt        get_prompt("coding")
  prompts/coding_v1.txt     get_prompt("coding", version=1)
  prompts/coding_v2.txt     get_prompt("coding", version=2)
  (highest numbered)        get_prompt("coding", version="latest")

Backward compatibility
──────────────────────
  All existing calls to get_prompt(name) continue to work unchanged —
  they load the unversioned ``{name}.txt`` file exactly as before.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-root anchor
# ---------------------------------------------------------------------------
# __file__ → .../IntelliMoE/utils/prompt_manager.py
# .parent  → .../IntelliMoE/utils/
# .parent  → .../IntelliMoE/          ← project root
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_PROMPTS_DIR:  Path = _PROJECT_ROOT / "prompts"

# Regex that matches versioned prompt filenames: {name}_v{N}.txt
_VERSIONED_FILE_RE = re.compile(r"^(?P<name>.+)_v(?P<version>\d+)$")

# Sentinel constant used as the ``version`` argument for the latest version.
LATEST = "latest"


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class PromptNotFoundError(FileNotFoundError):
    """Raised when the requested prompt file does not exist."""


class PromptLoadError(IOError):
    """Raised when a prompt file exists but cannot be read."""


class PromptVersionError(LookupError):
    """Raised when the requested version does not exist for an expert."""


# ---------------------------------------------------------------------------
# PromptVersion — lightweight metadata for a single versioned prompt
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class PromptVersion:
    """
    Metadata for one versioned prompt file.

    Attributes
    ----------
    version : int
        The version number parsed from the filename (e.g. 2 for ``coding_v2.txt``).
    path : Path
        Absolute path to the ``.txt`` file on disk.
    """
    version: int
    path: Path = field(compare=False)

    def __str__(self) -> str:
        return f"v{self.version} ({self.path.name})"


# ---------------------------------------------------------------------------
# PromptManager
# ---------------------------------------------------------------------------

class PromptManager:
    """
    Load, version, and serve prompt templates from the ``prompts/`` directory.

    Versioned prompt files follow the naming convention::

        prompts/{name}_v{N}.txt     e.g. prompts/coding_v1.txt

    Unversioned files (``{name}.txt``) remain the default and are loaded
    when no ``version`` argument is passed — preserving backward compatibility.

    Parameters
    ----------
    prompts_dir : Path | str | None
        Override the default ``prompts/`` directory. Useful for testing.

    Examples
    --------
    >>> pm = PromptManager()

    >>> # Existing unversioned API (unchanged)
    >>> prompt = pm.get_prompt("coding")

    >>> # Versioned access
    >>> prompt_v1 = pm.get_prompt("coding", version=1)
    >>> prompt_v2 = pm.get_prompt("coding", version=2)
    >>> latest    = pm.get_prompt("coding", version="latest")

    >>> # Introspection
    >>> pm.list_versions("coding")        # [PromptVersion(1, ...), PromptVersion(2, ...)]
    >>> pm.list_available_prompts()       # ["coding", "deeplearning", "genai", ...]
    >>> pm.cache_info()                   # {"size": 3, "keys": [...]}
    """

    def __init__(self, prompts_dir: Optional[Union[Path, str]] = None) -> None:
        self._prompts_dir: Path = (
            Path(prompts_dir).resolve() if prompts_dir else _PROMPTS_DIR
        )

        # In-memory cache keyed by (expert_name, version_key).
        # version_key is either an int or the string "default".
        self._cache: dict[tuple[str, Union[int, str]], str] = {}

        logger.debug("PromptManager initialised. Directory: %s", self._prompts_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prompt(
        self,
        expert_name: str,
        version: Optional[Union[int, str]] = None,
    ) -> str:
        """
        Return the prompt text for the given expert and optional version.

        Parameters
        ----------
        expert_name : str
            Name of the expert (without ``.txt``).
            Examples: ``"coding"``, ``"math"``, ``"systemdesign"``.
        version : int | "latest" | None
            - ``None`` (default): load the unversioned ``{name}.txt`` file.
            - ``int``: load ``{name}_v{N}.txt`` for the given version number.
            - ``"latest"``: load the highest-numbered ``{name}_vN.txt`` file.
              Raises ``PromptVersionError`` if no versioned files exist.

        Returns
        -------
        str
            Full prompt text, stripped of leading/trailing whitespace.

        Raises
        ------
        ValueError
            If ``expert_name`` is empty or contains illegal path characters,
            or if ``version`` is an invalid type/value.
        PromptNotFoundError
            If the target ``.txt`` file does not exist on disk.
        PromptVersionError
            If ``version="latest"`` is requested but no versioned files exist.
        PromptLoadError
            If the file exists but cannot be read (permissions, encoding).
        """
        expert_name = self._validate_name(expert_name)
        cache_key, prompt_path = self._resolve(expert_name, version)

        # Serve from cache if already loaded.
        if cache_key in self._cache:
            logger.debug("Cache hit: %s@%s", expert_name, cache_key[1])
            return self._cache[cache_key]

        # Load from disk and populate cache.
        prompt_text = self._load_file(expert_name, prompt_path)
        self._cache[cache_key] = prompt_text

        logger.info(
            "Prompt loaded: expert='%s', version=%s, chars=%d",
            expert_name, cache_key[1], len(prompt_text),
        )
        return prompt_text

    def list_versions(self, expert_name: str) -> list[PromptVersion]:
        """
        Return all versioned prompt files for the given expert, sorted ascending.

        Only files matching ``{expert_name}_v{N}.txt`` are included.
        The unversioned ``{expert_name}.txt`` is not included.

        Parameters
        ----------
        expert_name : str
            Expert name to query (e.g. ``"coding"``).

        Returns
        -------
        list[PromptVersion]
            Sorted list of PromptVersion objects (lowest → highest).
            Returns an empty list if no versioned files exist.
        """
        expert_name = self._validate_name(expert_name)
        return self._discover_versions(expert_name)

    def list_available_prompts(self) -> list[str]:
        """
        Return the base names of all unique experts that have any prompt file.

        Both versioned (``coding_v1.txt``) and unversioned (``coding.txt``)
        files contribute the name ``"coding"`` to the result.

        Returns
        -------
        list[str]
            Sorted, deduplicated list of expert base names.
        """
        if not self._prompts_dir.is_dir():
            return []

        names: set[str] = set()
        for p in self._prompts_dir.glob("*.txt"):
            m = _VERSIONED_FILE_RE.match(p.stem)
            names.add(m.group("name") if m else p.stem)

        return sorted(names)

    def cache_info(self) -> dict:
        """
        Return a snapshot of the current in-memory cache.

        Returns
        -------
        dict
            ``{"size": int, "keys": list[str]}`` describing the cache state.
        """
        keys = [f"{name}@{ver}" for name, ver in self._cache]
        return {"size": len(self._cache), "keys": sorted(keys)}

    def clear_cache(self) -> None:
        """
        Evict all cached prompts, forcing fresh disk reads on next access.

        Useful during testing or when prompt files are hot-updated at runtime.
        """
        evicted = len(self._cache)
        self._cache.clear()
        logger.info("Prompt cache cleared (%d entries evicted).", evicted)

    def invalidate(self, expert_name: str, version: Optional[Union[int, str]] = None) -> None:
        """
        Evict a single cache entry for the given expert (and optionally version).

        Parameters
        ----------
        expert_name : str
            Expert to invalidate.
        version : int | "latest" | None
            Same semantics as ``get_prompt()``. If None, evicts the unversioned
            entry. Does not raise if the entry is not cached.
        """
        expert_name = self._validate_name(expert_name)
        try:
            cache_key, _ = self._resolve(expert_name, version)
        except (PromptVersionError, PromptNotFoundError):
            return  # Nothing to evict

        removed = self._cache.pop(cache_key, None)
        if removed is not None:
            logger.debug("Cache invalidated: %s@%s", expert_name, cache_key[1])

    # ------------------------------------------------------------------
    # Internal — resolution logic
    # ------------------------------------------------------------------

    def _resolve(
        self,
        expert_name: str,
        version: Optional[Union[int, str]],
    ) -> tuple[tuple[str, Union[int, str]], Path]:
        """
        Determine the (cache_key, file_path) pair for a given request.

        Returns
        -------
        tuple
            ``((expert_name, version_key), absolute_path)``
        """
        if version is None:
            # Unversioned — load {name}.txt
            cache_key = (expert_name, "default")
            path = self._prompts_dir / f"{expert_name}.txt"

        elif version == LATEST:
            # Latest versioned file
            versions = self._discover_versions(expert_name)
            if not versions:
                raise PromptVersionError(
                    f"No versioned prompt files found for expert '{expert_name}'. "
                    f"Expected files like '{expert_name}_v1.txt' in {self._prompts_dir}."
                )
            latest_pv = versions[-1]    # list is sorted ascending
            cache_key = (expert_name, latest_pv.version)
            path = latest_pv.path

        elif isinstance(version, int):
            if version < 1:
                raise ValueError(
                    f"version must be a positive integer, got {version}."
                )
            cache_key = (expert_name, version)
            path = self._prompts_dir / f"{expert_name}_v{version}.txt"

        else:
            raise ValueError(
                f"version must be None, a positive int, or the string 'latest'. "
                f"Got: {version!r}"
            )

        return cache_key, path

    # ------------------------------------------------------------------
    # Internal — version discovery
    # ------------------------------------------------------------------

    def _discover_versions(self, expert_name: str) -> list[PromptVersion]:
        """
        Scan the prompts directory for all versioned files for ``expert_name``.

        Returns a list sorted by version number (ascending).
        """
        if not self._prompts_dir.is_dir():
            return []

        versions: list[PromptVersion] = []
        for p in self._prompts_dir.glob(f"{expert_name}_v*.txt"):
            m = _VERSIONED_FILE_RE.match(p.stem)
            if m and m.group("name") == expert_name:
                versions.append(PromptVersion(version=int(m.group("version")), path=p))

        return sorted(versions)    # PromptVersion is ordered by version int

    # ------------------------------------------------------------------
    # Internal — input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_name(expert_name: str) -> str:
        """
        Sanitise and validate the expert name.

        Raises
        ------
        ValueError
            If the name is blank or contains path-traversal characters.
        """
        name = expert_name.strip()
        if not name:
            raise ValueError("expert_name must not be empty.")

        # Guard against path-traversal attacks (e.g. "../../etc/passwd").
        if any(c in name for c in ("/", "\\", "..")):
            raise ValueError(
                f"expert_name contains illegal characters: '{expert_name}'"
            )
        return name

    # ------------------------------------------------------------------
    # Internal — file I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _load_file(expert_name: str, path: Path) -> str:
        """
        Read and return the contents of a prompt file.

        Raises
        ------
        PromptNotFoundError
            If the file does not exist at ``path``.
        PromptLoadError
            If the file cannot be read for any other reason.
        """
        if not path.exists():
            raise PromptNotFoundError(
                f"Prompt file not found for expert '{expert_name}'. "
                f"Expected: {path}"
            )

        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise PromptLoadError(
                f"Could not read prompt for '{expert_name}' at '{path}': {exc}"
            ) from exc
