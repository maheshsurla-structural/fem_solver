"""SectionLibrary -- in-memory registry of catalogued / named sections.

The library is the single user-facing entry point for looking up
sections by name. It is populated by sub-phase II.4 with the AISC,
Eurocode, and Indian catalogues. For Theme II.2 we ship the registry
core and a small unit-test set of seed sections; the real catalogue
plumbing arrives in II.4.

Usage::

    from femsolver.sections import SectionLibrary

    lib = SectionLibrary.global_instance()
    w14x90 = lib.get("W14x90")
    ipe300 = lib.get("IPE300")
    ismb400 = lib.get("ISMB400")

    # Or by family
    for name in lib.list_family("W"):
        ...

    # Register a custom section (e.g. a user-defined catalog)
    lib.register(my_section)
"""
from __future__ import annotations

from femsolver.sections.section import Section


class SectionLibrary:
    """A registry of named :class:`Section` instances.

    Names must be unique. Use :meth:`global_instance` for the single
    process-wide registry (lazily populated by sub-phase II.4
    catalogue loaders).
    """

    _global: "SectionLibrary | None" = None

    def __init__(self) -> None:
        self._by_name: dict[str, Section] = {}
        self._by_family: dict[str, list[str]] = {}

    # ----------------------------------------------------------- access
    @classmethod
    def global_instance(cls) -> "SectionLibrary":
        if cls._global is None:
            cls._global = cls()
        return cls._global

    @classmethod
    def reset_global(cls) -> None:
        """Test helper: clear the global registry."""
        cls._global = None

    # ----------------------------------------------------------- catalogue
    @classmethod
    def aisc(cls, *, material=None) -> "SectionLibrary":
        """Return the AISC W-shape catalogue as a :class:`SectionLibrary`.

        Cached after first call. Pass ``material`` to attach a steel
        material reference to every section (only effective on first
        call); to attach per-project materials, use
        :func:`femsolver.sections.catalogue.aisc_section`.
        """
        from femsolver.sections.catalogue.aisc import load_aisc_library
        return load_aisc_library(material=material)

    @classmethod
    def eurocode(cls, *, material=None) -> "SectionLibrary":
        """Return the Eurocode (IPE + HEA + HEB) catalogue."""
        from femsolver.sections.catalogue.eurocode import (
            load_eurocode_library,
        )
        return load_eurocode_library(material=material)

    @classmethod
    def indian(cls, *, material=None) -> "SectionLibrary":
        """Return the Indian (ISMB + ISMC + ISA) catalogue."""
        from femsolver.sections.catalogue.indian import (
            load_indian_library,
        )
        return load_indian_library(material=material)

    # ----------------------------------------------------------- registration
    def register(self, section: Section, *, overwrite: bool = False) -> None:
        """Add a section to the registry.

        Parameters
        ----------
        section : Section
            Must have a non-empty ``name``.
        overwrite : bool, default False
            If False, raises on name collision. If True, replaces.
        """
        if not section.name:
            raise ValueError(
                "section must have a non-empty name to be registered"
            )
        if section.name in self._by_name and not overwrite:
            raise ValueError(
                f"section {section.name!r} is already registered; "
                f"pass overwrite=True to replace"
            )
        self._by_name[section.name] = section
        family = section.family or "_unknown"
        self._by_family.setdefault(family, [])
        if section.name not in self._by_family[family]:
            self._by_family[family].append(section.name)

    # ----------------------------------------------------------- lookup
    def get(self, name: str) -> Section:
        if name not in self._by_name:
            raise KeyError(f"no section named {name!r}; available: "
                           f"{sorted(self._by_name)[:10]} ...")
        return self._by_name[name]

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> Section:
        return self.get(name)

    def __len__(self) -> int:
        return len(self._by_name)

    # ----------------------------------------------------------- listing
    def list_all(self) -> list[str]:
        return sorted(self._by_name.keys())

    def list_family(self, family: str) -> list[str]:
        return list(self._by_family.get(family, []))

    def families(self) -> list[str]:
        return sorted(self._by_family.keys())
