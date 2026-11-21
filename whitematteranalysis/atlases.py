# -*- coding: utf-8 -*-
"""Declarative multi-atlas registry and pre-flight consistency validation."""

from dataclasses import dataclass
import glob
import json
from pathlib import Path
import re

from ._hemisphere import load_location_policy


@dataclass(frozen=True)
class AtlasSpec:
    """Layout, expected tract contract, and policy metadata for one atlas family."""

    family: str
    reg_dirname: str
    fc_dirname: str
    hemispheric_tracts: tuple
    commissural_tracts: tuple
    apply_org_cpc_overrides: bool


@dataclass(frozen=True)
class ResolvedAtlas:
    """Validated paths and metadata for a concrete atlas checkout."""

    family: str
    base_dir: Path
    reg_dir: Path
    reg_atlas_path: Path
    fc_dir: Path
    cluster_paths: tuple
    cluster_names: tuple
    location_data: object
    hemispheric_tracts: tuple
    commissural_tracts: tuple
    apply_org_cpc_overrides: bool


_ORG_HEMISPHERIC_TRACTS = (
    "T_AF", "T_CB", "T_CPC", "T_MdLF", "T_PLIC", "T_SLF-I", "T_SLF-II", "T_SLF-III",
    "T_EC", "T_EmC", "T_ICP", "T_ILF", "T_IOFF", "T_UF",
    "T_Intra-CBLM-I&P", "T_Intra-CBLM-PaT", "T_CR-F", "T_CR-P", "T_CST",
    "T_SF", "T_SO", "T_SP", "T_TF", "T_TO", "T_TP",
    "T_Sup-F", "T_Sup-FP", "T_Sup-O", "T_Sup-OT", "T_Sup-P", "T_Sup-PO", "T_Sup-PT", "T_Sup-T",
)

_EAST_WEST_HEMISPHERIC_TRACTS = (
    "T_AF", "T_EmC", "T_ILF", "T_IOFF", "T_MdLF", "T_SLF-I", "T_SLF-II", "T_SLF-III",
    "T_UF", "T_ICP", "T_CBLM-I&P", "T_CBLM-PaT", "T_SCP", "T_CPC", "T_CST",
    "T_CR-F", "T_CR-P", "T_SF", "T_SO", "T_SP", "T_TF", "T_TO", "T_TT", "T_TP",
    "T_EC", "T_Sup-F", "T_Sup-FP", "T_Sup-O", "T_Sup-OT", "T_Sup-P", "T_Sup-PO",
    "T_Sup-PT", "T_Sup-T", "T_CB",
)

_NABA_HEMISPHERIC_TRACTS = (
    "T_AF", "T_CB_D", "T_CB_V", "T_EmC", "T_ILF", "T_IOFF", "T_MdLF",
    "T_SLF-I", "T_SLF-II", "T_SLF-III", "T_UF", "T_ICP",
    "T_Intra-CBLM-I&P", "T_Intra-CBLM-PaT", "T_SCP", "T_CPC", "T_CST",
    "T_CR-F", "T_CR-P", "T_SF", "T_SO", "T_SP", "T_TF", "T_TO", "T_TT", "T_TP",
    "T_EC", "T_Sup-F", "T_Sup-FP", "T_Sup-O", "T_Sup-OT", "T_Sup-P", "T_Sup-PO",
    "T_Sup-PT", "T_Sup-T",
)

_COMMON_COMMISSURAL_TRACTS = (
    "T_CC1", "T_CC2", "T_CC3", "T_CC4", "T_CC5", "T_CC6", "T_CC7", "T_MCP",
)

ATLAS_SPECS = {
    'org': AtlasSpec(
        family='org',
        reg_dirname='ORG-RegAtlas-100HCP',
        fc_dirname='ORG-800FC-100HCP',
        hemispheric_tracts=_ORG_HEMISPHERIC_TRACTS,
        commissural_tracts=_COMMON_COMMISSURAL_TRACTS,
        apply_org_cpc_overrides=True,
    ),
    'east-west': AtlasSpec(
        family='east-west',
        reg_dirname='East-West_WM_Atlas_Reg',
        fc_dirname='East-West_WM_Atlas_800FC',
        hemispheric_tracts=_EAST_WEST_HEMISPHERIC_TRACTS,
        commissural_tracts=_COMMON_COMMISSURAL_TRACTS,
        apply_org_cpc_overrides=False,
    ),
    'naba': AtlasSpec(
        family='naba',
        reg_dirname='NABA-RegAtlas',
        fc_dirname='NABA-800FC',
        hemispheric_tracts=_NABA_HEMISPHERIC_TRACTS,
        commissural_tracts=_COMMON_COMMISSURAL_TRACTS,
        apply_org_cpc_overrides=False,
    ),
}


def normalize_atlas_family(family):
    if family is None:
        return 'auto'
    key = str(family).strip().lower().replace('_', '-')
    if key == 'auto' or key in ATLAS_SPECS:
        return key
    raise ValueError(
        f"Unsupported atlas family {family!r}; expected one of auto, {', '.join(sorted(ATLAS_SPECS))}"
    )


def _read_atlas_json(base_dir):
    meta_path = base_dir / 'ATLAS.json'
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Failed to parse atlas metadata {meta_path}: {exc}") from exc


def _resolve_base_and_spec(root, family_key):
    if not root.is_dir():
        raise FileNotFoundError(f"Atlas root directory does not exist: {root}")

    if family_key != 'auto':
        root_meta = _read_atlas_json(root)
        if root_meta and root_meta.get('atlas_family'):
            meta_family = normalize_atlas_family(root_meta['atlas_family'])
            if meta_family != family_key:
                raise ValueError(
                    f"Atlas directory {root} declares atlas_family={meta_family!r}, "
                    f"expected {family_key!r}"
                )
        spec = ATLAS_SPECS[family_key]
        candidates = (root, root / family_key)
        for base in candidates:
            if (base / spec.reg_dirname).is_dir() and (base / spec.fc_dirname).is_dir():
                meta = _read_atlas_json(base)
                if meta and meta.get('atlas_family'):
                    meta_family = normalize_atlas_family(meta['atlas_family'])
                    if meta_family != family_key:
                        raise ValueError(
                            f"Atlas directory {base} declares atlas_family={meta_family!r}, "
                            f"expected {family_key!r}"
                        )
                return base, spec, meta
        raise FileNotFoundError(
            f"Atlas directory {root} does not contain {spec.reg_dirname} and {spec.fc_dirname}"
        )

    # Auto-detection: ATLAS.json at root takes precedence, then direct subdirectory signatures.
    meta = _read_atlas_json(root)
    if meta and meta.get('atlas_family'):
        detected = normalize_atlas_family(meta['atlas_family'])
        if detected != 'auto':
            spec = ATLAS_SPECS[detected]
            if (root / spec.reg_dirname).is_dir() and (root / spec.fc_dirname).is_dir():
                return root, spec, meta

    for spec in ATLAS_SPECS.values():
        if (root / spec.reg_dirname).is_dir() and (root / spec.fc_dirname).is_dir():
            return root, spec, _read_atlas_json(root)

    raise FileNotFoundError(
        f"Unable to identify atlas family under {root}; expected directories for "
        f"{', '.join(ATLAS_SPECS)}"
    )


def _validate_mrml_references(fc_dir, valid_names, hemispheric_tracts, commissural_tracts):
    if len(valid_names) >= 800:
        expected = [fc_dir / f"{t}.mrml" for t in hemispheric_tracts + commissural_tracts]
        missing = [str(p) for p in expected if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing atlas MRML files: {missing[:5]}")

    mrml_paths = sorted(fc_dir.glob('T_*.mrml'))
    if not mrml_paths:
        raise FileNotFoundError(f"No T_*.mrml tract definitions found in {fc_dir}")

    for mrml_path in mrml_paths:
        text = mrml_path.read_text(encoding='utf-8', errors='replace')
        for match in re.findall(r'cluster_\d+\.vtp', text):
            if match not in valid_names:
                raise RuntimeError(f"MRML {mrml_path} references missing cluster {match}")


def resolve_atlas(atlas_root, family='auto', validate_mrml=True):
    """Resolve and validate an ORG, East-West, or NABA tractography atlas directory."""
    root = Path(atlas_root).resolve()
    family_key = normalize_atlas_family(family)
    base_dir, spec, meta = _resolve_base_and_spec(root, family_key)

    reg_dir = base_dir / spec.reg_dirname
    fc_dir = base_dir / spec.fc_dirname

    reg_atlas_path = reg_dir / 'registration_atlas.vtk'
    if not reg_atlas_path.is_file():
        reg_atlas_path = reg_dir / 'registration_atlas.vtp'
    if not reg_atlas_path.is_file():
        raise FileNotFoundError(f"Registration atlas file not found in {reg_dir}")

    for required_name in ('atlas.p', 'atlas.vtp'):
        req_path = fc_dir / required_name
        if not req_path.is_file():
            raise FileNotFoundError(f"Missing required spectral atlas file: {req_path}")

    cluster_paths = tuple(sorted(glob.glob(str(fc_dir / 'cluster_*.vtp'))))
    if not cluster_paths:
        raise FileNotFoundError(f"No cluster_*.vtp files found in {fc_dir}")
    cluster_names = tuple(Path(p).name for p in cluster_paths)

    location_file = fc_dir / 'cluster_hemisphere_location.txt'
    if not location_file.is_file():
        raise FileNotFoundError(f"Missing hemisphere location file: {location_file}")
    location_data = load_location_policy(
        str(location_file), apply_org_cpc_overrides=spec.apply_org_cpc_overrides)
    if len(location_data) != len(cluster_names):
        raise RuntimeError(
            f"Cluster count ({len(cluster_names)}) does not match {location_file} ({len(location_data)})"
        )

    hemispheric = spec.hemispheric_tracts
    commissural = spec.commissural_tracts
    if meta and isinstance(meta.get('output_contract'), dict):
        groups = meta['output_contract'].get('groups')
        if isinstance(groups, dict):
            if groups.get('hemispheric'):
                hemispheric = tuple(groups['hemispheric'])
            if groups.get('commissural'):
                commissural = tuple(groups['commissural'])

    if validate_mrml:
        _validate_mrml_references(fc_dir, set(cluster_names), hemispheric, commissural)

    return ResolvedAtlas(
        family=spec.family,
        base_dir=base_dir,
        reg_dir=reg_dir,
        reg_atlas_path=reg_atlas_path,
        fc_dir=fc_dir,
        cluster_paths=cluster_paths,
        cluster_names=cluster_names,
        location_data=location_data,
        hemispheric_tracts=hemispheric,
        commissural_tracts=commissural,
        apply_org_cpc_overrides=spec.apply_org_cpc_overrides,
    )
