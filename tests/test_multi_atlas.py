"""Unit and integration tests for declarative multi-atlas resolution and validation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import vtk
import whitematteranalysis as wma

from test_org_pipeline_cli import (
    ORG_CLI_PATH,
    _bundle_coords,
    _load_module,
    _make_tract_polydata,
)


def _populate_family_atlas(target_root, family, write_atlas_json=True):
    spec = wma.atlases.ATLAS_SPECS[family]
    target_root = Path(target_root)
    reg_dir = target_root / spec.reg_dirname
    fc_dir = target_root / spec.fc_dirname
    reg_dir.mkdir(parents=True, exist_ok=True)
    fc_dir.mkdir(parents=True, exist_ok=True)

    left_fibers = _bundle_coords(-35.0, 0.0, 40, n_pts=21, step=3.0, seed=1, jitter_scale=1.0)
    right_fibers = _bundle_coords(35.0, 0.0, 40, n_pts=21, step=3.0, seed=2, jitter_scale=1.0)
    comm_fibers = _bundle_coords(0.0, 35.0, 40, n_pts=21, step=3.0, commissural=True, seed=3, jitter_scale=1.0)
    all_fibers = left_fibers + right_fibers + comm_fibers
    atlas_pd = _make_tract_polydata(all_fibers)
    wma.io.write_polydata(atlas_pd, str(reg_dir / 'registration_atlas.vtk'))

    nystrom_mask = np.ones(len(all_fibers), dtype=bool)
    nystrom_mask[115:] = False
    np.random.seed(42)
    out_pd, cluster_idx, _, _, _, atlas, _ = wma.cluster.spectral(
        atlas_pd,
        number_of_clusters=3,
        number_of_eigenvectors=3,
        sigma=25.0,
        threshold=0.0,
        number_of_jobs=1,
        use_nystrom=True,
        nystrom_mask=nystrom_mask,
        distance_method='MeanSquared',
        outlier_std_threshold=np.inf,
        bilateral=False,
        use_gpu=False,
    )
    atlas.save(str(fc_dir), 'atlas')

    cluster_pds = wma.cluster.mask_all_clusters(
        out_pd, cluster_idx, 3, preserve_point_data=True, preserve_cell_data=True, verbose=False)
    hemi_rows = ['Cluster Index\tLocation Label\n']
    hemispheric_clusters = []
    commissural_clusters = []
    for c_idx, c_pd in enumerate(cluster_pds):
        fname = f'cluster_{c_idx + 1:05d}.vtp'
        wma.io.write_polydata(c_pd, str(fc_dir / fname))
        fa = wma.fibers.FiberArray()
        fa.hemispheres = True
        fa.hemisphere_percent_threshold = 0.6
        fa.convert_from_polydata(c_pd, points_per_fiber=15)
        if fa.number_commissure > fa.number_left_hem + fa.number_right_hem:
            hemi_rows.append(f'cluster_{c_idx + 1:05d}\tc\n')
            commissural_clusters.append(fname)
        else:
            hemi_rows.append(f'cluster_{c_idx + 1:05d}\th\n')
            hemispheric_clusters.append(fname)

    (fc_dir / 'cluster_hemisphere_location.txt').write_text(''.join(hemi_rows))
    (fc_dir / f'{spec.hemispheric_tracts[0]}.mrml').write_text(
        ''.join(f'<FiberBundleStorage fileName="{name}"/>\n' for name in hemispheric_clusters))
    (fc_dir / f'{spec.commissural_tracts[0]}.mrml').write_text(
        ''.join(f'<FiberBundleStorage fileName="{name}"/>\n' for name in commissural_clusters))

    if write_atlas_json and family in ('east-west', 'naba'):
        (target_root / 'ATLAS.json').write_text(json.dumps({
            'atlas_family': family,
            'output_contract': {
                'atlas_family': family,
                'groups': {
                    'hemispheric': list(spec.hemispheric_tracts),
                    'commissural': list(spec.commissural_tracts),
                },
            },
        }, indent=2))
    return target_root


class MultiAtlasTests(unittest.TestCase):
    def test_load_location_policy_cpc_overrides_gating(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cluster_hemisphere_location.txt'
            rows = ['Cluster Index\tLocation Label\n'] + [
                f'cluster_{i + 1:05d}\tc\n' for i in range(800)
            ]
            path.write_text(''.join(rows))

            org_policy = wma._hemisphere.load_location_policy(str(path), apply_org_cpc_overrides=True)
            for idx in (145, 159, 557, 677, 770):
                self.assertEqual(org_policy[idx][1], b'h')

            other_policy = wma._hemisphere.load_location_policy(str(path), apply_org_cpc_overrides=False)
            for idx in (145, 159, 557, 677, 770):
                self.assertEqual(other_policy[idx][1], b'c')

    def test_resolve_atlas_auto_explicit_and_parent_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / 'bundles_wma'
            _populate_family_atlas(parent, 'org', write_atlas_json=False)
            _populate_family_atlas(parent / 'east-west', 'east-west', write_atlas_json=True)
            _populate_family_atlas(parent / 'naba', 'naba', write_atlas_json=True)

            # Parent directory without explicit family selects org at top level
            res_org = wma.atlases.resolve_atlas(parent, family='auto')
            self.assertEqual(res_org.family, 'org')
            self.assertTrue(res_org.apply_org_cpc_overrides)
            self.assertEqual(res_org.reg_dir.name, 'ORG-RegAtlas-100HCP')

            # Parent directory + explicit family resolves child subdirectories
            res_ew_from_parent = wma.atlases.resolve_atlas(parent, family='east-west')
            self.assertEqual(res_ew_from_parent.family, 'east-west')
            self.assertFalse(res_ew_from_parent.apply_org_cpc_overrides)
            self.assertEqual(res_ew_from_parent.base_dir, (parent / 'east-west').resolve())

            res_naba_direct = wma.atlases.resolve_atlas(parent / 'naba', family='auto')
            self.assertEqual(res_naba_direct.family, 'naba')
            self.assertIn('T_CB_D', res_naba_direct.hemispheric_tracts)
            self.assertIn('T_CB_V', res_naba_direct.hemispheric_tracts)
            self.assertEqual(len(res_naba_direct.hemispheric_tracts) * 2 + len(res_naba_direct.commissural_tracts), 78)

            # Auto-detection works even without ATLAS.json via directory names
            (parent / 'east-west' / 'ATLAS.json').unlink()
            res_ew_nojson = wma.atlases.resolve_atlas(parent / 'east-west', family='auto')
            self.assertEqual(res_ew_nojson.family, 'east-west')

    def test_preflight_validation_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'ew'
            _populate_family_atlas(root, 'east-west', write_atlas_json=True)

            # Conflicting family in ATLAS.json
            with self.assertRaisesRegex(ValueError, 'declares atlas_family'):
                wma.atlases.resolve_atlas(root, family='naba')

            # Dangling cluster reference in MRML
            bad_mrml = root / 'East-West_WM_Atlas_800FC' / 'T_AF.mrml'
            orig_mrml = bad_mrml.read_text()
            bad_mrml.write_text(orig_mrml + '<FiberBundleStorage fileName="cluster_00999.vtp"/>\n')
            with self.assertRaisesRegex(RuntimeError, 'missing cluster cluster_00999.vtp'):
                wma.atlases.resolve_atlas(root, family='east-west')
            bad_mrml.write_text(orig_mrml)

            # Mismatched cluster count vs cluster_hemisphere_location.txt
            loc_path = root / 'East-West_WM_Atlas_800FC' / 'cluster_hemisphere_location.txt'
            loc_path.write_text('Cluster Index\tLocation Label\ncluster_00001\th\n')
            with self.assertRaisesRegex(RuntimeError, 'does not match'):
                wma.atlases.resolve_atlas(root, family='east-west')

            # Missing atlas.vtp
            (root / 'East-West_WM_Atlas_800FC' / 'atlas.vtp').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'atlas.vtp'):
                wma.atlases.resolve_atlas(root, family='east-west', validate_mrml=False)

    def test_apply_org_cli_supports_east_west_and_naba(self):
        org_cli = _load_module('wm_apply_ORG_atlas_multi', ORG_CLI_PATH)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / 'bundles'
            _populate_family_atlas(parent / 'east-west', 'east-west', write_atlas_json=True)
            _populate_family_atlas(parent / 'naba', 'naba', write_atlas_json=True)

            fibers = (
                _bundle_coords(-35.0, 0.0, 5, n_pts=21, step=3.0, seed=10)
                + _bundle_coords(35.0, 0.0, 5, n_pts=21, step=3.0, seed=11)
                + _bundle_coords(0.0, 35.0, 5, n_pts=21, step=3.0, commissural=True, seed=12)
            )
            subj_path = root / 'sub-multi.vtp'
            wma.io.write_polydata(_make_tract_polydata(fibers), str(subj_path))

            for fam, atlas_arg in (('east-west', parent), ('naba', parent / 'naba')):
                out_dir = root / f'out_{fam}'
                family_arg = fam if atlas_arg == parent else 'auto'
                with patch.object(sys, 'argv', [
                    str(ORG_CLI_PATH), '-i', str(subj_path), '-o', str(out_dir),
                    '-a', str(atlas_arg), '--atlas-family', family_arg,
                    '-r', 'rig', '-j', '1', '-gpu', 'off']):
                    org_cli.main()
                qc = json.loads((out_dir / 'sub-multi' / 'qc_summary.json').read_text())
                self.assertEqual(qc['atlas_family'], fam)
                self.assertEqual(qc['hardening_backend'], 'pure_python_affine')
                self.assertTrue((out_dir / 'sub-multi' / 'AnatomicalTracts' / 'anatomical_tracts_complete.log').is_file())


if __name__ == '__main__':
    unittest.main()
