"""Check the source evidence and fail closed on checkpoint asset changes."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cad6_asset_provenance import cad6_asset_hashes, validate_cad6_checkpoint_assets


class Cad6AssetProvenanceTests(unittest.TestCase):
    def test_manifest_source_and_training_files_are_present(self):
        hashes = cad6_asset_hashes()
        self.assertEqual(set(hashes), {'usd', 'visuals', 'urdf', 'manifest'})
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))

    def test_checkpoint_rejects_changed_asset_hash(self):
        hashes = cad6_asset_hashes()
        validate_cad6_checkpoint_assets({'asset_sha256': hashes})
        changed = {**hashes, 'usd': '0' * 64}
        with self.assertRaisesRegex(ValueError, 'differ'):
            validate_cad6_checkpoint_assets({'asset_sha256': changed})


if __name__ == '__main__':
    unittest.main()
