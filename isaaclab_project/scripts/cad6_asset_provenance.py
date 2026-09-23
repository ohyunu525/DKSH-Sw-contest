"""Record the exact provisional CAD6 files used by a training run."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / 'isaaclab_project/assets/spiderbot_variants'
FILES = {
    'usd': ASSET_DIR / 'spiderbot_6leg/spiderbot_6leg.usd',
    'visuals': ASSET_DIR / 'spiderbot_6leg/visuals.usdc',
    'urdf': ASSET_DIR / 'spiderbot_6leg/spiderbot_6leg.urdf',
    'manifest': ASSET_DIR / 'assembly_manifest.json',
}


def sha256(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'CAD6 asset file missing or empty: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cad6_asset_hashes():
    """Check source evidence and return hashes of every training asset file.

    This verifies identity and presence, not physical CAD calibration.
    """
    manifest = json.loads(FILES['manifest'].read_text(encoding='utf-8'))
    source = ROOT / manifest['source'].replace('\\', '/')
    source_hash = sha256(source)
    if source_hash != manifest['source_sha256']:
        raise ValueError(f'CAD6 source hash differs from assembly manifest: {source}')
    if manifest.get('mount_radius_m') != 0.12:
        raise ValueError('CAD6 hip mount radius differs from the gait model')
    return {name: sha256(path) for name, path in FILES.items()}


def validate_cad6_checkpoint_assets(metadata):
    expected = metadata.get('asset_sha256')
    if not isinstance(expected, dict) or expected != cad6_asset_hashes():
        raise ValueError('CAD6 v3 checkpoint assets differ from the training files')
