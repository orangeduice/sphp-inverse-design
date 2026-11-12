import numpy as np

from sphp_inverse.data import fit_geo_scaler, load_dataset, reference_split

# Values recorded in data/DATA.md from the canonical dataset and the archived Keras scaler .pkl files.
REF_MEAN = [0.746234, 1.25644, 1.262633, 1.15154, 1.146871, 0.403273]
REF_SCALE = [0.144361, 0.429619, 0.432755, 0.492253, 0.493266, 0.222309]


def test_dataset_shape(data_path):
    ds = load_dataset(data_path)
    assert ds.geo.shape == (9800, 6)
    assert ds.R.shape == (9800, 161)
    assert np.isclose(ds.freq[0], 25.0) and np.isclose(ds.freq[-1], 29.0)


def test_reference_split_sizes():
    s = reference_split(9800)
    # Keras: split_at = floor(9310 * (1 - 0.3)) = 6517 (checked against Keras 2.15 in parity/export_keras_reference.py)
    assert (len(s.train), len(s.val), len(s.test)) == (6517, 2793, 490)
    all_idx = np.concatenate([s.train, s.val, s.test])
    assert len(np.unique(all_idx)) == 9800


def test_reference_split_matches_array_split(data_path):
    """Splitting indices must give the same rows as the reference split of the arrays."""
    from sklearn.model_selection import train_test_split

    ds = load_dataset(data_path)
    x_tr, x_te, _, _ = train_test_split(ds.geo, ds.R, test_size=0.05, random_state=42)
    s = reference_split(len(ds.geo))
    np.testing.assert_array_equal(ds.geo[s.test], x_te)
    np.testing.assert_array_equal(ds.geo[np.concatenate([s.train, s.val])], x_tr)


def test_scaler_matches_reference(data_path):
    sc = fit_geo_scaler(load_dataset(data_path).geo)
    np.testing.assert_allclose(sc.mean_, REF_MEAN, atol=1e-5)
    np.testing.assert_allclose(sc.scale_, REF_SCALE, atol=1e-5)
