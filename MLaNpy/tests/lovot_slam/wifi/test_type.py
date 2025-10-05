import time
import numpy as np
import pytest

from lovot_apis.lovot_minid.wifi.wifi_pb2 import AP, GetAvailableAPResponse

from lovot_slam.wifi.type import Covariance, Localizer, Ssid, StampedAccessPoints


@pytest.mark.parametrize('matrix_str,timestamp,expected', [
    ('1,2,3,4,5,6,7,8,9', 0.0,
     Covariance(0.0, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))),
    ('1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36',
     0.0,
     Covariance(0.0, np.array([[1, 2, 6], [7, 8, 12], [31, 32, 36]]))),
    ('1,2,3', 0.0, None),
])
def test_covariance(matrix_str, timestamp, expected):
    if expected:
        obtained = Covariance.from_csv_string(matrix_str, timestamp)
        assert obtained == expected
    else:
        with pytest.raises(ValueError):
            Covariance.from_csv_string(matrix_str, timestamp)


def test_localizer():
    assert Localizer.VISUAL == Localizer.value_of('visual')
    assert Localizer.DEPTH == Localizer.value_of('depth')
    with pytest.raises(ValueError):
        Localizer.value_of('unknown')


def test_ssid():
    assert Ssid.from_strings('00:11:22:33:44:55', 'test') == Ssid(
        bssid=bytes.fromhex('001122334455'), essid='test')

    with pytest.raises(ValueError):
        Ssid.from_strings('00:11:22:33:44:55:66', 'test')

    with pytest.raises(ValueError):
        Ssid.from_strings('0x:11:22:33:44:55', 'test')


def test_stamped_access_points():
    unix_stamp = time.time()
    monotonic_stamp = time.monotonic()

    last_seen = max(0, monotonic_stamp - 10)
    d_last_seen = last_seen - monotonic_stamp

    aps = [AP(ssid='test1', hw_address='00:11:22:33:44:55',
              strength=50, last_seen=int(monotonic_stamp+d_last_seen+0.5))]
    res = GetAvailableAPResponse()
    res.ap.extend(aps)
    stamped_access_points = StampedAccessPoints.from_response(res)

    assert Ssid(bytes.fromhex('001122334455'), 'test1') in stamped_access_points
    assert stamped_access_points[Ssid(bytes.fromhex('001122334455'), 'test1')] == aps[0]
    # NOTE: there may be up to 1 second difference, because last_seen is integer
    assert pytest.approx(stamped_access_points.stamp, abs=1.0) == unix_stamp + d_last_seen

    res.ap.append(AP(ssid='test1', hw_address='01:11:22:33:44:55',
                     strength=50, last_seen=int(monotonic_stamp-5+0.5)))
    stamped_access_points = StampedAccessPoints.from_response(res)
    assert pytest.approx(stamped_access_points.stamp, abs=1.0) == unix_stamp - 7.5


def test_stamped_access_points_equality():
    monotonic_stamp = time.monotonic()
    last_seen = max(0, monotonic_stamp - 10)
    d_last_seen = last_seen - monotonic_stamp

    aps = [AP(ssid='test1', hw_address='00:11:22:33:44:55',
              strength=50, last_seen=int(monotonic_stamp+d_last_seen+0.5))]
    res = GetAvailableAPResponse()
    res.ap.extend(aps)
    stamped_access_points = StampedAccessPoints.from_response(res)

    # check equality
    assert stamped_access_points == stamped_access_points

    # check equality with copy
    copied_with_new_stamp = StampedAccessPoints(dict(stamped_access_points), 12.0)
    copied_with_the_same_stamp = StampedAccessPoints(dict(stamped_access_points), 12.0)
    assert copied_with_new_stamp == copied_with_the_same_stamp

    # check inequality
    copied_with_another_stamp = StampedAccessPoints(dict(stamped_access_points), 13.0)
    assert copied_with_new_stamp != copied_with_another_stamp

    # check inequality with different type
    copied = StampedAccessPoints(dict(stamped_access_points), 12.0)
    copied_with_extra_ap = StampedAccessPoints(dict(stamped_access_points).update(
        {Ssid(bytes.fromhex('001122334455'), 'test1'): AP(
            ssid='test1', hw_address='00:11:22:33:44:55',
            strength=50, last_seen=int(monotonic_stamp+d_last_seen+0.5))}), 12.0)
    assert copied != copied_with_extra_ap


def test_stamped_access_points_monotonic_time_conversion(monkeypatch):
    unix_stamp = time.time()
    monotonic_stamp = time.monotonic()

    last_seen = max(0, monotonic_stamp - 10)
    d_last_seen = last_seen - monotonic_stamp

    aps = [AP(ssid='test1', hw_address='00:11:22:33:44:55',
              strength=50, last_seen=int(monotonic_stamp+d_last_seen+0.5))]
    res = GetAvailableAPResponse()
    res.ap.extend(aps)
    stamped_access_points = StampedAccessPoints.from_response(res)

    # unix time should be the same,
    # even when the difference between monotonic/unix time is changed
    monkeypatch.setattr(time, 'time', lambda: unix_stamp + 1.0)
    other_access_points = StampedAccessPoints.from_response(res)
    assert stamped_access_points.stamp == other_access_points.stamp
    assert stamped_access_points == other_access_points

    for i in range(StampedAccessPoints._LAST_SEEN_UNIX_DIFF_MAP_MAX_SIZE + 100):
        aps = [AP(ssid='test1', hw_address='00:11:22:33:44:55',
                  strength=50, last_seen=int(monotonic_stamp+d_last_seen+i+0.5))]
        res = GetAvailableAPResponse()
        res.ap.extend(aps)
        stamped_access_points = StampedAccessPoints.from_response(res)
    assert len(StampedAccessPoints._LAST_SEEN_UNIX_DIFF_MAP) == \
        StampedAccessPoints._LAST_SEEN_UNIX_DIFF_MAP_MAX_SIZE

@pytest.mark.parametrize('covariance, expected_variances, expected_angle', [
    (Covariance(0.0, np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])),
     np.sqrt(np.array((1.0, 2.0))), np.deg2rad(0.0)),
    (Covariance(0.0, np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]])),
     np.sqrt(np.array((1.0, 2.0))), np.deg2rad(90.0)),
    # Below is a case reported in sentry (slightly modified to fix the principal axis),
    # where eigen vector became conjugate
    (Covariance(0.0, np.array([[ 1.00000e+02, -1.42380e-16, 6.17089e-03],
                               [ 3.36084e-17, 0.90000e+02, 5.96368e-04],
                               [ 6.17089e-03, 5.96368e-04, 9.86960e+00]])),
     np.sqrt(np.array((90, 100))), np.deg2rad(0.0)),
])
def test_sigma_ellipse(covariance, expected_variances, expected_angle):
    var, angle = covariance.sigma_ellipse()
    np.testing.assert_almost_equal(var, expected_variances)
    np.testing.assert_almost_equal(angle, expected_angle)

