import json

import numpy as np
import pytest

from lovot_slam.utils.unwelcomed_area import Polygon, _fit_rectangle_to_vertices


@pytest.mark.parametrize('shape_str,expected_vertices', [
    # Test clockwise but rolled input
    (
        '{"shape": "polygon", "vertices": [[0.1, 2.3], [4.5, 6.7], [8.9, 9.0], [1.2, 2.3]]}',
        [[4.5, 6.7], [8.9, 9.0], [1.2, 2.3], [0.1, 2.3]]
    ),
    # Test counter-clockwise input
    (
        '{"shape": "polygon", "vertices": [[0.1, 2.3], [1.2, 2.3], [8.9, 9.0], [4.5, 6.7]]}',
        [[4.5, 6.7], [8.9, 9.0], [1.2, 2.3], [0.1, 2.3]]
    ),
])
def test_shape_serialization(shape_str, expected_vertices):
    j = json.loads(shape_str)
    if j.get('shape') == 'polygon':
        deserialized = Polygon.json_deserialize(j)
        reserialized = deserialized.json_serialize()
        
        assert reserialized['vertices'] == expected_vertices


@pytest.mark.parametrize('given, identical_expected', [
    ([[0.1, 2.3], [-4.5, 6.7], [8.9, 9.0], [1.2, 2.3]], False),
    ([[42., 42.], [42., 42.], [42., 42.], [42., 42.]], False),
    ([[2., 3.], [2.718281828, 3.], [2.718281828, 3.141592653], [2., 3.141592653]], True),
])
def test_fit_rectangle_to_vertices(given, identical_expected):
    given_array = np.array(given)
    vertices = _fit_rectangle_to_vertices(given_array)

    top_edge = vertices[1] - vertices[0]
    bottom_edge = vertices[2] - vertices[3]
    left_edge = vertices[3] - vertices[0]
    right_edge = vertices[2] - vertices[1]
    assert np.allclose(top_edge, bottom_edge), f"{vertices} doesn't form rectangle (top-bottom)"
    assert np.allclose(left_edge, right_edge), f"{vertices} doesn't form rectangle (left-right)"
    assert np.allclose(top_edge @ left_edge, 0), f"{vertices} doesn't form rectangle (angle)"
    if identical_expected:
        assert np.allclose(vertices, given_array), f"got {given_array}; expected {vertices}"


# NOTE:Both Polygon and Rectangle have built-in clockwise sorting.
class TestPolygonProperties:
    def test_polygon_n_less_than_3_should_raise_error(self):
        # Test with 0 vertices should raise ValueError
        with pytest.raises(ValueError, match="Polygon must have at least 3 vertices"):
            Polygon([])
        
        # Test with 1 vertex should raise ValueError
        with pytest.raises(ValueError, match="Polygon must have at least 3 vertices"):
            Polygon([[1, 2]])
        
        # Test with 2 vertices should raise ValueError
        with pytest.raises(ValueError, match="Polygon must have at least 3 vertices"):
            Polygon([[0, 0], [1, 1]])

    def test_polygon_n_equals_3(self):
        # Triangle
        vertices = [[0, 0], [3, 0], [1.5, 3]]
        polygon = Polygon(vertices)
        
        assert abs(polygon.area - 4.5) < 1e-10  # Area = 0.5 * base * height = 0.5 * 3 * 3 = 4.5
        np.testing.assert_allclose(polygon.center, [1.5, 1.0])
        
        # Test that vertices are accessible
        assert len(polygon.vertices) == 3

    def test_polygon_n_equals_4(self):
        # Square
        vertices = [[0, 0], [2, 0], [2, 2], [0, 2]]
        polygon = Polygon(vertices)
        
        assert abs(polygon.area - 4.0) < 1e-10
        np.testing.assert_allclose(polygon.center, [1.0, 1.0])

    def test_polygon_n_greater_than_4(self):
        # Pentagon (regular pentagon approximation)
        vertices = [[1, 0], [0.31, 0.95], [-0.81, 0.59], [-0.81, -0.59], [0.31, -0.95]]
        polygon = Polygon(vertices)
        
        # For a regular pentagon with radius ~1, area should be around 2.38
        assert polygon.area > 2.0 and polygon.area < 3.0
        np.testing.assert_allclose(polygon.center, [0.0, 0.0], atol=1e-10)
        # Check that we have 5 vertices
        assert len(polygon.vertices) == 5

