#!/usr/bin/env python3
"""
Unit tests for maplab_map_stats.py module
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

from lovot_slam.map_build.maplab_map_stats import MissionMetrics, MaplabMapStatsParser


class TestMissionMetrics:
    """Test cases for MissionMetrics attrs class"""

    def test_mission_metrics_creation(self):
        """Test creating MissionMetrics instance"""
        metrics = MissionMetrics(landmark_count=1234)
        assert metrics.landmark_count == 1234

    def test_mission_metrics_frozen(self):
        """Test that MissionMetrics is frozen (immutable)"""
        metrics = MissionMetrics(landmark_count=1234)

        # Should not be able to modify after creation
        with pytest.raises(AttributeError):
            metrics.landmark_count = 5678

    def test_mission_metrics_equality(self):
        """Test MissionMetrics equality comparison"""
        metrics1 = MissionMetrics(landmark_count=1234)
        metrics2 = MissionMetrics(landmark_count=1234)
        metrics3 = MissionMetrics(landmark_count=5678)

        assert metrics1 == metrics2
        assert metrics1 != metrics3


class TestMaplabMapStatsParser:
    """Test cases for MaplabMapStatsParser class"""

    @pytest.fixture
    def test_yaml_file_path(self):
        """Fixture providing the path to the test YAML file"""
        return Path(__file__).parent / 'map_stats.yaml'

    def test_parser_initialization_with_existing_file(self, test_yaml_file_path):
        """Test parser initialization with existing YAML file"""
        parser = MaplabMapStatsParser(test_yaml_file_path)
        assert parser._yaml_file_path == test_yaml_file_path
        assert parser._stats_data is not None

    def test_parser_initialization_with_nonexistent_file(self):
        """Test parser initialization with non-existent YAML file"""
        non_existent_path = Path('/non/existent/file.yaml')

        with patch('lovot_slam.map_build.maplab_map_stats._logger') as mock_logger:
            parser = MaplabMapStatsParser(non_existent_path)
            assert parser._stats_data is None
            mock_logger.warning.assert_called_once()
            assert 'Map statistics YAML not found' in mock_logger.warning.call_args[0][0]

    def test_parser_with_invalid_yaml(self):
        """Test parser with invalid YAML content"""
        invalid_yaml_content = "invalid: yaml: content: [unclosed"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml_content)
            temp_path = Path(f.name)

        try:
            with patch('lovot_slam.map_build.maplab_map_stats._logger') as mock_logger:
                parser = MaplabMapStatsParser(temp_path)
                assert parser._stats_data is None
                mock_logger.error.assert_called_once()
                assert 'Failed to load YAML' in mock_logger.error.call_args[0][0]
        finally:
            temp_path.unlink()

    def test_accumulated_metrics_with_real_data(self, test_yaml_file_path):
        """Test accumulated_metrics property with real test data"""
        parser = MaplabMapStatsParser(test_yaml_file_path)
        accumulated_metrics = parser.accumulated_metrics

        assert accumulated_metrics is not None
        assert isinstance(accumulated_metrics, MissionMetrics)
        assert accumulated_metrics.landmark_count == 689190

    def test_accumulated_metrics_with_missing_data(self):
        """Test accumulated_metrics property when data is missing"""
        yaml_content = {
            'missions': []
            # No accumulated_stats section
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = Path(f.name)

        try:
            parser = MaplabMapStatsParser(temp_path)
            accumulated_metrics = parser.accumulated_metrics
            assert accumulated_metrics is None
        finally:
            temp_path.unlink()

    def test_mission_metrics_with_real_data(self, test_yaml_file_path):
        """Test mission_metrics property with real test data"""
        parser = MaplabMapStatsParser(test_yaml_file_path)
        mission_metrics = parser.mission_metrics

        assert isinstance(mission_metrics, dict)
        assert len(mission_metrics) == 23  # Based on the test YAML file

        # Test some specific missions from the test data
        expected_missions = {
            '00cfd434fdbe07180e00000000000000': 35246,
            '6d9af4879fc407180e00000000000000': 27844,
            '0af4f0d64cca07180e00000000000000': 28222,
            '975fb6852ed007180e00000000000000': 35458,
            '1fab8423bba134180e00000000000000': 30993  # Last mission
        }

        for mission_id, expected_count in expected_missions.items():
            assert mission_id in mission_metrics
            assert isinstance(mission_metrics[mission_id], MissionMetrics)
            assert mission_metrics[mission_id].landmark_count == expected_count

    def test_mission_metrics_with_missing_mission_id(self):
        """Test mission_metrics property when mission_id is missing"""
        yaml_content = {
            'missions': [
                {
                    # No mission_id field
                    'statistics': {
                        'landmarks': {
                            'total': 1234
                        }
                    }
                },
                {
                    'mission_id': 'valid_mission_id',
                    'statistics': {
                        'landmarks': {
                            'total': 5678
                        }
                    }
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = Path(f.name)

        try:
            parser = MaplabMapStatsParser(temp_path)
            mission_metrics = parser.mission_metrics

            # Should only contain the valid mission
            assert len(mission_metrics) == 1
            assert 'valid_mission_id' in mission_metrics
            assert mission_metrics['valid_mission_id'].landmark_count == 5678
        finally:
            temp_path.unlink()

    def test_mission_metrics_with_missing_landmark_count(self):
        """Test mission_metrics property when landmark count is missing"""
        yaml_content = {
            'missions': [
                {
                    'mission_id': 'mission_with_landmarks',
                    'statistics': {
                        'landmarks': {
                            'total': 1234
                        }
                    }
                },
                {
                    'mission_id': 'mission_without_landmarks',
                    'statistics': {
                        # No landmarks section
                    }
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = Path(f.name)

        try:
            parser = MaplabMapStatsParser(temp_path)
            mission_metrics = parser.mission_metrics

            # Should only contain the mission with valid landmark count
            assert len(mission_metrics) == 1
            assert 'mission_with_landmarks' in mission_metrics
            assert mission_metrics['mission_with_landmarks'].landmark_count == 1234
            assert 'mission_without_landmarks' not in mission_metrics
        finally:
            temp_path.unlink()

    def test_mission_metrics_empty_missions_list(self):
        """Test mission_metrics property with empty missions list"""
        yaml_content = {
            'missions': [],
            'accumulated_stats': {
                'total_landmarks': {
                    'total': 1234
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = Path(f.name)

        try:
            parser = MaplabMapStatsParser(temp_path)
            mission_metrics = parser.mission_metrics

            assert isinstance(mission_metrics, dict)
            assert len(mission_metrics) == 0
        finally:
            temp_path.unlink()

    def test_both_properties_with_no_data(self):
        """Test both properties when YAML loading fails"""
        non_existent_path = Path('/non/existent/file.yaml')

        with patch('lovot_slam.map_build.maplab_map_stats._logger'):
            parser = MaplabMapStatsParser(non_existent_path)

            assert parser.accumulated_metrics is None
            assert parser.mission_metrics == {}

    def test_zero_landmark_counts(self):
        """Test handling of zero landmark counts (edge case)"""
        yaml_content = {
            'missions': [
                {
                    'mission_id': 'zero_landmarks_mission',
                    'statistics': {
                        'landmarks': {
                            'total': 0
                        }
                    }
                }
            ],
            'accumulated_stats': {
                'total_landmarks': {
                    'total': 0
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = Path(f.name)

        try:
            parser = MaplabMapStatsParser(temp_path)

            # Both should handle zero values correctly
            accumulated_metrics = parser.accumulated_metrics
            assert accumulated_metrics is not None
            assert accumulated_metrics.landmark_count == 0

            mission_metrics = parser.mission_metrics
            assert len(mission_metrics) == 1
            assert mission_metrics['zero_landmarks_mission'].landmark_count == 0
        finally:
            temp_path.unlink()

    def test_file_permission_error(self):
        """Test handling of file permission errors"""
        test_path = Path('/tmp/test_file.yaml')

        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with patch.object(Path, 'exists', return_value=True):
                with patch('lovot_slam.map_build.maplab_map_stats._logger') as mock_logger:
                    parser = MaplabMapStatsParser(test_path)

                    assert parser._stats_data is None
                    mock_logger.error.assert_called_once()
                    assert 'Failed to load YAML' in mock_logger.error.call_args[0][0]
