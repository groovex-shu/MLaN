from lovot_slam.subprocess.subprocess import SubprocessBase
from pathlib import Path
import anyio

class MockBuildMapSubprocess(SubprocessBase):
    """Mock that logs all subprocess calls and can use your existing results"""

    def __init__(self, model, output_to_console: bool = False, journal: bool = False):
        super().__init__(output_to_console)
        self._name = ""
        self.calls = []  # Track all calls

    def start_bag_conversion(self, original_bag: str, converted_bag: str):
        self._name = "bag_conversion"
        call_info = {
            'step': 'bag_conversion',
            'original_bag': original_bag,
            'converted_bag': converted_bag
        }
        self.calls.append(call_info)
        print(f"✓ Step 1: bag_conversion")
        print(f"  Input:  {Path(original_bag).name}")
        print(f"  Output: {Path(converted_bag).name}")

        # Use your existing converted bag if available, otherwise create dummy
        if Path(original_bag).exists():
            # Check if you have the real converted bag result
            if not Path(converted_bag).exists():
                Path(converted_bag).touch()
                print(f"  Created dummy (replace with your real result)")
            else:
                print(f"  Using existing result")
        cmd = ['sleep', '0.1']
        self._start_process(cmd)

    def start_bag_diminish(self, original_bag: str, topics: str, vertices_csv: str, converted_bag: str):
        self._name = "bag_diminish"
        call_info = {
            'step': 'bag_diminish',
            'original_bag': original_bag,
            'topics': topics,
            'vertices_csv': vertices_csv,
            'converted_bag': converted_bag
        }
        self.calls.append(call_info)
        print(f"✓ Step 2: bag_diminish")
        print(f"  Input:  {Path(original_bag).name}")
        print(f"  Topics: {topics}")
        print(f"  Output: {Path(converted_bag).name}")

        if Path(original_bag).exists():
            if not Path(converted_bag).exists():
                Path(converted_bag).touch()
        cmd = ['sleep', '0.1']
        self._start_process(cmd)

    def start_bag_prune(self, original_bag: str, topics: str, converted_bag: str):
        self._name = "bag_prune"
        call_info = {
            'step': 'bag_prune',
            'original_bag': original_bag,
            'topics': topics,
            'converted_bag': converted_bag
        }
        self.calls.append(call_info)
        print(f"✓ Step 3: bag_prune")
        print(f"  Input:  {Path(original_bag).name}")
        print(f"  Topics: {topics}")
        print(f"  Output: {Path(converted_bag).name}")

        if Path(original_bag).exists():
            if not Path(converted_bag).exists():
                Path(converted_bag).touch()
        cmd = ['sleep', '0.1']
        self._start_process(cmd)

    def start_build_feature_map(self, converted_bag: str, map_dir: str, config_dir: str):
        self._name = "build_feature_map"
        call_info = {
            'step': 'build_feature_map',
            'converted_bag': converted_bag,
            'map_dir': map_dir,
            'config_dir': config_dir
        }
        self.calls.append(call_info)
        print(f"✓ Step 4: build_feature_map")
        print(f"  Input bag: {Path(converted_bag).name}")
        print(f"  Map dir:   {Path(map_dir).name}")
        print(f"  Config:    {Path(config_dir).name}")

        # You can copy your existing feature map here if you have it
        # Example: shutil.copytree(your_existing_map, map_dir)

        cmd = ['sleep', '0.1']
        self._start_process(cmd)

    def start_merge_feature_maps(self, input_map: str, output_map: str, maps_to_append):
        self._name = "merge_feature_maps"
        call_info = {
            'step': 'merge_feature_maps',
            'input_map': input_map,
            'output_map': output_map,
            'maps_to_append': maps_to_append
        }
        self.calls.append(call_info)
        print(f"✓ Step 5: merge_feature_maps")
        print(f"  Base map:  {Path(input_map).name}")
        print(f"  Output:    {Path(output_map).name}")
        print(f"  Appending: {maps_to_append}")

        # You can copy your existing merged map here

        cmd = ['sleep', '0.1']
        self._start_process(cmd)

    def start_scale_map(self, map_name: str, source_maps, mission_ids):
        self._name = "scale_map"
        call_info = {
            'step': 'scale_map',
            'map_name': map_name,
            'source_maps': source_maps,
            'mission_ids': mission_ids
        }
        self.calls.append(call_info)
        print(f"✓ Step 6: scale_map")
        print(f"  Map: {map_name}")
        cmd = ['sleep', '0.1']
        self._start_process(cmd)
