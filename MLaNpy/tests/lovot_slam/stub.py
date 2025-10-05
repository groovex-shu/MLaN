class PseudoSlamManager:
    (
        STATE_IDLE,
        STATE_BAG_CONVERSION,
        STATE_BUILD_FEATURE_MAP,
        STATE_SCALE_MAP,
        STATE_BUILD_DENSE_MAP,
        STATE_BUILD_ERROR,
    ) = range(0, 6)

    def __init__(self):
        pass

    def change_state(self, state):
        self.state = state

    def is_processing_map(self):
        return self.STATE_IDLE != self.state
