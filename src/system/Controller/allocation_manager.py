# system/Controllers/allocation_manager.py (or inside RunManager for Phase-1)
from system.Model.allocation.softmax import SoftmaxAllocator

class AllocationManager:
    def __init__(self, store):
        self._store = store

    def run_softmax(self, cfg, preds_df):
        alloc = SoftmaxAllocator(
            temperature=cfg.allocator.temperature,
            weight_cap=cfg.allocator.weight_cap,
        )
        weights_df = alloc.allocate(preds_df)
        self._store.save_csv(weights_df, self._store.weights_path(), index=False)
        return weights_df