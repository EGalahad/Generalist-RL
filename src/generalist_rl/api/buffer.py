from generalist_rl.api.datatypes import SampleBatch

class Buffer:
    @property
    def qsize(self) -> int:
        raise NotImplementedError()

    def put(self, sample: SampleBatch):
        raise NotImplementedError()

    def get(self) -> SampleBatch:
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()
