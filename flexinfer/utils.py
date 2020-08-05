import os


def set_device(gpu_id=0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


class Compose:
    """Composes several operations together.
    Args:
        postprocess (list): list of operations to compose.
    """

    def __init__(self, ops):
        self.ops = ops

    def __call__(self, inp, **kwargs):
        """
        Args:
            inp(torch.tensor or np.ndarray)
        """
        for op in self.ops:
            inp = op(inp, **kwargs)

        return inp
