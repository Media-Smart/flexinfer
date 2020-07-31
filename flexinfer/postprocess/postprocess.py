import torch


class Compose:
    """Composes several postprocess together.

    Args:
        postprocess (list of ``Postprocess`` objects): list of postprocess to compose.

    """
    def __init__(self, postprocess):
        self.postprocess = postprocess

    def __call__(self, inp, **kwargs):
        """
        Args:
            inp(torch.tensor): shape B*C*H*W
        """
        for pp in self.postprocess:
            inp = pp(inp, **kwargs)
        return inp


class SoftmaxProcess:
    def __call__(self, inp, **kwargs):
        """
        Args:
            inp(torch.tensor): shape B*C*H*W
        """
        if isinstance(inp, torch.Tensor):
            output = inp.softmax(dim=1)
            _, output = torch.max(output, dim=1)
        else:
            raise TypeError('inp shoud be torch.Tensor. Got %s' % type(inp))
        return output.unsqueeze(dim=1)


class SigmoidProcess:
    def __init__(self, threshold=0.5):
        """
        Args:
            threshold(float): threshold to determine fg or bg
        """
        self.threshold = threshold

    def __call__(self, inp, **kwargs):
        """
        Args:
            inp(torch.tensor): shape B*C*H*W
        """
        if isinstance(inp, torch.Tensor):
            output = inp.sigmoid()
            output = torch.where(output >= self.threshold,
                                 torch.full_like(output, 1),
                                 torch.full_like(output, 0))
        else:
            raise TypeError('inp shoud be torch.Tensor. Got %s' % type(inp))
        return output


class InversePad:
    def __call__(self, inp, **kwargs):
        """
        Args:
            inp(torch.tensor): shape B*C*H*W
        """
        imgs_pp = []

        shape_list = kwargs['shape_list']
        assert inp.shape[0] == len(shape_list)

        if isinstance(inp, torch.Tensor):
            output_list = inp.split(1, 0)
            for output, shape in zip(output_list, shape_list):
                output = output[:, :, :shape[0], :shape[1]]
                imgs_pp.append(output.squeeze().cpu().numpy())
        else:
            raise TypeError(
                'inp shoud be torch.Tensor. Got %s' % type(inp))

        return imgs_pp
