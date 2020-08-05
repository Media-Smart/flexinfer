import torch


class SoftmaxProcess:
    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, inp, **kwargs):
        """
        Args:
            inp(torch.tensor): tensor, shape B*C*H*W
        """
        if isinstance(inp, torch.Tensor):
            output = inp.softmax(dim=self.dim)
            _, output = torch.max(output, dim=self.dim)
        else:
            raise TypeError('tensor shoud be torch.Tensor. Got %s' % type(inp))
        return output.unsqueeze(dim=self.dim)


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
            inp(torch.tensor): tensor, shape B*C*H*W
        """
        if isinstance(inp, torch.Tensor):
            output = inp.sigmoid()
            output = torch.where(output >= self.threshold,
                                 torch.full_like(output, 1),
                                 torch.full_like(output, 0))
        else:
            raise TypeError('tensor shoud be torch.Tensor. Got %s' % type(inp))
        return output


class IndexToString:
    """
    Current is designed for CTC decoder.
    TODO: Attn-based, FC-based.
    """

    def __init__(self, character):
        self.character = ['[blank]'] + list(character)

    def __call__(self, inp, **kwargs):
        texts = []
        batch_size = inp.shape[0]
        length = inp.shape[1]
        for i in range(batch_size):
            t = inp[i]
            char_list = []
            for idx in range(length):
                if t[idx] != 0 and (not (idx > 0 and t[idx - 1] == t[idx])):
                    char_list.append(self.character[t[idx]])
            text = ''.join(char_list)
            texts.append(text)
        return texts


class InversePad:
    def __call__(self, inp, **kwargs):
        """
        Args:
            inp(torch.tensor): tensor, shape B*C*H*W
        """
        imgs_pp = []

        shape_list = kwargs['shape_list']
        assert inp.shape[0] == len(shape_list)

        if isinstance(inp, torch.Tensor):
            output_list = inp.split(1, 0)
            for output, shape in zip(output_list, shape_list):
                output = output[:, :, :shape[0], :shape[1]]
                imgs_pp.append(output.squeeze())
        else:
            raise TypeError('tensor shoud be torch.Tensor. Got %s' % type(inp))

        return imgs_pp
