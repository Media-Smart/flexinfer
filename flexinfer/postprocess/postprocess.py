import torch


class Compose:
    def __init__(self, postprocess):
        self.postprocess = postprocess

    def __call__(self, tensor, **kwargs):
        """
        Args:
            tensor(torch.tensor): tensor, shape B*C*H*W
        """
        for pp in self.postprocess:
            tensor = pp(tensor, **kwargs)
        return tensor


class SoftmaxProcess:
    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, tensor, **kwargs):
        """
        Args:
            tensor(torch.tensor): tensor, shape B*C*H*W
        """
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.softmax(dim=self.dim)
            _, output = torch.max(tensor, dim=self.dim)
        else:
            raise TypeError('tensor shoud be torch.Tensor. Got %s' % type(tensor))
        return output.unsqueeze(dim=self.dim)


class SigmoidProcess:
    def __init__(self, threshold=0.5):
        """
        Args:
            threshold(float): threshold to determine fg or bg
        """
        self.threshold = threshold

    def __call__(self, tensor, **kwargs):
        """
        Args:
            tensor(torch.tensor): tensor, shape B*C*H*W
        """
        if isinstance(tensor, torch.Tensor):
            output = tensor.sigmoid()
            output = torch.where(output >= self.threshold,
                                 torch.full_like(output, 1),
                                 torch.full_like(output, 0))
        else:
            raise TypeError('tensor shoud be torch.Tensor. Got %s' % type(tensor))
        return output


class IndexToString:
    """
    Current is designed for CTC decoder.
    TODO: Attn-based, FC-based.
    """

    def __init__(self, character):
        self.character = ['[blank]'] + list(character)

    def __call__(self, tensor, **kwargs):
        print(tensor)
        texts = []
        batch_size = tensor.shape[0]
        length = tensor.shape[1]
        for i in range(batch_size):
            t = tensor[i]
            char_list = []
            for idx in range(length):
                if t[idx] != 0 and (not (idx > 0 and t[idx - 1] == t[idx])):
                    char_list.append(self.character[t[idx]])
            text = ''.join(char_list)
            texts.append(text)
        return texts


class InversePad:
    def __call__(self, tensor, **kwargs):
        """
        Args:
            tensor(torch.tensor): tensor, shape B*C*H*W
        """
        imgs_pp = []

        shape_list = kwargs['shape_list']
        assert tensor.shape[0] == len(shape_list)

        if isinstance(tensor, torch.Tensor):
            output_list = tensor.split(1, 0)
            for output, shape in zip(output_list, shape_list):
                output = output[:, :, :shape[0], :shape[1]]
                imgs_pp.append(output.squeeze())
        else:
            raise TypeError('tensor shoud be torch.Tensor. Got %s' % type(tensor))

        return imgs_pp
