import torch
import torch.nn as nn
import torch.distributions as dist


class PointDistribution(dist.Distribution):
    """
    The Delta Distribution where all the mass is concentrated in a single point.
    """

    def __init__(self, loc, scale=None):
        super(PointDistribution, self).__init__(
            batch_shape=loc.shape[:-1], event_shape=loc.shape[-1:], validate_args=False
        )
        self.loc = loc

        if scale == None:
            self.scale = torch.linalg.norm(loc, dim=1)
        else:
            self.scale = scale
        self.scale = torch.squeeze(self.scale)


    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev ** 2

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return torch.eq(value, self.loc).float().log()

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return self.loc.expand(shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return self.loc.expand(shape)
