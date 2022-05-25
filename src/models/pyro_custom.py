from torch.distributions import constraints
from torch.distributions.transforms import AbsTransform

from pyro.distributions.torch import TransformedDistribution
from torch.distributions.transforms import Transform

from torch.nn import Threshold

class IntervalTransform(Transform):
    r"""
    Transform via the mapping :math:`y = a <= x <= b`.
    """
    domain = constraints.real
    codomain = constraints.positive
    def __init__(self, a, b, event_dim=0, cache_size=0):
        self.a = a
        self.b = b
        self._event_dim = event_dim
        self._cache_size = cache_size

    def __eq__(self, other):
        return isinstance(other, IntervalTransform(self.a, self.b))

    def _call(self, x):
        x[self.a > x] = self.a
        x[self.b < x] = self.b
        
        # t1 = Threshold(self.a, self.a, inplace=False)
        # t2 = Threshold(-self.b, self.b, inplace=False)
        
        # x = t1(-x).abs()
        # x = t2(x)

        return x

    def _inverse(self, y):
        return y

class IntervalFoldedDistribution(TransformedDistribution):
    """
    Equivalent to ``TransformedDistribution(base_dist, AbsTransform())``,
    but additionally supports :meth:`log_prob` .

    :param ~torch.distributions.Distribution base_dist: The distribution to
        reflect.
    """

    support = constraints.positive

    def __init__(self, base_dist, validate_args=None, lower=8e3, upper=1.4e4):
        if base_dist.event_shape:
            raise ValueError("Only univariate distributions can be folded.")
        super().__init__(base_dist, IntervalTransform(a=lower, b=upper), validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        return super().expand(batch_shape, _instance=new)


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        dim = max(len(self.batch_shape), value.dim())
        plus_minus = value.new_tensor([1.0, -1.0]).reshape((2,) + (1,) * dim)
        return self.base_dist.log_prob(plus_minus * value).logsumexp(0)