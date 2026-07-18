"""Models: the shared base interface plus the BPTT and PC networks.

Every model conforms to :class:`~src.models.base.Model`. The comparison code
(RSA, iDSA) must never special-case a model — new variants just implement the
interface. The BPTT and PC nets share the SAME ``forward()`` (identical dynamics);
only their learning rule differs.
"""
