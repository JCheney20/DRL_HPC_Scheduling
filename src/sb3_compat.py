"""Compatibility shims for the pinned sb3-contrib. Import for side effects.

Must be imported before any MaskableCategorical is constructed; src/utils.py
does that for the whole project, and everything that touches an SB3 policy
imports src/utils.py.

--------------------------------------------------------------------------
Why this exists: the masked-A2C Simplex() crash
--------------------------------------------------------------------------
Masked runs died mid-training with

    ValueError: Expected parameter probs (Tensor of shape (20, 513)) ...
                to satisfy the constraint Simplex()

and the tensor torch printed looked perfectly healthy, because torch elides
all but the corner entries of a (20, 513) tensor. It is not divergence: at the
crash point the policy weights, the observations, the latents and the logits
are all finite, and the masked probs recompute to a valid distribution.

The failure is a stale cache. MaskableCategorical.apply_masking ends with

    super().__init__(logits=logits)      # re-init, this is where validation runs
    self.probs = logits_to_probs(logits) # cache probs on the instance

and apply_masking is called *twice* per forward pass:

  1. from MaskableCategorical.__init__, with masks=None
     (MaskableCategoricalDistribution.proba_distribution builds the
     distribution unmasked), which caches probs = softmax(raw logits);
  2. from MaskableActorCriticPolicy, with the real action masks.

torch.distributions.Distribution.__init__ skips checking a parameter only
while it is still a lazy_property -- "if param not in self.__dict__". Call 1
put probs in __dict__, so call 2 validates it. The tensor that fails the
Simplex check is therefore the *unmasked* softmax left over from call 1, not
the masked distribution being built. Since that assignment is the only route
by which probs enters __dict__ in 2.6.0, a Simplex error naming probs can only
be the previous call's cache.

How far that cache has to drift is just float32 summation error against a
1e-6 tolerance. Measured on CPU with peaked logits matching the crash state,
513 categories reaches about 6e-7 -- marginal, under the tolerance, and not
reproducible synthetically at this size; 4096 categories trips it every time,
which is why upstream describes this as a "many categories" bug. Training ran
on CUDA, whose softmax and sum reductions accumulate in a different order than
CPU's, and that is where the observed run crossed. Note the drift is not a
symptom of anything wrong: the run dies deterministically at the same timestep
because the arithmetic is deterministic, not because the policy is unhealthy.

That also explains the observations that ruled out every divergence theory:
unmasked `a2c` never crashes however saturated it gets, because stock SB3 uses
a plain Categorical whose probs stay lazy and are never validated; and runs
with ent_coef=0.01 never crash because a near-uniform policy sums cleanly.

Upstream fixed this in sb3-contrib 2.9.0 by dropping the cached probs before
the re-init. nix/sb3-contrib.nix pins 2.6.0, which predates the fix, so we
apply the same one-liner here. Popping a key that upstream also pops is a
no-op, so this stays correct if the pin is ever bumped.

Note this deliberately does *not* disable argument validation globally
(th.distributions.Distribution.set_default_validate_args(False)), which would
also stop the crash: the logits are still checked on every re-init, so a
genuinely non-finite policy head would still raise instead of silently
sampling garbage.
"""

from __future__ import annotations

from sb3_contrib.common.maskable.distributions import MaskableCategorical

_PATCH_FLAG = "_sb3_compat_clears_stale_probs"


def _patch_stale_probs_cache() -> None:
    if getattr(MaskableCategorical, _PATCH_FLAG, False):
        return

    original_apply_masking = MaskableCategorical.apply_masking

    def apply_masking(self, masks):
        # Drop the probs cached by the previous call so the re-init inside
        # apply_masking sees probs as an unset lazy_property and skips the
        # Simplex check on it. The correct masked probs are cached immediately
        # afterwards by the original method, so nothing downstream changes.
        self.__dict__.pop("probs", None)
        return original_apply_masking(self, masks)

    apply_masking.__doc__ = original_apply_masking.__doc__
    MaskableCategorical.apply_masking = apply_masking
    setattr(MaskableCategorical, _PATCH_FLAG, True)


_patch_stale_probs_cache()
