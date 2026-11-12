"""Second registered local-eligibility hypothesis; forward ALIF stays unchanged."""
import numpy as np
from .r8_matched_credit import MatchedLocalCredit

ARMS=('multi_gated','multi_gated_shuffle','multi_gated_zero','multi_gated_readout_only')


class StoppedResetCredit(MatchedLocalCredit):
    def __init__(self,config,arm='multi_gated'):
        if arm not in ARMS:raise ValueError('Unregistered stopped-reset arm')
        super().__init__(config,arm);self.reset_mode='stopped'

    def step(self,x,track=True):
        row=np.asarray(x,np.float32);c=self.config
        if row.shape!=(c.inputs,) or not np.isfinite(row).all():raise ValueError('Invalid input vector')
        if track:
            pre=np.concatenate((np.broadcast_to(row,(c.hidden,c.inputs)),self.core.spikes[self.core.indices]),axis=1)
            u_e=c.leak*self.e_v+pre
        features=super().step(row,track)
        # Parent uses the identical current-tick z_e/e_a/filter/forward. Only the
        # next-tick voltage derivative drops the spike-reset derivative terms.
        if track:self.e_v[:]=u_e
        return features
