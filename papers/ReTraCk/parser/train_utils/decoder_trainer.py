# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Dict, List, Tuple
import statistics
import torch

from allennlp.nn import util

from allennlp_semparse.state_machines.constrained_beam_search import ConstrainedBeamSearch
from allennlp_semparse.state_machines.states import State
from allennlp_semparse.state_machines.trainers.maximum_marginal_likelihood import MaximumMarginalLikelihood
from allennlp_semparse.state_machines.transition_functions import TransitionFunction

logger = logging.getLogger(__name__)


class NormalizedMaximumMarginalLikelihood(MaximumMarginalLikelihood):

    def decode(
        self,
        initial_state: State,
        transition_function: TransitionFunction,
        supervision: Tuple[torch.Tensor, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        targets, target_mask = supervision
        beam_search = ConstrainedBeamSearch(self._beam_size, targets, target_mask)
        finished_states: Dict[int, List[State]] = beam_search.search(
            initial_state, transition_function
        )

        loss = 0
        for instance_states in finished_states.values():
            lens = [len(state.action_history[0]) for state in instance_states]
            scores = [state.score[0].view(-1) for state in instance_states]
            # log_sum_exp dividing the average length
            loss += - util.logsumexp(torch.cat(scores)) / statistics.mean(lens)
        return {"loss": loss / len(finished_states)}
