"""Offline, LLM-shaped training on the maze: next-token prediction on demonstrations.

Everything else in the project trains a recurrent agent online by RL. This
package builds the same experiment in the form a language model takes: a
small autoregressive transformer sees the maze once, as a sequence of cell
tokens, and emits its whole route as action tokens, trained by cross-entropy on
expert demonstrations. The demonstrations, not the reward, carry the
colour-value correlation - so misgeneralisation here would arise from
imitation of correlated data, which is the LLM case.

``demos`` builds and stores the demonstration sets, ``model`` is the
transformer, ``train`` fits it, ``decode`` turns it back into routes the
existing behavioural metrics can score, and ``probe`` reads its residual stream
in the shape the existing per-cell probes expect.
"""
