"""Policy networks of our own, plugged into cleanba through ``PolicySpec``.

The DRC and the ResNet are cleanba's. Anything here exists to ask whether a
result is a property of the task or of the one architecture it was found on,
so each network keeps the same contract cleanba's ``Policy`` expects - a flat
hidden vector for the actor and critic heads - and exposes a per-cell state
that the existing probes can read.
"""
