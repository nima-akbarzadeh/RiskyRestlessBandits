The code Neurwin-risky.py implements risk-aware reinforcement learning algorithm
to learn dynamics of arms in any indexable restless bandits.

The approach estimates the safe Whittle index by a neural Whittle index network
which generates the Whittle index for each state of arms independently.
