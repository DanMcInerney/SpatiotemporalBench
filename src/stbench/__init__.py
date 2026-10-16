"""SpatiotemporalBench: realistic simulated worlds that change over time, recorded through a robot's sensors."""

# The string in the sub-seed keys of everything the first generator drew (items, schedule, route, clock, the gaps'
# changes, and the compiler's carried-item pool). It is frozen at "0.4.0" so that the worlds drawn since stay
# byte-identical to the kept 0.4.0 dataset (60 episodes, rendered and gold-compiled): a different string in those keys
# would redraw every world and need a re-render. It is a seed, not a version. The movers (T3), added after 0.4.0, have a
# frozen key of their own, `movers.MOVER_SEED_KEY`; the warehouse draws from both, whose keys hold the map name. Never
# change either (DECISIONS, "T3 contracts").
SEED_KEY = "0.4.0"
