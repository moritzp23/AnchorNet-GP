"""The nine UCI datasets of Maddox et al. (2022), as named in bayesian_benchmarks."""
DATASETS = {
    'poletele': 'wilson_pol',
    'elevator': 'wilson_elevators',
    'bike': 'wilson_bike',
    'kin40k': 'wilson_kin40k',
    'protein': 'wilson_protein',
    '3droad': 'wilson_3droad',
    'song': 'wilson_song',
    'buzz': 'wilson_buzz',
    'houseelectric': 'wilson_houseelectric',
}

# The five train/test splits used for every configuration.
SPLITS = (1, 2, 3, 4, 73)
