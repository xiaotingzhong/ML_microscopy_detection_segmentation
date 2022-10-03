hps_input = {'lr': [0.01, 0.001],
             'bacth_size': [32, 64],
             'flip': [True, False]}

def get_hp_combinations(hps_input):
    """
    input
        hps_input, a dictionary whose keys are the hyper parameters and each key associates with a list
            {'hp_a': [hp_a_1, hp_a_2, ...], 'hp_b': [hp_b_1, hp_b_2, ...], ...}
    output
        hp_settings, a list containing dicts:  each dict contain one experiment setting
            [{'hp_a': hp_a_1, 'hp_b': hp_b_1, ...}, {'hp_a': hp_a_2, 'hp_b': hp_b_2, ...}, ...]
    """
    import itertools

    hps = []
    for key in hps_input.keys():
        tmp = []
        for item in hps_input[key]:
            tmp.append({key: item})
        hps.append(tmp)
    tmp2 = list(itertools.product(*hps))

    hp_settings = []
    for setting_tuple in tmp2:
        setting_dict = {}
        for setting in setting_tuple:
            setting_dict.update(setting)
        hp_settings.append(setting_dict)
    return hp_settings

