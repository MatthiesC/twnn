config = {
    ### Option 1: Define classes for multiclass classifier
    'classes': {
        'sig': {
            'nice_name': 'tW signal',
            'subclasses': {
                'ST_tW_DR': {
                    'file': 'uhh2.AnalysisModuleRunner.MC.ST_tW_DR.root',
                    'cut': 'btw_dnnClass_tW_int == 1',
                    'theWeight': 'weight * 0.5',
                },
                'ST_tW_DS': {
                    'file': 'uhh2.AnalysisModuleRunner.MC.ST_tW_DS.root',
                    'cut': 'btw_dnnClass_tW_int == 1',
                    'theWeight': 'weight * 0.5',
                },
            },
        },
        'top': {
            'nice_name': 'Other top',
            'subclasses': {
                'TTbar': {
                    'file': 'uhh2.AnalysisModuleRunner.MC.TTbar.root',
                    'theWeight': 'weight / weight_toppt_applied',
                    # 'fraction': 0.1,
                },
                'ST_tW_DR': {
                    'file': 'uhh2.AnalysisModuleRunner.MC.ST_tW_DR.root',
                    'cut': 'btw_dnnClass_tW_int == 2',
                    'theWeight': 'weight * 0.5',
                },
                'ST_tW_DS': {
                    'file': 'uhh2.AnalysisModuleRunner.MC.ST_tW_DS.root',
                    'cut': 'btw_dnnClass_tW_int == 2',
                    'theWeight': 'weight * 0.5',
                },
                'ST_otherChannels': {
                    'file': 'uhh2.AnalysisModuleRunner.MC.ST_otherChannels.root',
                },
            },
        },
        'ewk': {
            'nice_name': 'V+jets, VV',
            'subclasses': {
                'VJetsAndVV': {
                    'file': 'uhh2.AnalysisModuleRunner.MC.VJetsAndVV.root',
                },
            },
        },
    },
    # ### Option 2: Define signal/background for binary classifier
    # ### Stick with class names 'sig' and 'bkg'!
    # 'classes': {
    #     'sig': [
    #         ['ST_tW_DR_trueSig', 'ST_tW_DS_trueSig'],
    #     ],
    #     'bkg': [
    #         'TTbar',
    #         ['ST_tW_DR_trueBkg', 'ST_tW_DS_trueBkg'],
    #         'ST_otherChannels',
    #         'WJets',
    #         'DYJets',
    #         'Diboson',
    #     ],
    # },
    'fraction_train': 0.7, # rest will be shared between validation and test samples equally
    'normalize_sample_weights': True, # normalizes sum of event weights per class to 1; effectively, this augments underrepresented classes
    'batch_norm': True,
    # 'input_dropout': 0.5, # probably only good for images etc. but not for tabular data like mine
    'hidden_layers': [
        {
            'units': 256,
            'activation': 'relu',
            'dropout': 0.5,
        },
        {
            'units': 256,
            'activation': 'relu',
            'dropout': 0.5,
        },
        {
            'units': 256,
            'activation': 'relu',
            'dropout': 0.5,
        },
        {
            'units': 256,
            'activation': 'relu',
            'dropout': 0.5,
        },
    ],
    # 'optimizer': {
    #     'name': 'Adam',
    #     'kwargs': {
    #         'learning_rate': 0.0001,
    #         'decay': 0.00025,
    #     },
    # },
    'lr_schedule': {
        'name': 'ExponentialDecay',
        'kwargs': {
            # 'initial_learning_rate': 0.0001,
            'initial_learning_rate': 0.001,
            'decay_steps': 10000,
            'decay_rate': 0.9,
            'staircase': False,
        },
    },
    'epochs': 1000,
    'batch_size': 8192,
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 50,
        'verbose': 1,
    },
    'input_prefix': 'dnnInput__',
    'inputs': [
        # 'year', # years should be one-hot encoded, else NN might e.g. learn that 2016 < 2017 < 2018 and that '2017' is an 'average'
        'channel', # one-hot encoding not needed since we have a binarized channel setup: only ele or muo

        'lepton_pt_log',
        'lepton_eta',
        'lepton_phi',
        'lepton_energy_log',
        'lepton_charge',
        # 'lepton_relpfiso',

        'met_pt_log',
        'met_phil',
        'met_phil_cos',
        'met_phil_sin',

        'nu_pt_log',
        'nu_eta',
        'nu_energy_log',

        'hemijet_pt_log',
        'hemijet_eta',
        'hemijet_phil',
        'hemijet_phil_cos',
        'hemijet_phil_sin',
        'hemijet_energy_log',
        'hemijet_deepjet',

        'hemi_pt_log',
        'hemi_eta',
        'hemi_phil',
        'hemi_phil_cos',
        'hemi_phil_sin',
        'hemi_energy_log',
        'hemi_mass_log',

        'thetag_pt_log',
        'thetag_eta',
        'thetag_phil',
        'thetag_phil_cos',
        'thetag_phil_sin',
        'thetag_energy_log',

        'hadrecoil_pt_log',
        'hadrecoil_eta',
        'hadrecoil_phil',
        'hadrecoil_phil_cos',
        'hadrecoil_phil_sin',
        'hadrecoil_energy_log',

        'ak4_pt_sum_log',
        'ak4_multiplicity',

        'ak4jet[1-6]_pt_log',
        'ak4jet[1-6]_eta',
        'ak4jet[1-6]_phil',
        'ak4jet[1-6]_phil_cos',
        'ak4jet[1-6]_phil_sin',
        'ak4jet[1-6]_energy_log',
        'ak4jet[1-6]_deepjet',
    ],
    # 'additonal_variables_to_read': [
    #     'btw_bool_reco_sel',
    #     'btw_Region_heavyTags_int',
    #     'btw_Region_bTags_int',
    #     'btw_dnnClass_tW_int',
    # ],
    'cut': '(btw_bool_reco_sel == True) & (btw_Region_heavyTags_int == 1) & (btw_Region_bTags_int == 2)',
    # 'cuts': [
    #     'btw_bool_reco_sel == True',
    #     'btw_Region_heavyTags_int == 1', # 1: 1t, 2: 0t1W
    #     'btw_Region_bTags_int == 2', # 2: ==1 b-tag, 3: >=2 b-tags
    # ],
}

from codebase.classifier import SimpleEventClassifier
nn = SimpleEventClassifier(config)
nn.get_input_keys()
nn.prepare_data(load_from_scratch=False)
# nn.prepare_data(load_from_scratch=True)
# nn.init_model(from_checkpoint=927)
nn.init_model()
nn.setup_callbacks()
nn.fit()
nn.predict()
