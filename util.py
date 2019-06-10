import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix
import numpy as np

from xrsdkit import definitions as xrsdefs

import xrsdkit
from xrsdkit.models import load_models, get_regression_models 
load_models('../xrsdkit_modeling/flowreactor_pd_nanoparticles/models')
from xrsdkit.models.predict import system_from_prediction, predict

from xrsdkit.tools.profiler import profile_keys

regression_models = get_regression_models()

#by https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(9,9))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def get_true_predicted_crossvalid(sys_cls, data):
    features = profile_keys
    results = {}
    sys_cls_data = data.loc[data['system_class']==sys_cls].copy()
    all_noise_models = list(sys_cls_data['noise_model'].unique())
    for modnm in all_noise_models:
            noise_model_data = sys_cls_data.loc[sys_cls_data['noise_model']==modnm].copy()
            for pnm in list(xrsdefs.noise_params[modnm].keys())+['I0_fraction']:
                if not pnm == 'I0':
                    param_header = 'noise_'+pnm
                    if (sys_cls in regression_models) \
                    and ('noise' in regression_models[sys_cls]) \
                    and (modnm in regression_models[sys_cls]['noise']) \
                    and (pnm in regression_models[sys_cls]['noise'][modnm]) \
                    and (regression_models[sys_cls]['noise'][modnm][pnm].trained): 
                        model = regression_models[sys_cls]['noise'][modnm][pnm]
                        name = sys_cls + "_noise_"  + modnm 
                        
                        results[name] = {}
                        results[name]['true_y'] = noise_model_data[model.target]
                        group_ids, _ = model.group_by_pc1(noise_model_data,features)
                        noise_model_data['group_id'] = group_ids
                        results[name]['cross_val'] =  model.run_cross_validation(noise_model_data)
                        
                        s_d = model.standardize(noise_model_data, features)
                        results[name]['predicted'] =  model.model.predict(s_d[model.features])
            # use the sys_cls to identify the populations and their structures
    for ipop,struct in enumerate(sys_cls.split('__')):
        pop_id = 'pop{}'.format(ipop)
        #param_header = pop_id+'_I0_fraction'
            
        if (sys_cls in regression_models) \
            and (pop_id in regression_models[sys_cls]) \
            and ('I0_fraction' in regression_models[sys_cls][pop_id]) \
            and (regression_models[sys_cls][pop_id]['I0_fraction'].trained): 
            model = regression_models[sys_cls][pop_id]['I0_fraction']
            name = sys_cls + "_I0_fraction_" + pop_id 
                        
            results[name] = {}
            results[name]['true_y'] = sys_cls_data[model.target]
            group_ids, _ = model.group_by_pc1(sys_cls_data,features)
            sys_cls_data['group_id'] = group_ids
            results[name]['cross_val'] =  model.run_cross_validation(sys_cls_data)
                        
            s_d = model.standardize(sys_cls_data, features)
            results[name]['predicted'] = model.model.predict(s_d[model.features])
            
        # add regressors for any modelable structure params 
        for stg_nm in xrsdefs.modelable_structure_settings[struct]:
                stg_header = pop_id+'_'+stg_nm
                stg_labels = list(sys_cls_data[stg_header].unique())
                for stg_label in stg_labels:
                    stg_label_data = sys_cls_data.loc[sys_cls_data[stg_header]==stg_label].copy()
                    for pnm in xrsdefs.structure_params(struct,{stg_nm:stg_label}):
                        param_header = pop_id+'_'+pnm
                        if (sys_cls in regression_models) \
                        and (pop_id in regression_models[sys_cls]) \
                        and (stg_nm in regression_models[sys_cls][pop_id]) \
                        and (stg_label in regression_models[sys_cls][pop_id][stg_nm]) \
                        and (pnm in regression_models[sys_cls][pop_id][stg_nm][stg_label]) \
                        and (regression_models[sys_cls][pop_id][stg_nm][stg_label][pnm].trained):
                            model = regression_models[sys_cls][pop_id][stg_nm][stg_label][pnm]
                            name = sys_cls + "_" + stg_nm + "_" + stg_label + "_" + pnm
                        
                            results[name] = {}
                            results[name]['true_y'] = stg_label_data[model.target]
                            group_ids, _ = model.group_by_pc1(stg_label_data,features)
                            stg_label_data['group_id'] = group_ids
                            results[name]['cross_val'] =  model.run_cross_validation(stg_label_data)
                
                            s_d = model.standardize(stg_label_data, features)
                            results[name]['predicted'] = model.model.predict(s_d[model.features])
                            
        # get all unique form factors for this population
        form_header = pop_id+'_form'
        form_specifiers = list(sys_cls_data[form_header].unique())

        # for each form find additional regression models
        for form_id in form_specifiers:
                form_data = sys_cls_data.loc[data[form_header]==form_id].copy()
                for pnm in xrsdefs.form_factor_params[form_id]:
                    param_header = pop_id+'_'+pnm
                    if (sys_cls in regression_models) \
                    and (pop_id in regression_models[sys_cls]) \
                    and (form_id in regression_models[sys_cls][pop_id]) \
                    and (pnm in regression_models[sys_cls][pop_id][form_id]) \
                    and (regression_models[sys_cls][pop_id][form_id][pnm].trained):
                        model = regression_models[sys_cls][pop_id][form_id][pnm]
                        name = sys_cls + "_" + form_header + "_" + form_id + "_" + pnm
                        results[name] = {}
                        results[name]['true_y'] = form_data[model.target]
                        group_ids, _ = model.group_by_pc1(form_data,features)
                        form_data['group_id'] = group_ids
                        results[name]['cross_val'] =  model.run_cross_validation(form_data)
                        
                        s_d = model.standardize(form_data, features)
                        results[name]['predicted'] =  model.model.predict(s_d[model.features])
                        
                # add regressors for any modelable form factor params 
                for stg_nm in xrsdefs.modelable_form_factor_settings[form_id]:
                    stg_header = pop_id+'_'+stg_nm
                    stg_labels = list(form_data[stg_header].unique())
                    for stg_label in stg_labels:
                        stg_label_data = form_data.loc[form_data[stg_header]==stg_label].copy()
                        for pnm in xrsdefs.additional_form_factor_params(form_id,{stg_nm:stg_label}):
                            param_header = pop_id+'_'+pnm
                            if (sys_cls in regression_models) \
                            and (pop_id in regression_models[sys_cls]) \
                            and (form_id in regression_models[sys_cls][pop_id]) \
                            and (stg_nm in regression_models[sys_cls][pop_id][form_id]) \
                            and (stg_label in regression_models[sys_cls][pop_id][form_id][stg_nm]) \
                            and (pnm in regression_models[sys_cls][pop_id][form_id][stg_nm][stg_label]) \
                            and (regression_models[sys_cls][pop_id][form_id][stg_nm][stg_label][pnm].trained):
                                model = regression_models[sys_cls][pop_id][form_id][stg_nm][stg_label][pnm]
                                name = sys_cls + "_" + form_header + "_" + form_id + "_" + stg_nm + '_' + stg_label + "_" + pnm
                                results[name] = {}
                                results[name]['true_y'] = stg_label_data[model.target]
                                group_ids, _ = model.group_by_pc1(stg_label_data,features)
                                stg_label_data['group_id'] = group_ids
                                results[name]['cross_val'] =  model.run_cross_validation(stg_label_data)
                        
                                s_d = model.standardize(stg_label_data, features)
                                results[name]['predicted'] = model.model.predict(s_d[model.features])                  
    return results