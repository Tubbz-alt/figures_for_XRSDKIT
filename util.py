import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

from xrsdkit import definitions as xrsdefs

import xrsdkit
from xrsdkit.models import load_models, get_regression_models, get_classification_models
load_models('../xrsdkit_modeling/flowreactor_pd_nanoparticles/models')
from xrsdkit.models.predict import system_from_prediction, predict

from xrsdkit.tools.profiler import profile_keys
from xrsdkit.tools.visualization_tools import doPCA

regression_models = get_regression_models()
classification_models = get_classification_models()




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
    regression_models = get_regression_models()
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
                        results[name]['true_y'],results[name]['predicted'],results[name]['cross_val'] = model.train(noise_model_data)
                                
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
            results[name]['true_y'],results[name]['predicted'],results[name]['cross_val'] = model.train(sys_cls_data)
            
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
                            results[name]['true_y'],results[name]['predicted'],results[name]['cross_val'] = model.train(stg_label_data)
                            
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
                        results[name]['true_y'],results[name]['predicted'],results[name]['cross_val'] = model.train(form_data)
                        
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
                                results[name]['true_y'],results[name]['predicted'],results[name]['cross_val'] = model.train(stg_label_data)
                                
    return results

def get_true_xvalid_binary_cls(data):
    classification_models = get_classification_models()
    u_f = []
    features = profile_keys
    results = {}
    all_sys_cls = data['system_class'].tolist()
    data_copy = data.copy()

    for struct_nm in xrsdefs.structure_names:
        model_id = struct_nm+'_binary'
        labels = [struct_nm in sys_cls for sys_cls in all_sys_cls]
        data_copy.loc[:,model_id] = labels

        if ('main_classifiers' in classification_models) \
        and (model_id in classification_models['main_classifiers']) \
        and (classification_models['main_classifiers'][model_id].trained):
            model = classification_models['main_classifiers'][model_id] 
            u_f.extend(model.features)
            name = model_id            
            results[name] = {}
            
            pca = doPCA(model.scaler.transform(data_copy[model.features]),2)
            transformed_data = pca.transform(model.scaler.transform(data_copy[model.features]))
            results[name]['pca1'] = transformed_data[ : , 0]
            results[name]['pca2'] = transformed_data[ : , 1]
            
            results[name]['true_y'] = labels
            group_ids, _ = model.group_by_pc1(data_copy,features)
            data_copy['group_id'] = group_ids
            results[name]['cross_val'] =  model.run_cross_validation(data_copy).to_list()
            
            r = []
            for i in range(len(labels)):
                y_t = labels[i]
                y_p = results[name]['cross_val'][i]
                if y_t== True and y_p== True:
                    r.append(0)#('true_positive')
                elif y_t==False and y_p == False:
                    r.append(1)#('true_negative')
                elif y_t == True and y_p == False:
                    r.append(2)#('false_negative')
                else:
                    r.append(3)#('false_positive')
            results[name]['compared'] = r
    return results, u_f 

def get_true_xvalid_multiclass_cls(data):
    classification_models = get_classification_models()
    features = profile_keys
    results = {}
    u_f = []
    all_sys_cls = data['system_class'].tolist()
           
    all_flag_combs = itertools.product([True,False],repeat=len(xrsdefs.structure_names))
    for flags in all_flag_combs:
        if sum(flags) > 0:
            flag_idx = np.ones(data.shape[0],dtype=bool)
            model_id = ''
            for struct_nm, flag in zip(xrsdefs.structure_names,flags):
                struct_flag_idx = np.array([(struct_nm in sys_cls) == flag for sys_cls in all_sys_cls])
                flag_idx = flag_idx & struct_flag_idx
                if flag:
                    if model_id: model_id += '__'
                    model_id += struct_nm
            # get all samples whose system_class matches the flags
            flag_data = data.loc[flag_idx,:].copy()
            if flag_data.shape[0] > 0: # we have data with these structure flags in the training set
                if ('main_classifiers' in classification_models) \
                and (model_id in classification_models['main_classifiers']) \
                and (classification_models['main_classifiers'][model_id].trained):
                    model = classification_models['main_classifiers'][model_id]
                    u_f.extend(model.features)
                    name = model_id            
                    results[name] = {}

                    pca = doPCA(model.scaler.transform(flag_data[model.features]),2)
                    transformed_data = pca.transform(model.scaler.transform(flag_data[model.features]))
                    results[name]['pca1'] = transformed_data[ : , 0]
                    results[name]['pca2'] = transformed_data[ : , 1]
                    labels = flag_data[model.target].copy().to_list()
                    results[name]['true_y'] = labels
                    group_ids, _ = model.group_by_pc1(flag_data,features)
                    flag_data['group_id'] = group_ids
                    results[name]['cross_val'] =  model.run_cross_validation(flag_data).to_list()         

                    r = []
                    for i in range(len(results[name]['true_y'])):
                        y_t = results[name]['true_y'][i]
                        y_p = results[name]['cross_val'][i]
                        if y_t == y_p:
                            r.append(0)
                        else:
                            r.append(1)
                    results[name]['compared'] = r
                    
    sys_cls_labels = list(data['system_class'].unique())
    # 'unidentified' systems will have no sub-classifiers; drop this label up front 
    if 'unidentified' in sys_cls_labels: sys_cls_labels.remove('unidentified')

    for sys_cls in sys_cls_labels:
        sys_cls_data = data.loc[data['system_class']==sys_cls].copy()
        if (sys_cls in classification_models) \
        and ('noise_model' in classification_models[sys_cls]) \
        and (classification_models[sys_cls]['noise_model'].trained):
            model = classification_models[sys_cls]['noise_model']
            u_f.extend(model.features)
            name = 'noise_model'            
            results[name] = {}

            pca = doPCA(model.scaler.transform(sys_cls_data[model.features]),2)
            transformed_data = pca.transform(model.scaler.transform(sys_cls_data[model.features]))
            results[name]['pca1'] = transformed_data[ : , 0]
            results[name]['pca2'] = transformed_data[ : , 1]
            labels = flag_data[model.target].copy().to_list()
            results[name]['true_y'] = labels
            group_ids, _ = model.group_by_pc1(sys_cls_data,features)
            sys_cls_data['group_id'] = group_ids
            results[name]['cross_val'] =  model.run_cross_validation(sys_cls_data).to_list()         

            r = []
            for i in range(len(results[name]['true_y'])):
                y_t = results[name]['true_y'][i]
                y_p = results[name]['cross_val'][i]
                if y_t == y_p:
                    r.append(0)
                else:
                    r.append(1)
            results[name]['compared'] = r
            
        # each population has some classifiers for form factor and settings
        for ipop, struct in enumerate(sys_cls.split('__')):
            pop_id = 'pop{}'.format(ipop)

            # every population must have a form classifier
            form_header = pop_id+'_form'
            if (sys_cls in classification_models) \
            and (pop_id in classification_models[sys_cls]) \
            and ('form' in classification_models[sys_cls][pop_id]) \
            and (classification_models[sys_cls][pop_id]['form'].trained):
                model = classification_models[sys_cls][pop_id]['form']
                u_f.extend(model.features)
                name = pop_id          
                results[name] = {}

                pca = doPCA(model.scaler.transform(sys_cls_data[model.features]),2)
                transformed_data = pca.transform(model.scaler.transform(sys_cls_data[model.features]))
                results[name]['pca1'] = transformed_data[ : , 0]
                results[name]['pca2'] = transformed_data[ : , 1]
                labels = sys_cls_data[model.target].copy().to_list()
                results[name]['true_y'] = labels
                group_ids, _ = model.group_by_pc1(sys_cls_data,features)
                sys_cls_data['group_id'] = group_ids
                results[name]['cross_val'] =  model.run_cross_validation(sys_cls_data).to_list()         

                r = []
                for i in range(len(results[name]['true_y'])):
                    y_t = results[name]['true_y'][i]
                    y_p = results[name]['cross_val'][i]
                    if y_t == y_p:
                        r.append(0)
                    else:
                        r.append(1)
                results[name]['compared'] = r

            # add classifiers for any model-able structure settings 
            for stg_nm in xrsdefs.modelable_structure_settings[struct]:
                stg_header = pop_id+'_'+stg_nm
                if (sys_cls in classification_models) \
                and (pop_id in classification_models[sys_cls]) \
                and (stg_nm in classification_models[sys_cls][pop_id]) \
                and (classification_models[sys_cls][pop_id][stg_nm].trained):
                    model = classification_models[sys_cls][pop_id][stg_nm]
                    u_f.extend(model.features)
                    name = stg_header      
                    results[name] = {}

                    pca = doPCA(model.scaler.transform(sys_cls_data[model.features]),2)
                    transformed_data = pca.transform(model.scaler.transform(sys_cls_data[model.features]))
                    results[name]['pca1'] = transformed_data[ : , 0]
                    results[name]['pca2'] = transformed_data[ : , 1]
                    labels = flag_data[model.target].copy().to_list()
                    results[name]['true_y'] = labels
                    group_ids, _ = model.group_by_pc1(flag_data,features)
                    sys_cls_data['group_id'] = group_ids
                    results[name]['cross_val'] =  model.run_cross_validation(sys_cls_data).to_list()         

                    r = []
                    for i in range(len(results[name]['true_y'])):
                        y_t = results[name]['true_y'][i]
                        y_p = results[name]['cross_val'][i]
                        if y_t == y_p:
                            r.append(0)
                        else:
                            r.append(1)
                    results[name]['compared'] = r



    return results, u_f