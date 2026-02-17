
import traceback
import cobra
import pandas
from typing import Dict

def scale_pseudo_metabolite_stoichiometry(split_reaction_obj: cobra.Reaction, scaling_factor: float) -> Dict[cobra.Metabolite, float]:
    """
    Scales the stoichiometry of pseudo metabolites in a reaction.

    @param split_reaction_obj: cobrapy reaction object
    @param scaling_factor: scaling factor for kcat values (e.g., 1.1 will increase kcat by 10%)
    @return: dictionary of pseudo metabolites and their new stoichiometry
    """

    pseudo_metabolite_dict = {}

    for metabolite_obj in split_reaction_obj.metabolites.keys():
        if metabolite_obj.id.startswith('ENZYME_') or metabolite_obj.id == 'prot_pool':
            stoichiometric_coefficient = split_reaction_obj.metabolites[metabolite_obj]
            pseudo_metabolite_dict[metabolite_obj] = stoichiometric_coefficient/scaling_factor
    
    split_reaction_obj.add_metabolites(pseudo_metabolite_dict, combine = False )

    return pseudo_metabolite_dict

def get_split_reactions_in_direction(reaction: cobra.Reaction, model: cobra.Model) -> list[str]:
    """
    Returns all parallel isoenzymes of a reaction in the same direction (those have the same kcat).

    @param reaction: cobrapy reaction object
    @param model: cobrapy model
    """

     # get direction
    direction = ''
    if '_TG_' in reaction.id:
        direction = reaction.id.split('_TG_')[1]

    # get original reaction id
    original_reaction_id = reaction.id.split('_TG_')[0].split('_GPRSPLIT_')[0]

    split_reactions_in_direction = []

    for r in model.reactions:
        if r.id.split('_TG_')[0].split('_GPRSPLIT_')[0] == original_reaction_id and r.id.endswith(direction):
            split_reactions_in_direction.append(r.id)

    return split_reactions_in_direction

def get_objective_value(model: cobra.Model, objective: str, split_reactions: list[str], scaling_factor: float) -> float:
    """
    Returns the objective value of a model with scaled kcat values.
    
    @param model: cobrapy model
    @param objective: objective reaction id
    @param split_reactions: list of reaction ids to scale
    @param scaling_factor: scaling factor for kcat values (e.g., 1.1 will increase kcat by 10%)
    @return: objective value
    """

    with model:
        for split_reaction in split_reactions:
            split_reaction_obj = model.reactions.get_by_id(split_reaction)
            scale_pseudo_metabolite_stoichiometry(split_reaction_obj, scaling_factor)
        
        try:
            # solution = cobra.flux_analysis.pfba(model)
            # objective_new = solution.fluxes[objective]
            objective_new = model.slim_optimize()
        except:
            print('infeasible solution.')
            objective_new = 0
    
    return objective_new

def get_sensitivity(objective_original: float, model: cobra.Model, objective: str, reaction: cobra.Reaction, split_reactions_in_direction: list[str], relative_step_size: float) -> pandas.DataFrame:
    
    mode = 'centered_difference'

    Z_minus = None
    Z_center = objective_original
    Z_plus = get_objective_value(model, objective, split_reactions_in_direction, 1 + relative_step_size)

    if mode == 'centered_difference':

        Z_minus = get_objective_value(model, objective, split_reactions_in_direction, 1 - relative_step_size)
        sensitivity = (Z_plus - Z_minus) / (2 * Z_center * relative_step_size)
    else:
        sensitivity = (Z_plus - Z_center) / (Z_center * relative_step_size) # "backward" difference
    
    return pandas.DataFrame({'reaction_id': [reaction.id], 'sensitivity': [sensitivity]})

def reaction_wise_sensitivity(model: cobra.Model, objective: str, relative_step_size: float) -> Dict[str, float]:

    """
    Calculates the sensitivity (flux control coefficient) of the objective value with respect to the kcat values of each reaction individually.
    
    @param model: cobrapy model
    @param objective: objective reaction id
    @param relative_step_size: relative step size for varying kcat values (e.g., 0.1 will increase/decrease kcat by 10%)
    @return: dataframe with reaction ids and sensitivities
    """
    
    # original objective value
    model.objective = objective
    solution_original = cobra.flux_analysis.pfba(model)
    objective_original = solution_original.fluxes[objective]

    kcat_sensitivity = pandas.DataFrame(columns=['reaction_id', 'sensitivity'])

    for reaction in model.reactions:

        # ignore delivery reactions
        if reaction.id.startswith('ENZYME_DELIVERY_') or reaction.id.startswith('ER_pool'):
            continue
        
        # check if reaction has pseudo metabolites
        has_pseudo_metabolites = False

        for metabolite in reaction.metabolites.keys():
            if metabolite.id.startswith('ENZYME_') or metabolite.id == 'prot_pool':
                has_pseudo_metabolites = True
                break
        
        # if not, continue with the next reaction
        if not has_pseudo_metabolites:
            continue

        new_row = get_sensitivity(objective_original, model, objective, reaction, [reaction.id], relative_step_size)

        kcat_sensitivity = pandas.concat([kcat_sensitivity, new_row], ignore_index=True)

    return kcat_sensitivity.sort_values(by='sensitivity', ascending=False)

def reaction_wise_sensitivity_all_splits(model: cobra.Model, objective: str, relative_step_size: float) -> Dict[str, float]:

    """
    Calculates the sensitivity (flux control coefficient) of the objective value with respect to the kcat values of all split reactions 
    (kcats of split reactions are coupled).
    
    @param model: cobrapy model
    @param objective: objective reaction id
    @param relative_step_size: relative step size for varying kcat values (e.g., 0.1 will increase/decrease kcat by 10%)
    @return: dataframe with reaction ids and sensitivities
    """
    
    # original objective value
    model.objective = objective
    # solution_original = cobra.flux_analysis.pfba(model)
    # solution_original = model.optimize()
    # objective_original = solution_original.fluxes[objective]
    objective_original = model.slim_optimize()

    kcat_sensitivity = pandas.DataFrame(columns=['reaction_id', 'sensitivity'])

    for reaction in model.reactions:

        # ignore delivery reactions
        if reaction.id.startswith('ENZYME_DELIVERY_') or reaction.id.startswith('ER_pool'):
            continue
        
        # check if reaction has pseudo metabolites
        has_pseudo_metabolites = False

        for metabolite in reaction.metabolites.keys():
            if metabolite.id.startswith('ENZYME_') or metabolite.id == 'prot_pool':
                has_pseudo_metabolites = True
                break
        
        # if not, continue with the next reaction
        if not has_pseudo_metabolites:
            continue
        
        # get splits (parallel isoenzyme reactions) of the reaction in the same direction
        split_reactions_in_direction = get_split_reactions_in_direction(reaction, model)

        new_row = get_sensitivity(objective_original, model, objective, reaction, split_reactions_in_direction, relative_step_size)

        kcat_sensitivity = pandas.concat([kcat_sensitivity, new_row], ignore_index=True)

    return kcat_sensitivity.sort_values(by='sensitivity', ascending=False)

def get_kcat_candidates(model: cobra.Model, objective: str, max_iterations: int, target_objective: float, target_deviation: float, relative_step_size: float, sensitivity_threshold: float) -> pandas.DataFrame:
    """
    Iteratively determines reactions with highest influence 
    on objective and removes them until target 
    objective value is reached. Removed reactions are candidates 
    for calibration.
    
    If the sMOMENT model before kcat calibration cannot reach a 
    desired objective value its restricted due to a set of 
    reactions hitting their protein constraints. Assuming that 
    proteomics measurements are precise, enzyme efficiency (i.e., kcats)
    are too low.

    This function checks the sensitivity of the objecitve value with 
    respect to the kcat values of each reaction. The reactions with
    the highest sensitivity are the most restricting ones. Reactions
    above a sensitivity threshold (0.1%) are therefore removed.
    This results in a new objective with a new set of constraining reactions.
    Therefore this process is repeated until the target objective value 
    is within reach. Deleted reactions are collected and candidates for 
    calibration.

    ("Sensitivity" is used synonymously to the flux control coefficient below)
    
    @param model: cobrapy model
    @param objective: objective reaction id
    @param max_iterations: maximum number of iterations
    @param target_objective: target objective value
    @param target_deviation: target deviation from target objective value
    @param relative_step_size: relative step size for sensitivity calculation
    @param sensitivity_threshold: sensitivity threshold for reaction removal
    @return: dataframe of candidates for calibration
    """

    # relative_step_size = 0.2
    # sensitivity_threshold = 0.001

    model.objective = objective

    removed_enzyme_reactions = {
        'reaction_id': [],
        'matlab_reaction_id': [],
        'original_name': [],
        'iteration': [], 
        'sensitivity': [],
        'max_factor': []
        }
    
    scaled_enzyme_reactions_dict = {}

    target_value_reached = False

    # solution_original = cobra.flux_analysis.pfba(model)
    # solution_original = model.optimize()
    solution_original = model.slim_optimize()
    # objective_original = solution_original.fluxes[objective]
    objective_original = solution_original
    print('orignal objective: ' + str(objective_original))

    current_solution = solution_original

    with model:

        for i in range(0, max_iterations):

            print('Checking sensitivity in iteration ' + str(i))

            # get sensitivity for all reactions
            kcat_sensitivity = reaction_wise_sensitivity_all_splits(model, objective, relative_step_size)

            # delete pseudo metabolites in reactions with highest sensitivity
            for index, reaction_sensitivity in kcat_sensitivity.iterrows():

                if reaction_sensitivity['sensitivity'] < sensitivity_threshold:
                    break

                # get split reactions in one direction
                split_reactions_in_direction = get_split_reactions_in_direction(model.reactions.get_by_id(reaction_sensitivity.loc['reaction_id']), model)

                # initial factor for scaling
                factor = 10

                for split_reaction in split_reactions_in_direction:
                    
                    # if reaction is already in dict and has not been scaled in this iteration, update scaling
                    if split_reaction in scaled_enzyme_reactions_dict and i > scaled_enzyme_reactions_dict[split_reaction]['iteration']:
                        next_factor = scaled_enzyme_reactions_dict[split_reaction]['kcat_factor'] * 10
                        
                        # only scale if factor is below 1000
                        if next_factor < 1000:
                            factor = next_factor
                        else:
                            print(split_reaction + ' is at 100! Imma skip this fella!')
                            continue
                    # ...otherwise just start scaling with 10

                    reaction_object = model.reactions.get_by_id(split_reaction)

                    # scales kcat actually: stoichiometry * 1/factor
                    # scale always by 10, actual factor is tracked by scaled_enzyme_reactions_dict
                    scale_pseudo_metabolite_stoichiometry(reaction_object, 10)
                    
                    scaled_enzyme_reactions_dict[split_reaction] = {'kcat_factor': factor, 'iteration': i}

                # check change in objective
                # try:
                #     current_solution = cobra.flux_analysis.pfba(model)
                # except Exception:
                #     try:
                #         current_solution = cobra.flux_analysis.pfba(model, fraction_of_optimum=0.99)
                #     except Exception:
                #         traceback.print_exc()
                current_solution = model.slim_optimize()
                current_objective = current_solution # current_solution.fluxes[objective]
                
                # add reaction to dataframe
                removed_enzyme_reactions['reaction_id'].append(reaction_sensitivity.loc['reaction_id'])
                removed_enzyme_reactions['matlab_reaction_id'].append('R_' + reaction_sensitivity.loc['reaction_id'])
                removed_enzyme_reactions['original_name'].append(reaction_sensitivity.loc['reaction_id'].split('_TG_')[0].split('_GPRSPLIT_')[0])
                removed_enzyme_reactions['iteration'].append(i)
                removed_enzyme_reactions['sensitivity'].append(reaction_sensitivity.loc['sensitivity'])
                removed_enzyme_reactions['max_factor'].append(factor)

                # print('Removing reaction: ' + reaction_sensitivity.loc['reaction_id'] + ' Resulting objective value ' + str(current_objective))
                print('Scaling kcats in reaction: ' + reaction_sensitivity.loc['reaction_id'] + ' by ' + str(factor) + ' resulting objective:  ' + str(current_objective))

            
            # terminate if objective is near target
            # if current_objective > target_objective - target_deviation:
            #     target_value_reached = True
            #     break
            
            # if target_value_reached:
            #     print('Target value hit within deviation or exceeded.')
            #     break

    max_factor = 10

    # for fold_change_flux in removed_enzyme_reactions['fold_change_flux']:
    for factor in removed_enzyme_reactions['max_factor']:

        if factor > max_factor:
            max_factor = factor
 
    max_factor = max_factor

    print('max change factor: ' + str(max_factor))

    output_df = pandas.DataFrame(data = removed_enzyme_reactions).drop_duplicates(subset='matlab_reaction_id', keep='last')

    return output_df, max_factor