function []  = mcs_analysis_run_2019_06_25()
    %% Load models :D
    [originalModel, ~] = CNAsbmlModel2MFNetwork('./.../iJO1366_saved_by_cobrapy_and_separated_reversible_reactions_SHUT_DOWN_SCENARIO.xml');
    % This is iJO1366*
    [smomentModel, ~] = CNAsbmlModel2MFNetwork('./.../iJOstar.xml');

    %% Set-up models :D
    % Protein pool \o/
    smomentModel.reacMax(strmatch('R_ER_pool_TG_', smomentModel.reacID, 'exact')) = .095;
    % Glucose uptake \o/
    originalModel.reacMin(strmatch('R_EX_glc__D_e', originalModel.reacID, 'exact')) = -15;
    smomentModel.reacMin(strmatch('R_EX_glc__D_e', smomentModel.reacID, 'exact')) = -1000;
    % No oxygen :O
    originalModel.reacMin(strmatch('R_EX_o2_e', originalModel.reacID, 'exact')) = 0;
    smomentModel.reacMin(strmatch('R_EX_o2_e', smomentModel.reacID, 'exact')) = 0;

    %% General variables :D
    growthRateReactionName = 'R_BIOMASS_Ec_iJO1366_core_53p95M';
    substrateRateReactionName = 'R_EX_glc__D_e';
    maxMCSsize = 6;


    %% Ethanol :D
    minGrowthRate = .1;
    minYield = 1.4;
    productRateReactionName = 'R_EX_etoh_e';

    % original model :-)
    outputFilename = './iJO1366star/iJO1366star_MCS_analysis_scripts/results_ethanol/original_6.txt';
    p_mcs_analysis(originalModel,...
                   outputFilename,...
                   growthRateReactionName,...
                   minGrowthRate,...
                   substrateRateReactionName,...
                   productRateReactionName,...
                   minYield,...
                   maxMCSsize)

    % geckoed model :-)
    outputFilename = './iJO1366star/iJO1366star_MCS_analysis_scripts/results_ethanol/geckoed_6.txt';
    p_mcs_analysis(smomentModel,...
                   outputFilename,...
                   growthRateReactionName,...
                   minGrowthRate,...
                   substrateRateReactionName,...
                   productRateReactionName,...
                   minYield,...
                   maxMCSsize)

    %% Leucine :D
    minGrowthRate = .1;
    minYield = .2;
    productRateReactionName = 'R_EX_leu__L_e';

    % original model :-)
    outputFilename = './iJO1366star/iJOstar_MCS_analysis_scripts/original_6.txt';
    p_mcs_analysis(originalModel,...
                   outputFilename,...
                   growthRateReactionName,...
                   minGrowthRate,...
                   substrateRateReactionName,...
                   productRateReactionName,...
                   minYield,...
                   maxMCSsize)

    % geckoed model :-)
    outputFilename = './iJO1366star/iJOstar_MCS_analysis_scripts/geckoed_6.txt';
    p_mcs_analysis(smomentModel,...
                   outputFilename,...
                   growthRateReactionName,...
                   minGrowthRate,...
                   substrateRateReactionName,...
                   productRateReactionName,...
                   minYield,...
                   maxMCSsize)

    %% Valine :D
    minGrowthRate = .1;
    minYield = .3;
    productRateReactionName = 'R_EX_val__L_e';

    % original model :-)
    outputFilename = './iJO1366star/iJOstar_MCS_analysis_scripts/original_6.txt';
    p_mcs_analysis(originalModel,...
                   outputFilename,...
                   growthRateReactionName,...
                   minGrowthRate,...
                   substrateRateReactionName,...
                   productRateReactionName,...
                   minYield,...
                   maxMCSsize)

    % geckoed model :-)
    outputFilename = './iJO1366star/iJOstar_MCS_analysis_scripts/geckoed_6.txt';
    p_mcs_analysis(smomentModel,...
                   outputFilename,...
                   growthRateReactionName,...
                   minGrowthRate,...
                   substrateRateReactionName,...
                   productRateReactionName,...
                   minYield,...
                   maxMCSsize)

    %% Succinate :D
    minGrowthRate = .1;
    minYield = 1;
    productRateReactionName = 'R_EX_succ_e';

    % original model :-)
    outputFilename = 'iJO1366star/iJO1366star_MCS_analysis_scripts/results_succinate/original_6.txt';
    p_mcs_analysis(originalModel,...
                   outputFilename,...
                   growthRateReactionName,...
                   minGrowthRate,...
                   substrateRateReactionName,...
                   productRateReactionName,...
                   minYield,...
                   maxMCSsize)

    % geckoed model :-)
    outputFilename = 'iJO1366star/iJO1366star_MCS_analysis_scripts/results_succinate/geckoed_6.txt';
    p_mcs_analysis(smomentModel,...
                   outputFilename,...
                   growthRateReactionName,...
                   minGrowthRate,...
                   substrateRateReactionName,...
                   productRateReactionName,...
                   minYield,...
                   maxMCSsize)
end