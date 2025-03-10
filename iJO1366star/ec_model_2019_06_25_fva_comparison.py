#!/usr/bin/env python3
#
# Copyright 2019 PSB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""analysis_fba_comparison.py


"""

import z_add_path

from autopacmen.submodules.fva_comparison import fva_comparison_with_sbml

SBML_ORIGINAL: str = (
    "./iJO1366star/ec_model_2019_06_25_input/iJO1366_saved_by_cobrapy_and_separated_reversible_reactions.xml"
)
SBML_SMOMENT: str = (
    "./iJO1366star/ec_model_2019_06_25_output/iJO1366_sMOMENT_2019_06_25.xml"
)
OUTPUT_FOLDER = "./iJO1366star/ec_model_2019_06_25_output"
OBJECTIVE: str = "BIOMASS_Ec_iJO1366_core_53p95M"

fva_comparison_with_sbml(SBML_ORIGINAL, SBML_SMOMENT, OBJECTIVE)
