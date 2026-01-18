#!/bin/bash
# ToTo experiments for all datasets
# Generated from datasets.yaml; 48 dataset/freq combinations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${ENV_NAME:-toto}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

setup_conda_env() {
    source "$(conda info --base)/etc/profile.d/conda.sh"

    if conda env list | awk '{print $1}' | grep -x "$ENV_NAME" >/dev/null 2>&1; then
        log_info "Activating existing env: $ENV_NAME"
        conda activate "$ENV_NAME"
    else
        log_info "Creating new env: $ENV_NAME"
        conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
        conda activate "$ENV_NAME"

        log_info "Installing dependencies..."
        pip install toto-ts
        pip install dotenv
    fi
}

setup_conda_env

########################### Nature ###########################
python experiments/toto_model.py --dataset "Water_Quality_Darwin/15T"
python experiments/toto_model.py --dataset "current_velocity/5T"
python experiments/toto_model.py --dataset "current_velocity/10T"
python experiments/toto_model.py --dataset "current_velocity/15T"
python experiments/toto_model.py --dataset "current_velocity/20T"
python experiments/toto_model.py --dataset "current_velocity/H"
python experiments/toto_model.py --dataset "CPHL/15T"
python experiments/toto_model.py --dataset "CPHL/30T"
python experiments/toto_model.py --dataset "CPHL/H"
python experiments/toto_model.py --dataset "Coastal_T_S/5T"
python experiments/toto_model.py --dataset "Coastal_T_S/15T"
python experiments/toto_model.py --dataset "Coastal_T_S/20T"
python experiments/toto_model.py --dataset "Coastal_T_S/H"
python experiments/toto_model.py --dataset "SG_Weather/D"
python experiments/toto_model.py --dataset "SG_PM25/H"
python experiments/toto_model.py --dataset "NE_China_Wind/H"

########################### Energy ###########################
python experiments/toto_model.py --dataset "Australia_Solar/H"
python experiments/toto_model.py --dataset "epf_electricity_price/H"
python experiments/toto_model.py --dataset "OpenElectricity_NEM/5T"
python experiments/toto_model.py --dataset "EWELD_Load/15T"

########################### Transportation ###########################
python experiments/toto_model.py --dataset "SG_Carpark/15T"
python experiments/toto_model.py --dataset "Finland_Traffic/15T"
python experiments/toto_model.py --dataset "Port_Activity/D"
python experiments/toto_model.py --dataset "Port_Activity/W"

########################### Healthcare ###########################
python experiments/toto_model.py --dataset "ECDC_COVID/D"
python experiments/toto_model.py --dataset "ECDC_COVID/W"

########################### Finance ###########################
python experiments/toto_model.py --dataset "Crypto/D"
python experiments/toto_model.py --dataset "US_Term_Structure/B" --context-length 1024
python experiments/toto_model.py --dataset "Oil_Price/B" --context-length 2048

########################### Economics ###########################
python experiments/toto_model.py --dataset "Job_Claims/W"
python experiments/toto_model.py --dataset "Uncertainty_1M/M"
python experiments/toto_model.py --dataset "Housing_Inventory/M"
python experiments/toto_model.py --dataset "JOLTS/M"
python experiments/toto_model.py --dataset "US_Labor/M"
python experiments/toto_model.py --dataset "Vehicle_Supply/M"
python experiments/toto_model.py --dataset "Auto_Production_SF/M"
python experiments/toto_model.py --dataset "Commodity_Production/M"
python experiments/toto_model.py --dataset "Commodity_Import/M"
python experiments/toto_model.py --dataset "WUI_Global/Q" --context-length 1024
python experiments/toto_model.py --dataset "Global_Price/Q" --context-length 1024

########################### Sales ###########################
python experiments/toto_model.py --dataset "Vehicle_Sales/M" --context-length 1024
python experiments/toto_model.py --dataset "Online_Retail_2_UCI/D"
python experiments/toto_model.py --dataset "Supply_Chain_Customer/D" --context-length 512
python experiments/toto_model.py --dataset "Supply_Chain_Location/D" --context-length 512

########################### CloudOPS ###########################
python experiments/toto_model.py --dataset "azure2019_D/5T"
python experiments/toto_model.py --dataset "azure2019_I/5T"
python experiments/toto_model.py --dataset "azure2019_U/5T"

########################### Industry ###########################
python experiments/toto_model.py --dataset "Smart_Manufacturing/H"
