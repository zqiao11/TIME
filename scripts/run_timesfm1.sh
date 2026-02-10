#!/bin/bash
# timesfm1.0 experiments for all datasets
# Generated from datasets.yaml; 50 dataset/freq combinations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${ENV_NAME:-timesfm}"
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
        pip install torch datasets gluonts python-dotenv huggingface_hub pandas numpy utilsforecast
    fi
}

setup_conda_env


########################### Nature ###########################
# python experiments/timesfm1.0.py --dataset "Water_Quality_Darwin/15T"
# python experiments/timesfm1.0.py --dataset "current_velocity/5T"
# python experiments/timesfm1.0.py --dataset "current_velocity/10T"
# python experiments/timesfm1.0.py --dataset "current_velocity/15T"
# python experiments/timesfm1.0.py --dataset "current_velocity/20T"
# python experiments/timesfm1.0.py --dataset "current_velocity/H"
# python experiments/timesfm1.0.py --dataset "CPHL/15T"
# python experiments/timesfm1.0.py --dataset "CPHL/30T"
# python experiments/timesfm1.0.py --dataset "CPHL/H"
# python experiments/timesfm1.0.py --dataset "Coastal_T_S/5T"
# python experiments/timesfm1.0.py --dataset "Coastal_T_S/15T"
# python experiments/timesfm1.0.py --dataset "Coastal_T_S/20T"
# python experiments/timesfm1.0.py --dataset "Coastal_T_S/H"
# python experiments/timesfm1.0.py --dataset "SG_Weather/D"
# python experiments/timesfm1.0.py --dataset "SG_PM25/H"
# python experiments/timesfm1.0.py --dataset "NE_China_Wind/H"

########################### Energy ###########################
python experiments/timesfm1.0.py --dataset "Australia_Solar/H"
# python experiments/timesfm1.0.py --dataset "epf_electricity_price/H"
# python experiments/timesfm1.0.py --dataset "OpenElectricity_NEM/5T"
# python experiments/timesfm1.0.py --dataset "EWELD_Load/15T"

########################### Transportation ###########################
# python experiments/timesfm1.0.py --dataset "SG_Carpark/15T"
# python experiments/timesfm1.0.py --dataset "Finland_Traffic/15T"
python experiments/timesfm1.0.py --dataset "Port_Activity/D"
python experiments/timesfm1.0.py --dataset "Port_Activity/W"

########################### Healthcare ###########################
# python experiments/timesfm1.0.py --dataset "ECDC_COVID/D"
# python experiments/timesfm1.0.py --dataset "ECDC_COVID/W"
python experiments/timesfm1.0.py --dataset "Global_Influenza/W"

########################### Finance ###########################
python experiments/timesfm1.0.py --dataset "Crypto/D"
# python experiments/timesfm1.0.py --dataset "US_Term_Structure/B"
# python experiments/timesfm1.0.py --dataset "Oil_Price/B"

########################### Economics ###########################
python experiments/timesfm1.0.py --dataset "Job_Claims/W"
# python experiments/timesfm1.0.py --dataset "Uncertainty_1M/M"
python experiments/timesfm1.0.py --dataset "Housing_Inventory/M"
# python experiments/timesfm1.0.py --dataset "JOLTS/M"
python experiments/timesfm1.0.py --dataset "US_Labor/M"
# python experiments/timesfm1.0.py --dataset "Vehicle_Supply/M"
# python experiments/timesfm1.0.py --dataset "Auto_Production_SF/M"
# python experiments/timesfm1.0.py --dataset "Commodity_Production/M"
# python experiments/timesfm1.0.py --dataset "Commodity_Import/M"
# python experiments/timesfm1.0.py --dataset "WUI_Global/Q"
# python experiments/timesfm1.0.py --dataset "Global_Price/Q"

########################### Sales ###########################
# python experiments/timesfm1.0.py --dataset "Vehicle_Sales/M"
python experiments/timesfm1.0.py --dataset "Online_Retail_2_UCI/D"
python experiments/timesfm1.0.py --dataset "Supply_Chain_Customer/D"
python experiments/timesfm1.0.py --dataset "Supply_Chain_Location/D"

########################### CloudOPS ###########################
# python experiments/timesfm1.0.py --dataset "azure2019_D/5T"
# python experiments/timesfm1.0.py --dataset "azure2019_I/5T"
# python experiments/timesfm1.0.py --dataset "azure2019_U/5T"

########################### Industry ###########################
# python experiments/timesfm1.0.py --dataset "Smart_Manufacturing/H"
# python experiments/timesfm1.0.py --dataset "MetroPT-3/5T"
