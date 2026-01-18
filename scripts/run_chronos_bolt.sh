#!/bin/bash
# chronos_bolt experiments for all datasets
# Generated from datasets.yaml; 48 dataset/freq combinations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${ENV_NAME:-chronos}"
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
        pip install chronos-forecasting torch datasets gluonts dotenv
    fi
}

setup_conda_env

########################### Nature ###########################
python experiments/chronos_bolt.py --dataset "Water_Quality_Darwin/15T"
python experiments/chronos_bolt.py --dataset "current_velocity/5T"
python experiments/chronos_bolt.py --dataset "current_velocity/10T"
python experiments/chronos_bolt.py --dataset "current_velocity/15T"
python experiments/chronos_bolt.py --dataset "current_velocity/20T"
python experiments/chronos_bolt.py --dataset "current_velocity/H"
python experiments/chronos_bolt.py --dataset "CPHL/15T"
python experiments/chronos_bolt.py --dataset "CPHL/30T"
python experiments/chronos_bolt.py --dataset "CPHL/H"
python experiments/chronos_bolt.py --dataset "Coastal_T_S/5T"
python experiments/chronos_bolt.py --dataset "Coastal_T_S/15T"
python experiments/chronos_bolt.py --dataset "Coastal_T_S/20T"
python experiments/chronos_bolt.py --dataset "Coastal_T_S/H"
python experiments/chronos_bolt.py --dataset "SG_Weather/D"
python experiments/chronos_bolt.py --dataset "SG_PM25/H"
python experiments/chronos_bolt.py --dataset "NE_China_Wind/H"

########################### Energy ###########################
python experiments/chronos_bolt.py --dataset "Australia_Solar/H"
python experiments/chronos_bolt.py --dataset "epf_electricity_price/H"
python experiments/chronos_bolt.py --dataset "OpenElectricity_NEM/5T"
python experiments/chronos_bolt.py --dataset "EWELD_Load/15T"

########################### Transportation ###########################
python experiments/chronos_bolt.py --dataset "SG_Carpark/15T"
python experiments/chronos_bolt.py --dataset "Finland_Traffic/15T"
python experiments/chronos_bolt.py --dataset "Port_Activity/D"
python experiments/chronos_bolt.py --dataset "Port_Activity/W"

########################### Healthcare ###########################
python experiments/chronos_bolt.py --dataset "ECDC_COVID/D"
python experiments/chronos_bolt.py --dataset "ECDC_COVID/W"

########################### Finance ###########################
python experiments/chronos_bolt.py --dataset "Crypto/D"
python experiments/chronos_bolt.py --dataset "US_Term_Structure/B"
python experiments/chronos_bolt.py --dataset "Oil_Price/B"

########################### Economics ###########################
python experiments/chronos_bolt.py --dataset "Job_Claims/W"
python experiments/chronos_bolt.py --dataset "Uncertainty_1M/M"
python experiments/chronos_bolt.py --dataset "Housing_Inventory/M"
python experiments/chronos_bolt.py --dataset "JOLTS/M"
python experiments/chronos_bolt.py --dataset "US_Labor/M"
python experiments/chronos_bolt.py --dataset "Vehicle_Supply/M"
python experiments/chronos_bolt.py --dataset "Auto_Production_SF/M"
python experiments/chronos_bolt.py --dataset "Commodity_Production/M"
python experiments/chronos_bolt.py --dataset "Commodity_Import/M"
python experiments/chronos_bolt.py --dataset "WUI_Global/Q"
python experiments/chronos_bolt.py --dataset "Global_Price/Q"

########################### Sales ###########################
python experiments/chronos_bolt.py --dataset "Vehicle_Sales/M"
python experiments/chronos_bolt.py --dataset "Online_Retail_2_UCI/D"
# python experiments/chronos_bolt.py --dataset "Supply_Chain_Customer/D"
python experiments/chronos_bolt.py --dataset "Supply_Chain_Location/D"

########################## CloudOPS ###########################
python experiments/chronos_bolt.py --dataset "azure2019_D/5T"
python experiments/chronos_bolt.py --dataset "azure2019_I/5T"
python experiments/chronos_bolt.py --dataset "azure2019_U/5T"

########################## Industry ###########################
python experiments/chronos_bolt.py --dataset "Smart_Manufacturing/H"
