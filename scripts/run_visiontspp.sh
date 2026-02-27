#!/bin/bash
# visiontspp experiments for all datasets
# Generated from datasets.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${ENV_NAME:-visiontspp}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

ensure_conda_env() {
    if ! command -v conda >/dev/null 2>&1; then
        log_error "Conda not found. Please install conda first."
        exit 1
    fi

    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"

    if conda env list | awk '{print $1}' | grep -x "$ENV_NAME" >/dev/null 2>&1; then
        log_info "Activating conda env: $ENV_NAME"
        conda activate "$ENV_NAME"
    else
        log_info "Creating conda env: $ENV_NAME"
        conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
        conda activate "$ENV_NAME"
        log_info "Installing dependencies..."
        pip install visionts datasets gluonts dotenv torch
    fi
}

ensure_conda_env



########################### Nature ###########################
python experiments/visiontspp.py --dataset "Water_Quality_Darwin/15T"
python experiments/visiontspp.py --dataset "current_velocity/5T"
python experiments/visiontspp.py --dataset "current_velocity/10T"
python experiments/visiontspp.py --dataset "current_velocity/15T"
python experiments/visiontspp.py --dataset "current_velocity/20T"
python experiments/visiontspp.py --dataset "current_velocity/H"
python experiments/visiontspp.py --dataset "CPHL/15T"
python experiments/visiontspp.py --dataset "CPHL/30T"
python experiments/visiontspp.py --dataset "CPHL/H"
python experiments/visiontspp.py --dataset "Coastal_T_S/5T"
python experiments/visiontspp.py --dataset "Coastal_T_S/15T"
python experiments/visiontspp.py --dataset "Coastal_T_S/20T"
python experiments/visiontspp.py --dataset "Coastal_T_S/H"
python experiments/visiontspp.py --dataset "SG_Weather/D"
python experiments/visiontspp.py --dataset "SG_PM25/H"
python experiments/visiontspp.py --dataset "NE_China_Wind/H"

########################### Energy ###########################
python experiments/visiontspp.py --dataset "Australia_Solar/H"
python experiments/visiontspp.py --dataset "epf_electricity_price/H"
python experiments/visiontspp.py --dataset "OpenElectricity_NEM/5T"
python experiments/visiontspp.py --dataset "EWELD_Load/15T"

########################### Transportation ###########################
python experiments/visiontspp.py --dataset "SG_Carpark/15T"
python experiments/visiontspp.py --dataset "Finland_Traffic/15T"
python experiments/visiontspp.py --dataset "Port_Activity/D"
python experiments/visiontspp.py --dataset "Port_Activity/W"

########################### Healthcare ###########################
python experiments/visiontspp.py --dataset "ECDC_COVID/D"
python experiments/visiontspp.py --dataset "ECDC_COVID/W"
python experiments/visiontspp.py --dataset "Global_Influenza/W"

########################### Finance ###########################
python experiments/visiontspp.py --dataset "Crypto/D"
python experiments/visiontspp.py --dataset "US_Term_Structure/B"
python experiments/visiontspp.py --dataset "Oil_Price/B"

########################### Economics ###########################
python experiments/visiontspp.py --dataset "Job_Claims/W"
python experiments/visiontspp.py --dataset "Uncertainty_1M/M"
python experiments/visiontspp.py --dataset "Housing_Inventory/M"
python experiments/visiontspp.py --dataset "JOLTS/M"
python experiments/visiontspp.py --dataset "US_Labor/M"
python experiments/visiontspp.py --dataset "Vehicle_Supply/M"
python experiments/visiontspp.py --dataset "Auto_Production_SF/M"
python experiments/visiontspp.py --dataset "Commodity_Production/M"
python experiments/visiontspp.py --dataset "Commodity_Import/M"
python experiments/visiontspp.py --dataset "WUI_Global/Q"
python experiments/visiontspp.py --dataset "Global_Price/Q"

########################### Sales ###########################
python experiments/visiontspp.py --dataset "Vehicle_Sales/M"
python experiments/visiontspp.py --dataset "Online_Retail_2_UCI/D"
python experiments/visiontspp.py --dataset "Supply_Chain_Customer/D"
python experiments/visiontspp.py --dataset "Supply_Chain_Location/D"

########################### CloudOPS ###########################
python experiments/visiontspp.py --dataset "azure2019_D/5T"
python experiments/visiontspp.py --dataset "azure2019_I/5T"
python experiments/visiontspp.py --dataset "azure2019_U/5T"

########################### Industry ###########################
python experiments/visiontspp.py --dataset "Smart_Manufacturing/H"
python experiments/visiontspp.py --dataset "MetroPT-3/5T"

