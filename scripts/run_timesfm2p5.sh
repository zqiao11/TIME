#!/bin/bash
# timesfm2.5 experiments for all datasets
# Generated from datasets.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${ENV_NAME:-timesfm2p5}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TIMESFM_REPO="${TIMESFM_REPO:-https://github.com/google-research/timesfm.git}"
TIMESFM_DIR="${TIMESFM_DIR:-$ROOT_DIR/experiments/timesfm}"
HF_HOME="${HF_HOME:-$ROOT_DIR/experiments/.cache/huggingface}"


log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

setup_timesfm_repo() {
    if [ ! -d "$TIMESFM_DIR/.git" ]; then
        log_info "Cloning TimesFM-2.5 repo..."
        mkdir -p "$(dirname "$TIMESFM_DIR")"
        git clone "$TIMESFM_REPO" "$TIMESFM_DIR"
    fi
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
        cd "$TIMESFM_DIR"
        pip install -e .
        pip install datasets gluonts dotenv torch
        cd "$ROOT_DIR"
    fi
}


setup_timesfm_repo
setup_conda_env

export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

cd "$ROOT_DIR"

########################### Nature ###########################
python experiments/timesfm2.5.py --dataset "Water_Quality_Darwin/15T"
python experiments/timesfm2.5.py --dataset "current_velocity/5T"
python experiments/timesfm2.5.py --dataset "current_velocity/10T"
python experiments/timesfm2.5.py --dataset "current_velocity/15T"
python experiments/timesfm2.5.py --dataset "current_velocity/20T"
python experiments/timesfm2.5.py --dataset "current_velocity/H"
python experiments/timesfm2.5.py --dataset "CPHL/15T"
python experiments/timesfm2.5.py --dataset "CPHL/30T"
python experiments/timesfm2.5.py --dataset "CPHL/H"
python experiments/timesfm2.5.py --dataset "Coastal_T_S/5T"
python experiments/timesfm2.5.py --dataset "Coastal_T_S/15T"
python experiments/timesfm2.5.py --dataset "Coastal_T_S/20T"
python experiments/timesfm2.5.py --dataset "Coastal_T_S/H"
python experiments/timesfm2.5.py --dataset "SG_Weather/D"
python experiments/timesfm2.5.py --dataset "SG_PM25/H"
python experiments/timesfm2.5.py --dataset "NE_China_Wind/H"

########################### Energy ###########################
python experiments/timesfm2.5.py --dataset "Australia_Solar/H"
python experiments/timesfm2.5.py --dataset "epf_electricity_price/H"
python experiments/timesfm2.5.py --dataset "OpenElectricity_NEM/5T"
python experiments/timesfm2.5.py --dataset "EWELD_Load/15T"

########################### Transportation ###########################
python experiments/timesfm2.5.py --dataset "SG_Carpark/15T"
python experiments/timesfm2.5.py --dataset "Finland_Traffic/15T"
python experiments/timesfm2.5.py --dataset "Port_Activity/D"
python experiments/timesfm2.5.py --dataset "Port_Activity/W"

########################### Healthcare ###########################
python experiments/timesfm2.5.py --dataset "ECDC_COVID/D"
python experiments/timesfm2.5.py --dataset "ECDC_COVID/W"
python experiments/timesfm2.5.py --dataset "Global_Influenza/W"

########################### Finance ###########################
python experiments/timesfm2.5.py --dataset "Crypto/D"
python experiments/timesfm2.5.py --dataset "US_Term_Structure/B"
python experiments/timesfm2.5.py --dataset "Oil_Price/B"

########################### Economics ###########################
python experiments/timesfm2.5.py --dataset "Job_Claims/W"
python experiments/timesfm2.5.py --dataset "Uncertainty_1M/M"
python experiments/timesfm2.5.py --dataset "Housing_Inventory/M"
python experiments/timesfm2.5.py --dataset "JOLTS/M"
python experiments/timesfm2.5.py --dataset "US_Labor/M"
python experiments/timesfm2.5.py --dataset "Vehicle_Supply/M"
python experiments/timesfm2.5.py --dataset "Auto_Production_SF/M"
python experiments/timesfm2.5.py --dataset "Commodity_Production/M"
python experiments/timesfm2.5.py --dataset "Commodity_Import/M"
python experiments/timesfm2.5.py --dataset "WUI_Global/Q"
python experiments/timesfm2.5.py --dataset "Global_Price/Q"

########################### Sales ###########################
python experiments/timesfm2.5.py --dataset "Vehicle_Sales/M"
python experiments/timesfm2.5.py --dataset "Online_Retail_2_UCI/D"
python experiments/timesfm2.5.py --dataset "Supply_Chain_Customer/D"
python experiments/timesfm2.5.py --dataset "Supply_Chain_Location/D"

########################### CloudOPS ###########################
python experiments/timesfm2.5.py --dataset "azure2019_D/5T"
python experiments/timesfm2.5.py --dataset "azure2019_I/5T"
python experiments/timesfm2.5.py --dataset "azure2019_U/5T"

########################### Industry ###########################
python experiments/timesfm2.5.py --dataset "Smart_Manufacturing/H"
python experiments/timesfm2.5.py --dataset "MetroPT-3/5T"
