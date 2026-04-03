#!/bin/bash
# patchtst-fm experiments for all datasets
# Generated from datasets.yaml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${ENV_NAME:-granite-tsfm}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
GRANITE_TSFM_REPO="${GRANITE_TSFM_REPO:-https://github.com/ibm-granite/granite-tsfm.git}"
GRANITE_TSFM_DIR="${GRANITE_TSFM_DIR:-$ROOT_DIR/experiments/granite-tsfm}"

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1"
}

setup_granite_tsfm_repo() {
    if [ ! -d "$GRANITE_TSFM_DIR/.git" ]; then
        log_info "Cloning Granite-TSFM repo..."
        mkdir -p "$(dirname "$GRANITE_TSFM_DIR")"
        git clone "$GRANITE_TSFM_REPO" "$GRANITE_TSFM_DIR"
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
        pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
        pip install datasets gluonts dotenv
        # pip install datasets gluonts dotenv torch

        cd "$GRANITE_TSFM_DIR"
        pip install ".[notebooks]"
        cd "$ROOT_DIR"
    fi
}

setup_granite_tsfm_repo
setup_conda_env



########################### Nature ###########################
python experiments/patchtst_fm.py --dataset "Water_Quality_Darwin/15T"
python experiments/patchtst_fm.py --dataset "current_velocity/5T"
python experiments/patchtst_fm.py --dataset "current_velocity/10T"
python experiments/patchtst_fm.py --dataset "current_velocity/15T"
python experiments/patchtst_fm.py --dataset "current_velocity/20T"
python experiments/patchtst_fm.py --dataset "current_velocity/H"
python experiments/patchtst_fm.py --dataset "CPHL/15T"
python experiments/patchtst_fm.py --dataset "CPHL/30T"
python experiments/patchtst_fm.py --dataset "CPHL/H"
python experiments/patchtst_fm.py --dataset "Coastal_T_S/5T"
python experiments/patchtst_fm.py --dataset "Coastal_T_S/15T"
python experiments/patchtst_fm.py --dataset "Coastal_T_S/20T"
python experiments/patchtst_fm.py --dataset "Coastal_T_S/H"
python experiments/patchtst_fm.py --dataset "SG_Weather/D"
python experiments/patchtst_fm.py --dataset "SG_PM25/H"
python experiments/patchtst_fm.py --dataset "NE_China_Wind/H"

########################### Energy ###########################
python experiments/patchtst_fm.py --dataset "Australia_Solar/H"
python experiments/patchtst_fm.py --dataset "epf_electricity_price/H"
python experiments/patchtst_fm.py --dataset "OpenElectricity_NEM/5T"
python experiments/patchtst_fm.py --dataset "EWELD_Load/15T"

########################### Transportation ###########################
python experiments/patchtst_fm.py  --dataset "SG_Carpark/15T"
python experiments/patchtst_fm.py --dataset "Finland_Traffic/15T"
python experiments/patchtst_fm.py --dataset "Port_Activity/D"
python experiments/patchtst_fm.py --dataset "Port_Activity/W"

########################### Healthcare ###########################
python experiments/patchtst_fm.py  --dataset "ECDC_COVID/D"
python experiments/patchtst_fm.py  --dataset "ECDC_COVID/W"
python experiments/patchtst_fm.py  --dataset "Global_Influenza/W"

########################### Finance ###########################
python experiments/patchtst_fm.py  --dataset "Crypto/D"
python experiments/patchtst_fm.py  --dataset "US_Term_Structure/B"
python experiments/patchtst_fm.py  --dataset "Oil_Price/B"

########################### Economics ###########################
python experiments/patchtst_fm.py  --dataset "Job_Claims/W"
python experiments/patchtst_fm.py  --dataset "Uncertainty_1M/M"
python experiments/patchtst_fm.py  --dataset "Housing_Inventory/M"
python experiments/patchtst_fm.py  --dataset "JOLTS/M"
python experiments/patchtst_fm.py  --dataset "US_Labor/M"
python experiments/patchtst_fm.py  --dataset "Vehicle_Supply/M"
python experiments/patchtst_fm.py  --dataset "Auto_Production_SF/M"
python experiments/patchtst_fm.py  --dataset "Commodity_Production/M"
python experiments/patchtst_fm.py  --dataset "Commodity_Import/M"
python experiments/patchtst_fm.py  --dataset "WUI_Global/Q"
python experiments/patchtst_fm.py  --dataset "Global_Price/Q"

########################### Sales ###########################
python experiments/patchtst_fm.py  --dataset "Vehicle_Sales/M"
python experiments/patchtst_fm.py  --dataset "Online_Retail_2_UCI/D"
python experiments/patchtst_fm.py  --dataset "Supply_Chain_Customer/D"
python experiments/patchtst_fm.py  --dataset "Supply_Chain_Location/D"

########################### CloudOPS ###########################
python experiments/patchtst_fm.py  --dataset "azure2019_D/5T"
python experiments/patchtst_fm.py  --dataset "azure2019_I/5T"
python experiments/patchtst_fm.py  --dataset "azure2019_U/5T"

########################### Industry ###########################
python experiments/patchtst_fm.py  --dataset "Smart_Manufacturing/H"
python experiments/patchtst_fm.py  --dataset "MetroPT-3/5T"
