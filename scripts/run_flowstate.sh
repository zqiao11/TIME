#!/bin/bash
# FlowState experiments for all datasets
# Generated from datasets.yaml; 50 dataset/freq combinations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${ENV_NAME:-flowstate}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TSFM_REPO="${TSFM_REPO:-https://github.com/ibm-granite/granite-tsfm.git}"
TSFM_PUBLIC_DIR="${TSFM_PUBLIC_DIR:-$ROOT_DIR/granite-tsfm}"

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
        pip install torch datasets gluonts python-dotenv pandas numpy
        if [ -n "$TSFM_PUBLIC_DIR" ]; then
            pip install -e "$TSFM_PUBLIC_DIR"
        else
            pip install tsfm-public
        fi
    fi
}

setup_tsfm_repo() {
    if [ -z "$TSFM_PUBLIC_DIR" ]; then
        log_info "TSFM_PUBLIC_DIR not set; skipping clone."
        return
    fi
    if [ -d "$TSFM_PUBLIC_DIR/.git" ]; then
        log_info "granite-tsfm repo already exists: $TSFM_PUBLIC_DIR"
        return
    fi

    log_info "Cloning granite-tsfm repo into: $TSFM_PUBLIC_DIR"
    mkdir -p "$(dirname "$TSFM_PUBLIC_DIR")"
    git clone "$TSFM_REPO" "$TSFM_PUBLIC_DIR"
}

setup_tsfm_repo
setup_conda_env

########################### Nature ###########################
# python experiments/flowstate.py --dataset "Water_Quality_Darwin/15T"
# python experiments/flowstate.py --dataset "current_velocity/5T"
# python experiments/flowstate.py --dataset "current_velocity/10T"
# python experiments/flowstate.py --dataset "current_velocity/15T"
# python experiments/flowstate.py --dataset "current_velocity/20T"
# python experiments/flowstate.py --dataset "current_velocity/H"
# python experiments/flowstate.py --dataset "CPHL/15T"
# python experiments/flowstate.py --dataset "CPHL/30T"
# python experiments/flowstate.py --dataset "CPHL/H"
# python experiments/flowstate.py --dataset "Coastal_T_S/5T"
# python experiments/flowstate.py --dataset "Coastal_T_S/15T"
# python experiments/flowstate.py --dataset "Coastal_T_S/20T"
# python experiments/flowstate.py --dataset "Coastal_T_S/H"
# python experiments/flowstate.py --dataset "SG_Weather/D"
# python experiments/flowstate.py --dataset "SG_PM25/H"
# python experiments/flowstate.py --dataset "NE_China_Wind/H"

########################### Energy ###########################
python experiments/flowstate.py --dataset "Australia_Solar/H"
# python experiments/flowstate.py --dataset "epf_electricity_price/H"
# python experiments/flowstate.py --dataset "OpenElectricity_NEM/5T"
# python experiments/flowstate.py --dataset "EWELD_Load/15T"

########################### Transportation ###########################
# python experiments/flowstate.py --dataset "SG_Carpark/15T"
# python experiments/flowstate.py --dataset "Finland_Traffic/15T"
python experiments/flowstate.py --dataset "Port_Activity/D"
python experiments/flowstate.py --dataset "Port_Activity/W"

########################### Healthcare ###########################
# python experiments/flowstate.py --dataset "ECDC_COVID/D"
# python experiments/flowstate.py --dataset "ECDC_COVID/W"
python experiments/flowstate.py --dataset "Global_Influenza/W"

########################### Finance ###########################
python experiments/flowstate.py --dataset "Crypto/D"
# python experiments/flowstate.py --dataset "US_Term_Structure/B"
# python experiments/flowstate.py --dataset "Oil_Price/B"

########################### Economics ###########################
python experiments/flowstate.py --dataset "Job_Claims/W"
# python experiments/flowstate.py --dataset "Uncertainty_1M/M"
python experiments/flowstate.py --dataset "Housing_Inventory/M"
# python experiments/flowstate.py --dataset "JOLTS/M"
python experiments/flowstate.py --dataset "US_Labor/M"
# python experiments/flowstate.py --dataset "Vehicle_Supply/M"
# python experiments/flowstate.py --dataset "Auto_Production_SF/M"
# python experiments/flowstate.py --dataset "Commodity_Production/M"
# python experiments/flowstate.py --dataset "Commodity_Import/M"
# python experiments/flowstate.py --dataset "WUI_Global/Q"
# python experiments/flowstate.py --dataset "Global_Price/Q"

########################### Sales ###########################
# python experiments/flowstate.py --dataset "Vehicle_Sales/M"
python experiments/flowstate.py --dataset "Online_Retail_2_UCI/D"
python experiments/flowstate.py --dataset "Supply_Chain_Customer/D"
python experiments/flowstate.py --dataset "Supply_Chain_Location/D"

########################### CloudOPS ###########################
# python experiments/flowstate.py --dataset "azure2019_D/5T"
# python experiments/flowstate.py --dataset "azure2019_I/5T"
# python experiments/flowstate.py --dataset "azure2019_U/5T"

########################### Industry ###########################
# python experiments/flowstate.py --dataset "Smart_Manufacturing/H"
# python experiments/flowstate.py --dataset "MetroPT-3/5T"
